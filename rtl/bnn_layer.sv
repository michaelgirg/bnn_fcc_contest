`default_nettype none

// =============================================================================
// Module: bnn_layer 
// =============================================================================
// Key changes vs original:
//   1. DISPATCH+COMPUTE merged: RAM address registered one cycle ahead,
//      RAM output feeds NPs directly — saves one cycle per beat.
//   2. weight_addr_base_r per-NP array eliminated: address computed from
//      group_idx_r * INPUT_BEATS + beat_idx_r directly.
//   3. WAIT_OUT only entered on the final beat; intermediate beats go
//      straight back to the next beat dispatch.
//   4. Threshold cache retained: read once per group, reused across beats.
//   5. Operand isolation retained: inactive NP slots receive zeroed inputs.
// =============================================================================
module bnn_layer #(
    parameter int INPUTS       = 784,
    parameter int NEURONS      = 256,
    parameter int PW           = 8,
    parameter int PN           = 8,
    parameter bit OUTPUT_LAYER = 0,

    // Derived — do not override
    parameter int COUNT_WIDTH           = $clog2(INPUTS + 1),
    parameter int INPUT_BEATS           = (INPUTS + PW - 1) / PW,
    parameter int CFG_NEURON_WIDTH      = (NEURONS > 1) ? $clog2(NEURONS) : 1,
    parameter int CFG_WEIGHT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1
) (
    input wire logic clk,
    input wire logic rst,

    input wire logic              input_load,
    input wire logic [INPUTS-1:0] input_vector,
    input wire logic              start,

    output logic done,
    output logic busy,
    output logic output_valid,

    output logic [            NEURONS-1:0] activations_out,
    output logic [NEURONS*COUNT_WIDTH-1:0] popcounts_out,

    input  wire logic                             cfg_write_en,
    input  wire logic [     CFG_NEURON_WIDTH-1:0] cfg_neuron_idx,
    input  wire logic [CFG_WEIGHT_ADDR_WIDTH-1:0] cfg_weight_addr,
    input  wire logic [                   PW-1:0] cfg_weight_data,
    input  wire logic [          COUNT_WIDTH-1:0] cfg_threshold_data,
    input  wire logic                             cfg_threshold_write,
    output logic                                  cfg_ready
);

    // ---------------------------------------------------------------------------
    // Derived local parameters
    // ---------------------------------------------------------------------------
    localparam int NEURON_GROUPS = (NEURONS + PN - 1) / PN;
    localparam int INPUT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1;
    localparam int WEIGHT_DEPTH = INPUT_BEATS * NEURON_GROUPS;
    localparam int WEIGHT_ADDR_WIDTH = (WEIGHT_DEPTH > 1) ? $clog2(WEIGHT_DEPTH) : 1;
    localparam int GROUP_WIDTH = (NEURON_GROUPS > 1) ? $clog2(NEURON_GROUPS) : 1;
    localparam int PADDED_INPUTS = INPUT_BEATS * PW;

    localparam int LAST_GROUP_ACTIVE = (NEURONS % PN == 0) ? PN : (NEURONS % PN);
    localparam logic [PN-1:0] LAST_GROUP_MASK = PN'((1 << LAST_GROUP_ACTIVE) - 1);
    localparam logic [GROUP_WIDTH-1:0] LAST_GROUP_IDX = GROUP_WIDTH'(NEURON_GROUPS - 1);
    localparam logic [INPUT_ADDR_WIDTH-1:0] LAST_BEAT_IDX = INPUT_ADDR_WIDTH'(INPUT_BEATS - 1);

    // NP pipeline depth — fixed at 3 stages (Stage A, B, C in neuron_processor)
    localparam int NP_PIPELINE_DEPTH = 3;

    // ---------------------------------------------------------------------------
    // FSM encoding
    // ---------------------------------------------------------------------------
    // BEAT   : register next beat's address, feed current beat to NPs
    // WAIT_OUT: wait for NP pipeline to drain after last beat of a group
    typedef enum logic [1:0] {
        IDLE,
        BEAT,
        WAIT_OUT
    } state_t;

    // ---------------------------------------------------------------------------
    // Helper functions
    // ---------------------------------------------------------------------------
    function automatic int get_np_idx(input int neuron_idx);
        return neuron_idx % PN;
    endfunction

    function automatic int get_group_idx(input int neuron_idx);
        return neuron_idx / PN;
    endfunction

    // ---------------------------------------------------------------------------
    // FSM and control registers
    // ---------------------------------------------------------------------------
    state_t                                   state_r;
    logic   [                GROUP_WIDTH-1:0] group_idx_r;
    logic   [           INPUT_ADDR_WIDTH-1:0] beat_idx_r;
    logic   [                         PN-1:0] np_active_r;
    logic                                     done_r;
    logic                                     output_valid_r;
    logic                                     collect_done;
    logic                                     all_groups_done;
    logic                                     all_np_done;

    // Drain counter: counts NP_PIPELINE_DEPTH cycles in WAIT_OUT
    // so we don't need to check individual valid_out signals for timing
    logic   [$clog2(NP_PIPELINE_DEPTH+1)-1:0] drain_cnt_r;

    assign all_groups_done = (group_idx_r == LAST_GROUP_IDX);

    // ---------------------------------------------------------------------------
    // Input buffer
    // ---------------------------------------------------------------------------
    logic [           PW-1:0] input_buffer        [INPUT_BEATS];
    logic [PADDED_INPUTS-1:0] input_vector_padded;

    assign input_vector_padded = {{(PADDED_INPUTS - INPUTS) {1'b0}}, input_vector};

    always_ff @(posedge clk) begin
        if (input_load && !busy) begin
            for (int i = 0; i < INPUT_BEATS; i++) input_buffer[i] <= input_vector_padded[i*PW+:PW];
        end
    end

    // ---------------------------------------------------------------------------
    // Weight and threshold RAMs
    // ---------------------------------------------------------------------------
    logic [         PW-1:0] weight_rams   [PN][ WEIGHT_DEPTH];
    logic [COUNT_WIDTH-1:0] threshold_rams[PN][NEURON_GROUPS];

    always_ff @(posedge clk) begin
        if (!busy) begin
            automatic int np_idx = get_np_idx(int'(cfg_neuron_idx));
            automatic int grp_idx = get_group_idx(int'(cfg_neuron_idx));
            automatic int local_addr = grp_idx * INPUT_BEATS + int'(cfg_weight_addr);

            if (cfg_threshold_write && !cfg_write_en) threshold_rams[np_idx][grp_idx] <= cfg_threshold_data;
            else if (cfg_write_en && !cfg_threshold_write) weight_rams[np_idx][local_addr] <= cfg_weight_data;
        end
    end

    // ---------------------------------------------------------------------------
    // Registered weight read address
    // ---------------------------------------------------------------------------
    // We register the RAM read address rather than the RAM output data.
    // This gives synthesis freedom on the RAM output path and saves one
    // pipeline stage vs the original DISPATCH->COMPUTE two-state approach [5].
    // The address points to the beat that will be consumed NEXT cycle.
    logic [WEIGHT_ADDR_WIDTH-1:0] weight_rd_addr_r [PN];
    logic [ INPUT_ADDR_WIDTH-1:0] input_rd_addr_r;

    // NP inputs — driven directly from RAM outputs (combinational read)
    logic [               PW-1:0] np_weight_in     [PN];
    logic [               PW-1:0] np_input_in;
    logic [      COUNT_WIDTH-1:0] np_threshold_in  [PN];
    logic                         np_valid_in;
    logic                         np_last;

    // Threshold cache — loaded once per group on beat 0
    logic [      COUNT_WIDTH-1:0] threshold_cache_r[PN];

    // NP valid/last: asserted in BEAT state, uses registered beat_idx to
    // determine if this is the last beat of the group
    assign np_valid_in = (state_r == BEAT);
    assign np_last     = (state_r == BEAT) && (beat_idx_r == LAST_BEAT_IDX);

    // Combinational RAM reads using registered addresses
    always_comb begin
        np_input_in = input_buffer[input_rd_addr_r];
        for (int i = 0; i < PN; i++) begin
            np_weight_in[i]    = np_active_r[i] ? weight_rams[i][weight_rd_addr_r[i]] : '0;
            np_threshold_in[i] = threshold_cache_r[i];
        end
    end

    // Register addresses and threshold cache
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < PN; i++) begin
                weight_rd_addr_r[i]  <= '0;
                threshold_cache_r[i] <= '0;
            end
            input_rd_addr_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    if (start) begin
                        // Pre-load addresses for beat 0 of group 0
                        input_rd_addr_r <= '0;
                        for (int i = 0; i < PN; i++) begin
                            weight_rd_addr_r[i]  <= WEIGHT_ADDR_WIDTH'(i == i ? 0 : 0);  // beat 0, group 0
                            threshold_cache_r[i] <= threshold_rams[i][0];
                        end
                    end
                end

                BEAT: begin
                    if (beat_idx_r == LAST_BEAT_IDX) begin
                        // Last beat of current group — preload addresses for beat 0 of next group
                        automatic int next_group = int'(group_idx_r) + 1;
                        input_rd_addr_r <= '0;
                        for (int i = 0; i < PN; i++) begin
                            weight_rd_addr_r[i] <= WEIGHT_ADDR_WIDTH'(next_group * INPUT_BEATS);
                            if (next_group < NEURON_GROUPS)
                                threshold_cache_r[i] <= threshold_rams[i][next_group];
                        end
                    end else begin
                        // Advance to next beat within the same group
                        automatic int next_beat = int'(beat_idx_r) + 1;
                        input_rd_addr_r <= INPUT_ADDR_WIDTH'(next_beat);
                        for (int i = 0; i < PN; i++)
                        weight_rd_addr_r[i] <= weight_rd_addr_r[i] + WEIGHT_ADDR_WIDTH'(1);
                    end
                end

                default: ;
            endcase
        end
    end

    // ---------------------------------------------------------------------------
    // NP active mask
    // ---------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            np_active_r <= '0;
        end else if (state_r == IDLE && start) begin
            np_active_r <= (NEURONS >= PN) ? {PN{1'b1}} : LAST_GROUP_MASK;
        end else if (state_r == WAIT_OUT && drain_cnt_r == NP_PIPELINE_DEPTH[($clog2(
                NP_PIPELINE_DEPTH+1
            ))-1:0]) begin
            if (group_idx_r == LAST_GROUP_IDX - GROUP_WIDTH'(1)) np_active_r <= LAST_GROUP_MASK;
            else np_active_r <= {PN{1'b1}};
        end
    end

    // ---------------------------------------------------------------------------
    // Neuron processor instances
    // ---------------------------------------------------------------------------
    logic [         PN-1:0] np_valid_out;
    logic [         PN-1:0] np_activation;
    logic [COUNT_WIDTH-1:0] np_popcount   [PN];

    assign all_np_done = &(np_valid_out | ~np_active_r);

    generate
        for (genvar i = 0; i < PN; i++) begin : gen_nps
            neuron_processor #(
                .PW               (PW),
                .MAX_NEURON_INPUTS(INPUTS),
                .OUTPUT_LAYER     (OUTPUT_LAYER)
            ) u_np (
                .clk         (clk),
                .rst         (rst),
                .valid_in    (np_valid_in && np_active_r[i]),
                .last        (np_last),
                .x           (np_active_r[i] ? np_input_in : '0),
                .w           (np_active_r[i] ? np_weight_in[i] : '0),
                .threshold   (np_threshold_in[i]),
                .valid_out   (np_valid_out[i]),
                .activation  (np_activation[i]),
                .popcount_out(np_popcount[i])
            );
        end
    endgenerate

    // ---------------------------------------------------------------------------
    // Output collection
    // ---------------------------------------------------------------------------
    logic [    NEURONS-1:0] activations_buffer;
    logic [COUNT_WIDTH-1:0] popcounts_buffer   [NEURONS];

    always_ff @(posedge clk) begin
        if (rst) begin
            activations_buffer <= '0;
            popcounts_buffer   <= '{default: '0};
            collect_done       <= 1'b0;
        end else begin
            collect_done <= 1'b0;

            if (state_r == IDLE && start) begin
                activations_buffer <= '0;
                popcounts_buffer   <= '{default: '0};
            end else if (state_r == WAIT_OUT && all_np_done) begin
                for (int i = 0; i < PN; i++) begin
                    if (np_valid_out[i]) begin
                        automatic int neuron_idx = int'(group_idx_r) * PN + i;
                        if (neuron_idx < NEURONS) begin
                            popcounts_buffer[neuron_idx] <= np_popcount[i];
                            if (OUTPUT_LAYER) activations_buffer[neuron_idx] <= 1'b0;
                            else activations_buffer[neuron_idx] <= np_activation[i];
                        end
                    end
                end
                if (all_groups_done) collect_done <= 1'b1;
            end
        end
    end

    // ---------------------------------------------------------------------------
    // Main FSM
    // ---------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state_r        <= IDLE;
            group_idx_r    <= '0;
            beat_idx_r     <= '0;
            done_r         <= 1'b0;
            output_valid_r <= 1'b0;
            drain_cnt_r    <= '0;
        end else begin
            done_r <= 1'b0;

            if (collect_done) begin
                output_valid_r <= 1'b1;
                done_r         <= 1'b1;
            end else if (state_r == IDLE && start) begin
                output_valid_r <= 1'b0;
            end

            case (state_r)
                IDLE: begin
                    if (start) begin
                        state_r     <= BEAT;
                        group_idx_r <= '0;
                        beat_idx_r  <= '0;
                        drain_cnt_r <= '0;
                    end
                end

                BEAT: begin
                    if (beat_idx_r == LAST_BEAT_IDX) begin
                        // Last beat of this group — drain the NP pipeline
                        state_r     <= WAIT_OUT;
                        beat_idx_r  <= '0;
                        drain_cnt_r <= '0;
                    end else begin
                        beat_idx_r <= beat_idx_r + INPUT_ADDR_WIDTH'(1);
                    end
                end

                WAIT_OUT: begin
                    // Count NP_PIPELINE_DEPTH cycles then collect results
                    if (all_np_done) begin
                        if (all_groups_done) begin
                            state_r <= IDLE;
                        end else begin
                            group_idx_r <= group_idx_r + GROUP_WIDTH'(1);
                            state_r     <= BEAT;
                            drain_cnt_r <= '0;
                        end
                    end else begin
                        drain_cnt_r <= drain_cnt_r + 1'b1;
                    end
                end

                default: state_r <= IDLE;
            endcase
        end
    end

    // ---------------------------------------------------------------------------
    // Output assignments
    // ---------------------------------------------------------------------------
    assign busy            = (state_r != IDLE);
    assign done            = done_r;
    assign cfg_ready       = !busy;
    assign output_valid    = output_valid_r;
    assign activations_out = activations_buffer;

    always_comb begin
        for (int i = 0; i < NEURONS; i++) popcounts_out[i*COUNT_WIDTH+:COUNT_WIDTH] = popcounts_buffer[i];
    end

endmodule
`default_nettype wire
