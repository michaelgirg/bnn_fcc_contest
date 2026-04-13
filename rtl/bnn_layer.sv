`default_nettype none
// =============================================================================
// Module: bnn_layer (FIXED for BRAM inference and timing)
// =============================================================================
module bnn_layer #(
    parameter int INPUTS                = 784,
    parameter int NEURONS               = 256,
    parameter int PW                    = 8,
    parameter int PN                    = 8,
    parameter bit OUTPUT_LAYER          = 0,
    // Derived — do not override
    parameter int COUNT_WIDTH           = $clog2(INPUTS + 1),
    parameter int INPUT_BEATS           = (INPUTS + PW - 1) / PW,
    parameter int CFG_NEURON_WIDTH      = (NEURONS > 1) ? $clog2(NEURONS) : 1,
    parameter int CFG_WEIGHT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1
) (
    input  wire logic                             clk,
    input  wire logic                             rst,
    input  wire logic                             input_load,
    input  wire logic [               INPUTS-1:0] input_vector,
    input  wire logic                             start,
    output logic                                  done,
    output logic                                  busy,
    output logic                                  output_valid,
    output logic      [              NEURONS-1:0] activations_out,
    output logic      [  NEURONS*COUNT_WIDTH-1:0] popcounts_out,
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
    localparam int NP_PIPELINE_DEPTH = 3;

    // ---------------------------------------------------------------------------
    // FSM encoding
    // ---------------------------------------------------------------------------
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
            for (int i = 0; i < INPUT_BEATS; i++) 
                input_buffer[i] <= input_vector_padded[i*PW+:PW];
        end
    end

    // ===========================================================================
    // CRITICAL FIX #1: Convert 3D arrays to separate 2D arrays with BRAM attributes
    // ===========================================================================
    generate
        for (genvar i = 0; i < PN; i++) begin : gen_memories
            // Separate weight RAM for each NP with BRAM inference attribute
            (* ram_style = "block" *) logic [PW-1:0] weight_ram [WEIGHT_DEPTH];
            
            // Separate threshold RAM for each NP with BRAM inference attribute
            (* ram_style = "block" *) logic [COUNT_WIDTH-1:0] threshold_ram [NEURON_GROUPS];
        end
    endgenerate

    // ---------------------------------------------------------------------------
    // Configuration write logic (SIMPLIFIED)
    // ---------------------------------------------------------------------------
    logic [CFG_NEURON_WIDTH-1:0] cfg_np_idx_r;
    logic [GROUP_WIDTH-1:0]      cfg_grp_idx_r;
    logic [WEIGHT_ADDR_WIDTH-1:0] cfg_local_addr_r;

    // Pre-compute configuration indices in combinational logic
    always_comb begin
        cfg_np_idx_r = CFG_NEURON_WIDTH'(get_np_idx(int'(cfg_neuron_idx)));
        cfg_grp_idx_r = GROUP_WIDTH'(get_group_idx(int'(cfg_neuron_idx)));
        cfg_local_addr_r = WEIGHT_ADDR_WIDTH'(int'(cfg_grp_idx_r) * INPUT_BEATS + int'(cfg_weight_addr));
    end

    // Configuration writes (only when idle)
    generate
        for (genvar i = 0; i < PN; i++) begin : gen_cfg_write
            always_ff @(posedge clk) begin
                if (!busy) begin
                    if (cfg_threshold_write && !cfg_write_en && (cfg_np_idx_r == i)) begin
                        gen_memories[i].threshold_ram[cfg_grp_idx_r] <= cfg_threshold_data;
                    end else if (cfg_write_en && !cfg_threshold_write && (cfg_np_idx_r == i)) begin
                        gen_memories[i].weight_ram[cfg_local_addr_r] <= cfg_weight_data;
                    end
                end
            end
        end
    endgenerate

    // ===========================================================================
    // CRITICAL FIX #2: Add registered memory read stage
    // ===========================================================================
    
    // Address registers
    logic [WEIGHT_ADDR_WIDTH-1:0] weight_rd_addr_r [PN];
    logic [ INPUT_ADDR_WIDTH-1:0] input_rd_addr_r;
    
    // Memory output registers (critical for BRAM timing)
    logic [         PW-1:0] weight_mem_out_r [PN];
    logic [COUNT_WIDTH-1:0] threshold_mem_out_r [PN];
    
    // NP inputs (one more pipeline stage)
    logic [         PW-1:0] np_weight_in     [PN];
    logic [         PW-1:0] np_input_in;
    logic [COUNT_WIDTH-1:0] np_threshold_in  [PN];
    logic                   np_valid_in;
    logic                   np_last;

    // Read from memories with registered outputs
    generate
        for (genvar i = 0; i < PN; i++) begin : gen_mem_read
            always_ff @(posedge clk) begin
                weight_mem_out_r[i] <= gen_memories[i].weight_ram[weight_rd_addr_r[i]];
                threshold_mem_out_r[i] <= gen_memories[i].threshold_ram[group_idx_r];
            end
        end
    endgenerate

    // Pipeline to NP inputs
    always_ff @(posedge clk) begin
        for (int i = 0; i < PN; i++) begin
            np_weight_in[i] <= np_active_r[i] ? weight_mem_out_r[i] : '0;
            np_threshold_in[i] <= threshold_mem_out_r[i];
        end
        np_input_in <= input_buffer[input_rd_addr_r];
    end

    assign np_valid_in = (state_r == BEAT);
    assign np_last     = (state_r == BEAT) && (beat_idx_r == LAST_BEAT_IDX);

    // ---------------------------------------------------------------------------
    // Address generation (unchanged)
    // ---------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < PN; i++) begin
                weight_rd_addr_r[i] <= '0;
            end
            input_rd_addr_r <= '0;
        end else begin
            case (state_r)
                IDLE: begin
                    if (start) begin
                        input_rd_addr_r <= '0;
                        for (int i = 0; i < PN; i++) begin
                            weight_rd_addr_r[i] <= WEIGHT_ADDR_WIDTH'(0);
                        end
                    end
                end
                BEAT: begin
                    if (beat_idx_r == LAST_BEAT_IDX) begin
                        automatic int next_group = int'(group_idx_r) + 1;
                        input_rd_addr_r <= '0;
                        for (int i = 0; i < PN; i++) begin
                            weight_rd_addr_r[i] <= WEIGHT_ADDR_WIDTH'(next_group * INPUT_BEATS);
                        end
                    end else begin
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
        end else if (state_r == WAIT_OUT && drain_cnt_r == NP_PIPELINE_DEPTH[($clog2(NP_PIPELINE_DEPTH+1))-1:0]) begin
            if (group_idx_r == LAST_GROUP_IDX - GROUP_WIDTH'(1)) 
                np_active_r <= LAST_GROUP_MASK;
            else 
                np_active_r <= {PN{1'b1}};
        end
    end

    // ---------------------------------------------------------------------------
    // Neuron processor instances (unchanged)
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
    // Output collection (unchanged)
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
                            if (OUTPUT_LAYER) 
                                activations_buffer[neuron_idx] <= 1'b0;
                            else 
                                activations_buffer[neuron_idx] <= np_activation[i];
                        end
                    end
                end
                if (all_groups_done) collect_done <= 1'b1;
            end
        end
    end

    // ---------------------------------------------------------------------------
    // Main FSM (unchanged)
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
                        state_r     <= WAIT_OUT;
                        beat_idx_r  <= '0;
                        drain_cnt_r <= '0;
                    end else begin
                        beat_idx_r <= beat_idx_r + INPUT_ADDR_WIDTH'(1);
                    end
                end
                WAIT_OUT: begin
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
    // Output assignments (unchanged)
    // ---------------------------------------------------------------------------
    assign busy            = (state_r != IDLE);
    assign done            = done_r;
    assign cfg_ready       = !busy;
    assign output_valid    = output_valid_r;
    assign activations_out = activations_buffer;

    always_comb begin
        for (int i = 0; i < NEURONS; i++) 
            popcounts_out[i*COUNT_WIDTH+:COUNT_WIDTH] = popcounts_buffer[i];
    end

endmodule
`default_nettype wire