`default_nettype none
module bnn_layer #(
    parameter int INPUTS                = 784,
    parameter int NEURONS               = 256,
    parameter int PW                    = 8,
    parameter int PN                    = 8,
    parameter bit OUTPUT_LAYER          = 0,
    parameter int COUNT_WIDTH           = $clog2(INPUTS + 1),
    parameter int INPUT_BEATS           = (INPUTS + PW - 1) / PW,
    parameter int CFG_NEURON_WIDTH      = (NEURONS > 1) ? $clog2(NEURONS) : 1,
    parameter int CFG_WEIGHT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1
) (
    input wire logic clk,
    input wire logic rst,

    // Load a full input vector into the local input buffer before starting
    input wire logic              input_load,
    input wire logic [INPUTS-1:0] input_vector,

    // Start processing the currently loaded input against all configured neurons
    input wire logic start,

    // Status outputs
    output logic done,
    output logic busy,

    // Hidden-layer output bits, one per neuron
    output logic [NEURONS-1:0] activations_out,

    // Raw popcounts, mostly useful for output layers or debugging
    output logic [NEURONS*COUNT_WIDTH-1:0] popcounts_out,

    // Goes high when outputs are ready/valid
    output logic output_valid,

    // Simple configuration port:
    // write one neuron's weight beat or threshold at a time
    input  wire logic                             cfg_write_en,
    input  wire logic [     CFG_NEURON_WIDTH-1:0] cfg_neuron_idx,
    input  wire logic [CFG_WEIGHT_ADDR_WIDTH-1:0] cfg_weight_addr,
    input  wire logic [                   PW-1:0] cfg_weight_data,
    input  wire logic [          COUNT_WIDTH-1:0] cfg_threshold_data,
    input  wire logic                             cfg_threshold_write,
    output logic                                  cfg_ready
);

    // ========================================================================
    // Parameters and Local Parameters
    // ========================================================================
    // NEURON_GROUPS:
    //   We only have PN neuron processors, but NEURONS may be much larger.
    //   So we execute the neurons in groups of size PN.
    //
    // Example:
    //   NEURONS = 32, PN = 4  ->  NEURON_GROUPS = 8
    //   Group 0 handles neurons 0..3
    //   Group 1 handles neurons 4..7
    //   ...
    localparam int NEURON_GROUPS = (NEURONS + PN - 1) / PN;

    // Number of beats needed to feed all INPUTS bits when PW bits are processed/cycle
    //
    // Example:
    //   INPUTS = 32, PW = 8 -> INPUT_BEATS = 4
    localparam int INPUT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1;

    // Each NP stores weights for every group it may execute.
    // For one NP:
    //   depth = INPUT_BEATS per neuron * number of groups
    localparam int WEIGHT_DEPTH = INPUT_BEATS * NEURON_GROUPS;
    localparam int WEIGHT_ADDR_WIDTH = (WEIGHT_DEPTH > 1) ? $clog2(WEIGHT_DEPTH) : 1;

    // Width needed to index which neuron group we are currently processing
    localparam int GROUP_WIDTH = (NEURON_GROUPS > 1) ? $clog2(NEURON_GROUPS) : 1;

    // We pad the input vector so it always splits cleanly into INPUT_BEATS chunks of PW bits
    localparam int PADDED_INPUTS = INPUT_BEATS * PW;

    // If NEURONS is not a multiple of PN, the last group has fewer than PN valid neurons
    localparam int LAST_GROUP_ACTIVE = (NEURONS % PN == 0) ? PN : (NEURONS % PN);

    // Bit mask telling us which NPs are active in the final partial group
    localparam logic [PN-1:0] LAST_GROUP_MASK = (1 << LAST_GROUP_ACTIVE) - 1;

    // Convenience constants for detecting the final group and final beat
    localparam logic [GROUP_WIDTH-1:0] LAST_GROUP_IDX = GROUP_WIDTH'(NEURON_GROUPS - 1);
    localparam logic [INPUT_ADDR_WIDTH-1:0] LAST_BEAT_IDX = INPUT_ADDR_WIDTH'(INPUT_BEATS - 1);

    // ========================================================================
    // Helper Functions
    // ========================================================================
    // These two helper functions define how a global neuron index maps
    // onto the physical NP memories.
    //
    // Mapping rule:
    //   np_idx    = neuron_idx % PN
    //   group_idx = neuron_idx / PN
    //
    // Example with PN=4:
    //   neuron 0  -> np0, group0
    //   neuron 1  -> np1, group0
    //   neuron 2  -> np2, group0
    //   neuron 3  -> np3, group0
    //   neuron 4  -> np0, group1
    //   neuron 5  -> np1, group1
    //   ...
    function automatic int get_np_idx(input int neuron_idx);
        return neuron_idx % PN;
    endfunction

    function automatic int get_group_idx(input int neuron_idx);
        return neuron_idx / PN;
    endfunction

    // ========================================================================
    // FSM States
    // ========================================================================
    // IDLE:
    //   Waiting for start
    //
    // DISPATCH:
    //   Read the next input chunk + weight beat + threshold from local memories
    //   and register them into stage1
    //
    // COMPUTE:
    //   Present that registered beat to all NPs for one cycle
    //
    // WAIT_OUT:
    //   After the last beat of a neuron group is sent, wait for all NPs to
    //   produce valid outputs, then collect them
    typedef enum logic [1:0] {
        IDLE,
        DISPATCH,
        COMPUTE,
        WAIT_OUT
    } state_t;

    state_t                        state_r;

    // ========================================================================
    // Control Signals
    // ========================================================================
    // group_idx_r:
    //   Which group of neurons we are currently processing
    //
    // beat_idx_r:
    //   Which PW-bit chunk of the input/weights we are currently processing
    //
    // np_active_r:
    //   Which NPs are active for this group
    //   Usually all 1s, except maybe the last group if NEURONS%PN != 0
    logic   [     GROUP_WIDTH-1:0] group_idx_r;
    logic   [INPUT_ADDR_WIDTH-1:0] beat_idx_r;
    logic   [              PN-1:0] np_active_r;

    // Registered done/output-valid flags
    logic                          done_r;
    logic                          output_valid_r;

    // Goes high for one cycle when the final group's outputs have been collected
    logic                          collect_done;

    // Convenience conditions
    logic                          all_groups_done;
    logic                          all_beats_done;
    logic                          all_np_done;
    logic                          np_valid_in;
    logic                          np_last;

    assign all_beats_done  = (beat_idx_r == LAST_BEAT_IDX);
    assign all_groups_done = (group_idx_r == LAST_GROUP_IDX);

    // ========================================================================
    // Input Buffer
    // ========================================================================
    // We store the full input vector locally, split into INPUT_BEATS chunks.
    //
    // Example:
    //   INPUTS=32, PW=8
    //   input_buffer[0] = bits 7:0
    //   input_buffer[1] = bits 15:8
    //   input_buffer[2] = bits 23:16
    //   input_buffer[3] = bits 31:24
    logic [           PW-1:0] input_buffer        [INPUT_BEATS];
    logic [PADDED_INPUTS-1:0] input_vector_padded;

    // Pad with zeros if INPUTS is not an exact multiple of PW
    assign input_vector_padded = {{(PADDED_INPUTS - INPUTS) {1'b0}}, input_vector};

    always_ff @(posedge clk) begin
        if (input_load && !busy) begin
            for (int i = 0; i < INPUT_BEATS; i++) begin
                input_buffer[i] <= input_vector_padded[i*PW+:PW];
            end
        end
    end

    // ========================================================================
    // Weight and Threshold RAMs
    // ========================================================================
    // Each physical NP has its own weight RAM and threshold RAM.
    //
    // weight_rams[np][addr]:
    //   Holds all PW-bit weight beats for every neuron that NP may execute.
    //
    // threshold_rams[np][group]:
    //   Holds one threshold per neuron group for that NP.
    //
    // Example:
    //   PN=4, INPUT_BEATS=4
    //   neuron 5 -> np1, group1
    //   its weight beats go into weight_rams[1][4..7]
    //   its threshold goes into threshold_rams[1][1]
    logic [         PW-1:0] weight_rams   [PN][ WEIGHT_DEPTH];
    logic [COUNT_WIDTH-1:0] threshold_rams[PN][NEURON_GROUPS];

    always_ff @(posedge clk) begin
        if (cfg_write_en && !busy) begin
            automatic int np_idx = get_np_idx(cfg_neuron_idx);
            automatic int group_idx = get_group_idx(cfg_neuron_idx);

            // local_addr chooses the correct block of INPUT_BEATS entries
            // inside this NP's RAM for the target neuron group
            automatic int local_addr = group_idx * INPUT_BEATS + int'(cfg_weight_addr);

            if (cfg_threshold_write) begin
                threshold_rams[np_idx][group_idx] <= cfg_threshold_data;
            end else begin
                weight_rams[np_idx][local_addr] <= cfg_weight_data;
            end
        end
    end

    // ========================================================================
    // Pipeline Registers
    // ========================================================================
    // weight_addr_base_r[i]:
    //   Base address in NP i's weight RAM for the current neuron group.
    //
    // Then the current beat is selected as:
    //   weight_addr_base_r[i] + beat_idx_r
    logic [WEIGHT_ADDR_WIDTH-1:0] weight_addr_base_r   [PN];

    // Stage 1 registers:
    // These capture the input chunk, weight beat, threshold, and active mask
    // for the current beat when the FSM is in DISPATCH.
    //
    // One cycle later, COMPUTE presents them to the NPs.
    logic [               PW-1:0] weight_data_stage1   [PN];
    logic [      COUNT_WIDTH-1:0] threshold_data_stage1[PN];
    logic [               PW-1:0] input_chunk_stage1;
    logic                         last_stage1;
    logic [               PN-1:0] np_active_stage1;

    always_ff @(posedge clk) begin
        if (state_r == DISPATCH) begin
            // Read the current PW-bit input chunk
            input_chunk_stage1 <= input_buffer[beat_idx_r];

            // Mark whether this is the final beat for the neuron
            last_stage1 <= (beat_idx_r == LAST_BEAT_IDX);

            // Snapshot which NPs are active for this group
            np_active_stage1 <= np_active_r;

            // Read one weight beat and one threshold per NP
            for (int i = 0; i < PN; i++) begin
                weight_data_stage1[i] <= weight_rams[i][weight_addr_base_r[i]+beat_idx_r];
                threshold_data_stage1[i] <= threshold_rams[i][group_idx_r];
            end
        end
    end

    // ========================================================================
    // Stage 2: Fanout / Distribution to NPs
    // ========================================================================
    // This stage is combinational in the current version.
    //
    // All NPs receive the same input chunk for a given beat,
    // but each NP receives its own weight beat and threshold.
    logic [         PW-1:0] input_chunk_replicated[PN];
    logic [         PW-1:0] weight_data_stage2    [PN];
    logic [COUNT_WIDTH-1:0] threshold_data_stage2 [PN];
    logic                   last_stage2;
    logic [         PN-1:0] np_active_stage2;

    always_comb begin
        for (int i = 0; i < PN; i++) begin
            input_chunk_replicated[i] = input_chunk_stage1;
            weight_data_stage2[i]     = weight_data_stage1[i];
            threshold_data_stage2[i]  = threshold_data_stage1[i];
        end

        last_stage2      = last_stage1;
        np_active_stage2 = np_active_stage1;
    end

    // ========================================================================
    // NP Active Control
    // ========================================================================
    // Normally all PN processors are active.
    //
    // The only time fewer are active is when the very last neuron group
    // is partial, meaning NEURONS is not an exact multiple of PN.
    always_ff @(posedge clk) begin
        if (rst) begin
            np_active_r <= '0;
        end else if (state_r == IDLE && start) begin
            // At the beginning, group 0 is always full unless NEURONS < PN
            np_active_r <= (NEURONS >= PN) ? {PN{1'b1}} : LAST_GROUP_MASK;
        end else if (state_r == WAIT_OUT && all_np_done) begin
            // When moving to the next group, decide if it is a full group
            // or the final partial group
            if (group_idx_r == LAST_GROUP_IDX - GROUP_WIDTH'(1)) begin
                np_active_r <= LAST_GROUP_MASK;
            end else begin
                np_active_r <= {PN{1'b1}};
            end
        end
    end

    // ========================================================================
    // Neuron Processors
    // ========================================================================
    // Each NP receives:
    //   - the same input chunk x for this beat
    //   - its own weight beat w
    //   - its own threshold
    //
    // valid_in:
    //   tells the NP that this beat is real and should be accumulated
    //
    // last:
    //   tells the NP this is the final beat for the current neuron, so it
    //   should finish accumulation and produce valid_out
    logic [         PN-1:0] np_valid_out;
    logic [         PN-1:0] np_activation;
    logic [COUNT_WIDTH-1:0] np_popcount   [PN];

    // We are done waiting when every active NP has asserted valid_out.
    // Inactive NPs are ignored by OR-ing with ~np_active_stage2.
    assign all_np_done = &(np_valid_out | ~np_active_stage2);

    // NPs consume one beat during COMPUTE
    assign np_valid_in = (state_r == COMPUTE);

    // last is only meaningful on the final beat during COMPUTE
    assign np_last = (state_r == COMPUTE) && last_stage2;

    generate
        for (genvar i = 0; i < PN; i++) begin : gen_nps
            neuron_processor #(
                .PW               (PW),
                .MAX_NEURON_INPUTS(INPUTS),
                .OUTPUT_LAYER     (OUTPUT_LAYER)
            ) u_np (
                .clk         (clk),
                .rst         (rst),
                .valid_in    (np_valid_in && np_active_stage2[i]),
                .last        (np_last),
                .x           (input_chunk_replicated[i]),
                .w           (weight_data_stage2[i]),
                .threshold   (threshold_data_stage2[i]),
                .valid_out   (np_valid_out[i]),
                .activation  (np_activation[i]),
                .popcount_out(np_popcount[i])
            );
        end
    endgenerate

    // ========================================================================
    // Output Buffers
    // ========================================================================
    // Once a neuron group finishes, we store each NP's result into a global
    // output buffer indexed by the true neuron number.
    //
    // neuron_idx = group_idx_r * PN + i
    //
    // That reconstructs the original global neuron ordering.
    logic [    NEURONS-1:0] activations_buffer;
    logic [COUNT_WIDTH-1:0] popcounts_buffer   [NEURONS];

    always_ff @(posedge clk) begin
        if (rst) begin
            activations_buffer <= '0;
            popcounts_buffer   <= '{default: '0};
            collect_done       <= 1'b0;
        end else begin
            // Default: only pulse collect_done for one clock when final collection happens
            collect_done <= 1'b0;

            if (state_r == IDLE && start) begin
                // Starting a new inference, so clear prior outputs
                activations_buffer <= '0;
                popcounts_buffer   <= '{default: '0};

            end else if (state_r == WAIT_OUT && all_np_done) begin
                // Store results from whichever NPs were active in this group
                for (int i = 0; i < PN; i++) begin
                    if (np_valid_out[i]) begin
                        automatic int neuron_idx = int'(group_idx_r) * PN + i;

                        if (neuron_idx < NEURONS) begin
                            popcounts_buffer[neuron_idx] <= np_popcount[i];

                            if (OUTPUT_LAYER) begin
                                // Output layer uses raw popcount; hidden layer uses activation bit
                                activations_buffer[neuron_idx] <= 1'b0;
                            end else begin
                                activations_buffer[neuron_idx] <= np_activation[i];
                            end
                        end
                    end
                end

                // If this was the final group, tell the FSM outputs are now complete
                if (all_groups_done) begin
                    collect_done <= 1'b1;
                end
            end
        end
    end

    // ========================================================================
    // Main FSM
    // ========================================================================
    // High-level processing flow:
    //
    // 1) IDLE:
    //      wait for start
    //
    // 2) DISPATCH:
    //      read current beat from input buffer and NP RAMs into stage1 regs
    //
    // 3) COMPUTE:
    //      feed those registered values into the NPs for one cycle
    //      if not last beat, increment beat_idx and repeat
    //      if last beat, go wait for NP results
    //
    // 4) WAIT_OUT:
    //      wait until all active NPs assert valid_out
    //      if more groups remain, move to next group and repeat
    //      if final group finished, assert done/output_valid and return IDLE
    always_ff @(posedge clk) begin
        if (rst) begin
            state_r            <= IDLE;
            group_idx_r        <= '0;
            beat_idx_r         <= '0;
            done_r             <= 1'b0;
            output_valid_r     <= 1'b0;
            weight_addr_base_r <= '{default: '0};
        end else begin
            // done is intended as a one-cycle pulse
            done_r <= 1'b0;

            // Once final outputs are collected, mark them valid.
            // output_valid stays high until a new start arrives.
            if (collect_done) begin
                output_valid_r <= 1'b1;
                done_r         <= 1'b1;
            end else if (state_r == IDLE && start) begin
                output_valid_r <= 1'b0;
            end

            case (state_r)
                IDLE: begin
                    if (start) begin
                        // Start at the first group and first beat
                        state_r            <= DISPATCH;
                        group_idx_r        <= '0;
                        beat_idx_r         <= '0;

                        // Base weight address 0 means:
                        // each NP begins reading the first neuron's first beat
                        weight_addr_base_r <= '{default: '0};
                    end
                end

                DISPATCH: begin
                    // One cycle after reading memories into stage1,
                    // present that beat to the NPs
                    state_r <= COMPUTE;
                end

                COMPUTE: begin
                    if (last_stage2) begin
                        // All beats for this neuron group have been sent,
                        // now wait for NP outputs to finish coming out
                        state_r    <= WAIT_OUT;
                        beat_idx_r <= '0;
                    end else begin
                        // More beats remain for this same neuron group
                        beat_idx_r <= beat_idx_r + INPUT_ADDR_WIDTH'(1);
                        state_r    <= DISPATCH;
                    end
                end

                WAIT_OUT: begin
                    if (all_np_done) begin
                        if (all_groups_done) begin
                            // Entire layer is complete
                            state_r <= IDLE;
                        end else begin
                            // Move to the next group of neurons
                            group_idx_r <= group_idx_r + GROUP_WIDTH'(1);

                            // Advance each NP's base address by INPUT_BEATS
                            // so next time it reads the next neuron's weights
                            for (int i = 0; i < PN; i++) begin
                                weight_addr_base_r[i] <= weight_addr_base_r[i] +
                                                         WEIGHT_ADDR_WIDTH'(INPUT_BEATS);
                            end

                            state_r <= DISPATCH;
                        end
                    end
                end

                default: state_r <= IDLE;
            endcase
        end
    end

    // ========================================================================
    // Output Assignments
    // ========================================================================
    assign busy            = (state_r != IDLE);
    assign done            = done_r;
    assign cfg_ready       = !busy;
    assign output_valid    = output_valid_r;
    assign activations_out = activations_buffer;

    // Flatten the popcount array into one packed output bus
    always_comb begin
        for (int i = 0; i < NEURONS; i++) begin
            popcounts_out[i*COUNT_WIDTH+:COUNT_WIDTH] = popcounts_buffer[i];
        end
    end

endmodule
`default_nettype wire
