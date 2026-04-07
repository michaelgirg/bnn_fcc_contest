`default_nettype none
// ============================================================================
// MODULE: bnn_layer
// PURPOSE: Complete Binary Neural Network (BNN) layer with parallel processing
//
// ARCHITECTURE OVERVIEW:
// This module implements a complete BNN layer that can process multiple neurons
// in parallel using PN neuron processors (NPs). The design uses a time-multiplexed
// approach where each NP handles multiple neurons across different groups.
//
// KEY FEATURES:
// - Input buffer: Stores complete input vector (auto-padded to PW-bit boundaries)
// - Parallel processing: PN neuron processors work simultaneously
// - Distributed memory: Each NP has its own weight RAM and threshold storage
// - Pipeline stages: Clean DISPATCH (read memory) → COMPUTE (process) cycle
// - Dual mode: Supports hidden layers (binary outputs) and output layers (popcounts)
//
// NEURON-TO-NP MAPPING STRATEGY:
// Neurons are distributed round-robin across NPs:
// - Neuron n is assigned to: NP[n % PN] in group[n / PN]
// - Example with PN=4, 10 neurons:
//   * Neurons 0,4,8  → NP0 (groups 0,1,2)
//   * Neurons 1,5,9  → NP1 (groups 0,1,2)
//   * Neurons 2,6    → NP2 (groups 0,1)
//   * Neurons 3,7    → NP3 (groups 0,1)
//
// MEMORY ADDRESSING:
// Each NP stores weights for all its assigned neurons:
// - Weight address = group_idx * INPUT_BEATS + beat_idx
// - This allows sequential access during computation
//
// OPERATION FLOW:
// 1. Configuration: Load weights and thresholds (one-time setup)
// 2. Input loading: Store input vector in buffer
// 3. Computation: Process all neurons group-by-group, beat-by-beat
// 4. Output: Collect results from all NPs
// ============================================================================

module bnn_layer #(
    // ========================================================================
    // USER PARAMETERS
    // ========================================================================
    parameter int INPUTS       = 784,  // Number of input features
    parameter int NEURONS      = 256,  // Number of neurons in this layer
    parameter int PW           = 8,    // Processing width (bits processed per beat)
    parameter int PN           = 8,    // Parallelism (number of neuron processors)
    parameter bit OUTPUT_LAYER = 0,    // 0=hidden layer (binary), 1=output layer (popcount)

    // ========================================================================
    // DERIVED PARAMETERS (automatically calculated)
    // ========================================================================
    parameter int COUNT_WIDTH = $clog2(INPUTS + 1),  // Bits needed for popcount value
    parameter int INPUT_BEATS = (INPUTS + PW - 1) / PW,  // Beats to process all inputs
    parameter int CFG_NEURON_WIDTH = (NEURONS > 1) ? $clog2(NEURONS) : 1,  // Neuron index width
    parameter int CFG_WEIGHT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1  // Beat address width
) (
    input wire logic clk,
    input wire logic rst,  // Active-high synchronous reset

    // ========================================================================
    // COMPUTE INTERFACE
    // ========================================================================
    // Input loading (before computation starts)
    input wire logic              input_load,   // Pulse to load input_vector into buffer
    input wire logic [INPUTS-1:0] input_vector, // Input feature vector to process

    // Computation control
    input  wire logic start,  // Pulse to begin layer computation
    output logic      done,   // Pulses for 1 cycle when computation completes
    output logic      busy,   // High while computation in progress

    // Results (valid when done=1)
    output logic [            NEURONS-1:0] activations_out,  // Binary outputs (hidden layer)
    output logic [NEURONS*COUNT_WIDTH-1:0] popcounts_out,    // Popcount outputs (output layer)
    output logic                           output_valid,     // High when outputs are valid

    // ========================================================================
    // CONFIGURATION INTERFACE (for programming weights/thresholds)
    // ========================================================================
    input  wire logic                             cfg_write_en,         // Enable write operation
    input  wire logic [     CFG_NEURON_WIDTH-1:0] cfg_neuron_idx,       // Which neuron to configure
    input  wire logic [CFG_WEIGHT_ADDR_WIDTH-1:0] cfg_weight_addr,      // Which beat of weights
    input  wire logic [                   PW-1:0] cfg_weight_data,      // Weight data to write
    input  wire logic [          COUNT_WIDTH-1:0] cfg_threshold_data,   // Threshold value to write
    input  wire logic                             cfg_threshold_write,  // 1=write threshold, 0=write weights
    output logic                                  cfg_ready             // High when ready for configuration
);

    // ========================================================================
    // DERIVED PARAMETERS (width-safe calculations)
    // These ensure we never create zero-width signals even with edge-case parameters
    // ========================================================================
    localparam int NEURON_GROUPS = (NEURONS + PN - 1) / PN;  // How many groups (ceil division)
    localparam int INPUT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1;  // Avoid width=0
    localparam int WEIGHT_DEPTH = INPUT_BEATS * NEURON_GROUPS;  // Total weight entries per NP
    localparam int WEIGHT_ADDR_WIDTH = (WEIGHT_DEPTH > 1) ? $clog2(WEIGHT_DEPTH) : 1;
    localparam int GROUP_WIDTH = (NEURON_GROUPS > 1) ? $clog2(NEURON_GROUPS) : 1;
    localparam int PADDED_INPUTS = INPUT_BEATS * PW;  // Input size rounded up to PW boundary

    // ========================================================================
    // SIGNAL DECLARATIONS
    // ========================================================================

    // FSM (Finite State Machine) signals
    typedef enum logic [2:0] {
        IDLE,      // Waiting for start signal.
        DISPATCH,  // Reading memory (input, weights, thresholds)
        COMPUTE,   // Sending data to NPs for processing
        WAIT_OUT,  // Waiting for NPs to finish current group
        DONE       // Computation complete, asserting done signal
    } state_t;

    state_t state_r;  // Current FSM state
    logic [GROUP_WIDTH-1:0] group_idx_r;  // Current neuron group being processed
    logic [INPUT_ADDR_WIDTH-1:0] beat_idx_r;  // Current input beat being processed
    logic done_r;  // Done signal register

    // ------------------------------------------------------------------------
    // Control signals (derived from state/counters)
    // ------------------------------------------------------------------------
    logic all_groups_done;  // High when we've processed all neuron groups
    logic all_beats_done;  // High when we've processed all input beats
    logic all_np_done;  // High when all active NPs have finished

    // ------------------------------------------------------------------------
    // Neuron processor control signals
    // ------------------------------------------------------------------------
    logic np_valid_in;  // Valid signal to NPs (asserted during COMPUTE)
    logic np_last;  // Last beat indicator to NPs

    // ------------------------------------------------------------------------
    // NP active mask (tracks which NPs have work to do)
    // In the last group, some NPs may be inactive if NEURONS % PN != 0
    // ------------------------------------------------------------------------
    logic [PN-1:0] np_active_r;  // Bit mask: 1=NP active, 0=NP idle

    // ------------------------------------------------------------------------
    // Input buffer: Stores input vector for entire layer computation
    // Broken into INPUT_BEATS chunks of PW bits each
    // ------------------------------------------------------------------------
    logic [PW-1:0] input_buffer_r[INPUT_BEATS];  // Input storage
    logic [PADDED_INPUTS-1:0] input_vector_padded;  // Input with zero-padding

    // ------------------------------------------------------------------------
    // Weight and threshold storage (distributed across NPs)
    // Each NP stores weights for all its assigned neurons
    // Weight organization: [NP][group * INPUT_BEATS + beat]
    // Threshold organization: [NP][group]
    // ------------------------------------------------------------------------
    logic [PW-1:0] weight_rams_r[PN][WEIGHT_DEPTH];  // Weight memories
    logic [COUNT_WIDTH-1:0] threshold_rams_r[PN][NEURON_GROUPS];  // Threshold memories

    // ------------------------------------------------------------------------
    // DISPATCH stage registers (pipeline between memory read and compute)
    // These hold the data for one beat while NPs process it
    // ------------------------------------------------------------------------
    logic [PW-1:0] input_chunk_r;  // Current input beat data
    logic [PW-1:0] weight_data_r[PN];  // Weight data for each NP
    logic [COUNT_WIDTH-1:0] threshold_data_r[PN];  // Threshold for each NP
    logic last_dispatch_r;  // Last beat flag (captured in DISPATCH)
    logic [PN-1:0] np_active_dispatch_r;  // Active mask (captured in DISPATCH)

    // ------------------------------------------------------------------------
    // Memory read addresses (combinational - generated from counters)
    // ------------------------------------------------------------------------
    logic [INPUT_ADDR_WIDTH-1:0] input_read_addr;  // Which beat of input buffer
    logic [WEIGHT_ADDR_WIDTH-1:0] weight_read_addr[PN];  // Weight address for each NP
    logic [GROUP_WIDTH-1:0] threshold_read_addr;  // Which group's threshold

    // ------------------------------------------------------------------------
    // Neuron processor outputs
    // Each NP produces: valid, activation, and popcount
    // ------------------------------------------------------------------------
    logic [PN-1:0] np_valid_out;  // Valid output from each NP
    logic [PN-1:0] np_activation;  // Binary activation from each NP
    logic [COUNT_WIDTH-1:0] np_popcount[PN];  // Popcount value from each NP

    // ------------------------------------------------------------------------
    // Output collection buffers
    // Accumulate results from all groups of neurons
    // ------------------------------------------------------------------------
    logic [NEURONS-1:0] activations_buffer_r;  // All neuron activations
    logic [COUNT_WIDTH-1:0] popcounts_buffer_r[NEURONS];  // All neuron popcounts
    logic output_valid_r;  // Output valid flag

    // ========================================================================
    // DESIGN-FOR-VERIFICATION ASSERTIONS
    // These check for protocol violations during simulation
    // ========================================================================
`ifndef SYNTHESIS
    // Don't load input while computing
    assert property (@(posedge clk) disable iff (rst) busy |-> !input_load)
    else $error("input_load asserted while busy");

    // Don't configure while computing
    assert property (@(posedge clk) disable iff (rst) busy |-> !cfg_write_en)
    else $error("cfg_write_en asserted while busy");

    // Don't start twice
    assert property (@(posedge clk) disable iff (rst) busy |-> !start)
    else $error("start asserted while already busy");
`endif

    // ========================================================================
    // CONTINUOUS ASSIGNMENTS (combinational logic)
    // ========================================================================

    // Pad input vector with zeros to match INPUT_BEATS * PW size
    assign input_vector_padded = {{(PADDED_INPUTS - INPUTS) {1'b0}}, input_vector};

    // Check if we've completed all beats/groups
    assign all_beats_done = (beat_idx_r == INPUT_ADDR_WIDTH'(INPUT_BEATS - 1));
    assign all_groups_done = (group_idx_r == GROUP_WIDTH'(NEURON_GROUPS - 1));

    // All NPs done when: (valid_out=1) OR (not active)
    // This handles the case where inactive NPs never assert valid_out
    assign all_np_done = &(np_valid_out | ~np_active_dispatch_r);

    // NP control signals derived from FSM state
    assign np_valid_in = (state_r == COMPUTE);  // Assert during COMPUTE cycle
    assign np_last = (state_r == COMPUTE) && last_dispatch_r;  // Last beat indicator

    // Status outputs
    assign busy = (state_r != IDLE);
    assign done = done_r;
    assign cfg_ready = !busy;
    assign output_valid = output_valid_r;
    assign activations_out = activations_buffer_r;

    // ========================================================================
    // INPUT BUFFER
    // Captures input vector when input_load=1
    // Breaks vector into INPUT_BEATS chunks of PW bits each
    // ========================================================================
    always_ff @(posedge clk) begin
        if (input_load && !busy) begin
            for (int i = 0; i < INPUT_BEATS; i++) begin
                // Extract PW bits starting at position i*PW
                input_buffer_r[i] <= input_vector_padded[i*PW+:PW];
            end
        end
    end

    // ========================================================================
    // WEIGHT RAMS (distributed across neuron processors)
    // Configuration: Write weights for specific neuron during setup
    // Computation: Read sequentially during layer execution
    //
    // ADDRESSING SCHEME:
    // Global neuron N → NP[N%PN], Group[N/PN]
    // Local address in NP = Group * INPUT_BEATS + beat_index
    // ========================================================================
    always_ff @(posedge clk) begin
        if (cfg_write_en && !cfg_threshold_write && !busy) begin
            // Decode which NP and address to write
            automatic logic [$clog2(PN)-1:0] np_idx;
            automatic logic [GROUP_WIDTH-1:0] group_idx;
            automatic logic [WEIGHT_ADDR_WIDTH-1:0] local_addr;

            // Map global neuron index to (NP, group)
            np_idx = $clog2(PN)'(cfg_neuron_idx % PN);
            group_idx = GROUP_WIDTH'(cfg_neuron_idx / PN);

            // Calculate local address within NP's RAM
            local_addr = WEIGHT_ADDR_WIDTH'(group_idx * INPUT_BEATS + cfg_weight_addr);

            // Write to the appropriate NP's weight RAM
            weight_rams_r[np_idx][local_addr] <= cfg_weight_data;
        end
    end

    // ========================================================================
    // THRESHOLD STORAGE (one threshold per neuron per group)
    // Thresholds determine activation: (popcount >= threshold) ? 1 : 0
    // ========================================================================
    always_ff @(posedge clk) begin
        if (cfg_write_en && cfg_threshold_write && !busy) begin
            // Decode which NP and group to write
            automatic logic [ $clog2(PN)-1:0] np_idx;
            automatic logic [GROUP_WIDTH-1:0] group_idx;

            np_idx = $clog2(PN)'(cfg_neuron_idx % PN);
            group_idx = GROUP_WIDTH'(cfg_neuron_idx / PN);

            threshold_rams_r[np_idx][group_idx] <= cfg_threshold_data;
        end
    end

    // ========================================================================
    // NP ACTIVE MASK MANAGEMENT
    // Controls which NPs are active for the current group
    //
    // KEY CASES:
    // 1. All groups except last: All PN processors active
    // 2. Last group: Only (NEURONS % PN) processors active
    //    Example: 10 neurons, PN=4 → last group has 2 active (neurons 8,9)
    // ========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            np_active_r <= '0;
        end else if (state_r == IDLE && start) begin
            // Initial activation on start
            if (NEURONS >= PN) begin
                np_active_r <= '1;  // All NPs active
            end else begin
                // Fewer neurons than NPs: activate only what we need
                for (int i = 0; i < PN; i++) begin
                    np_active_r[i] <= (i < NEURONS);
                end
            end
        end else if (state_r == WAIT_OUT && all_np_done) begin
            // Update mask when moving to next group
            if (!all_groups_done) begin
                automatic logic [GROUP_WIDTH-1:0] next_group;
                automatic int remaining;

                next_group = group_idx_r + GROUP_WIDTH'(1);

                // Check if next group is the last group
                if (next_group == GROUP_WIDTH'(NEURON_GROUPS - 1)) begin
                    // Last group: calculate how many neurons remain
                    remaining = NEURONS - (int'(next_group) * PN);
                    for (int i = 0; i < PN; i++) begin
                        np_active_r[i] <= (i < remaining);
                    end
                end else begin
                    // Not last group: all NPs active
                    np_active_r <= '1;
                end
            end
        end
    end

    // ========================================================================
    // MEMORY READ ADDRESS GENERATION (combinational)
    // Generates addresses for reading input buffer, weights, and thresholds
    // during the DISPATCH state
    // ========================================================================
    always_comb begin
        // Input buffer address: simply the current beat index
        input_read_addr = beat_idx_r;

        // Threshold address: simply the current group index
        threshold_read_addr = group_idx_r;

        // Weight address calculation for each NP
        // Formula: group * INPUT_BEATS + beat
        // This gives sequential access pattern during computation
        for (int i = 0; i < PN; i++) begin
            weight_read_addr[i] = WEIGHT_ADDR_WIDTH'(group_idx_r) *
                                  WEIGHT_ADDR_WIDTH'(INPUT_BEATS) +
                                  WEIGHT_ADDR_WIDTH'(beat_idx_r);
        end
    end

    // ========================================================================
    // DISPATCH STAGE REGISTERS
    // Pipeline stage between memory read and neuron processor compute
    //
    // TIMING:
    // Clock N:   DISPATCH state → read addresses generated
    // Clock N+1: Data arrives from memories → captured here
    // Clock N+2: COMPUTE state → data sent to NPs
    //
    // This staging allows 1-cycle memory access without timing issues
    // ========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            input_chunk_r <= '0;
            last_dispatch_r <= 1'b0;
            np_active_dispatch_r <= '0;
            for (int i = 0; i < PN; i++) begin
                weight_data_r[i] <= '0;
                threshold_data_r[i] <= '0;
            end
        end else if (state_r == DISPATCH) begin
            // Capture memory outputs during DISPATCH
            input_chunk_r <= input_buffer_r[input_read_addr];
            np_active_dispatch_r <= np_active_r;
            last_dispatch_r <= all_beats_done;

            // Capture weight and threshold for each NP
            for (int i = 0; i < PN; i++) begin
                weight_data_r[i] <= weight_rams_r[i][weight_read_addr[i]];
                threshold_data_r[i] <= threshold_rams_r[i][threshold_read_addr];
            end
        end
    end

    // ========================================================================
    // MAIN FSM (Finite State Machine)
    // Controls the overall layer computation flow
    //
    // STATE FLOW:
    // IDLE → DISPATCH → COMPUTE → (repeat for all beats) → WAIT_OUT →
    //   (repeat for all groups) → DONE → IDLE
    //
    // DETAILED FLOW:
    // 1. IDLE: Wait for start signal
    // 2. DISPATCH: Read memory for one beat
    // 3. COMPUTE: Send data to NPs (they start processing)
    // 4. If more beats: go back to DISPATCH
    //    If all beats done: go to WAIT_OUT
    // 5. WAIT_OUT: Wait for all NPs to finish
    // 6. If more groups: increment group counter, go to DISPATCH
    //    If all groups done: go to DONE
    // 7. DONE: Pulse done signal, go to IDLE
    // ========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            state_r <= IDLE;
            group_idx_r <= '0;
            beat_idx_r <= '0;
            done_r <= 1'b0;
        end else begin
            done_r <= 1'b0;  // Default: done is a 1-cycle pulse

            case (state_r)
                IDLE: begin
                    if (start) begin
                        // Begin computation
                        state_r <= DISPATCH;
                        group_idx_r <= '0;
                        beat_idx_r <= '0;
                    end
                end

                DISPATCH: begin
                    // Always go to COMPUTE after DISPATCH
                    // (memory read takes 1 cycle, data will be ready)
                    state_r <= COMPUTE;
                end

                COMPUTE: begin
                    if (last_dispatch_r) begin
                        // Finished all beats for this group
                        state_r <= WAIT_OUT;
                        beat_idx_r <= '0;  // Reset beat counter for next group
                    end else begin
                        // More beats to process
                        beat_idx_r <= beat_idx_r + INPUT_ADDR_WIDTH'(1);
                        state_r <= DISPATCH;
                    end
                end

                WAIT_OUT: begin
                    if (all_np_done) begin
                        // All NPs finished processing current group
                        if (all_groups_done) begin
                            // Finished all groups → computation complete
                            state_r <= DONE;
                        end else begin
                            // More groups to process
                            group_idx_r <= group_idx_r + GROUP_WIDTH'(1);
                            state_r <= DISPATCH;
                        end
                    end
                    // Else: stay in WAIT_OUT until NPs finish
                end

                DONE: begin
                    done_r  <= 1'b1;  // Pulse done for 1 cycle
                    state_r <= IDLE;
                end

                default: begin
                    // Illegal state recovery
`ifndef SYNTHESIS
                    // In simulation: propagate X to catch bugs
                    state_r <= state_t'('x);
                    group_idx_r <= 'x;
                    beat_idx_r <= 'x;
                    done_r <= 1'bx;
`else
                    // In synthesis: recover to safe state
                    state_r <= IDLE;
                    group_idx_r <= '0;
                    beat_idx_r <= '0;
                    done_r <= 1'b0;
`endif
                end
            endcase
        end
    end

    // ========================================================================
    // NEURON PROCESSOR INSTANTIATIONS
    // Creates PN parallel neuron processors
    //
    // Each NP:
    // - Receives input chunk and weight data
    // - Computes XNOR-popcount (similarity measure)
    // - Compares to threshold → activation
    // - Operates independently in parallel
    // ========================================================================
    generate
        for (genvar i = 0; i < PN; i++) begin : gen_nps
            neuron_processor #(
                .PW               (PW),
                .MAX_NEURON_INPUTS(INPUTS),
                .OUTPUT_LAYER     (OUTPUT_LAYER)
            ) u_np (
                .clk         (clk),
                .rst         (rst),
                // Control: only active if this NP has work
                .valid_in    (np_valid_in && np_active_dispatch_r[i]),
                .last        (np_last),
                // Data inputs (shared across all NPs)
                .x           (input_chunk_r),
                // Data inputs (unique to this NP)
                .w           (weight_data_r[i]),
                .threshold   (threshold_data_r[i]),
                // Outputs
                .valid_out   (np_valid_out[i]),
                .activation  (np_activation[i]),
                .popcount_out(np_popcount[i])
            );
        end
    endgenerate

    // ========================================================================
    // OUTPUT COLLECTION
    // Gathers results from all neuron processors after each group completes
    //
    // OPERATION:
    // When all NPs finish (WAIT_OUT + all_np_done):
    // - For each active NP (i):
    //   * Calculate global neuron index: group * PN + i
    //   * Store NP's result in the appropriate buffer slot
    //
    // OUTPUT MODES:
    // - Hidden layer (OUTPUT_LAYER=0): Store binary activations
    // - Output layer (OUTPUT_LAYER=1): Store popcount values
    // ========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            output_valid_r <= 1'b0;
            activations_buffer_r <= '0;
            for (int i = 0; i < NEURONS; i++) begin
                popcounts_buffer_r[i] <= '0;
            end
        end else if (state_r == WAIT_OUT && all_np_done) begin
            // Collect results from all active NPs
            for (int i = 0; i < PN; i++) begin
                if (np_active_dispatch_r[i]) begin
                    // Calculate global neuron index
                    // Example: group=2, PN=4, i=1 → neuron = 2*4+1 = 9
                    automatic int neuron_idx;
                    neuron_idx = int'(group_idx_r) * PN + i;

                    if (OUTPUT_LAYER) begin
                        // Output layer: store popcount
                        popcounts_buffer_r[neuron_idx] <= np_popcount[i];
                    end else begin
                        // Hidden layer: store binary activation
                        activations_buffer_r[neuron_idx] <= np_activation[i];
                    end
                end
            end
        end else if (state_r == DONE) begin
            // Set output valid flag
            output_valid_r <= 1'b1;
        end else if (state_r == IDLE) begin
            // Clear output valid flag
            output_valid_r <= 1'b0;
        end
    end

    // ========================================================================
    // OUTPUT PACKING
    // Packs individual popcount values into flat output vector
    // Format: [neuron_N-1, ..., neuron_1, neuron_0]
    // Each neuron's popcount occupies COUNT_WIDTH bits
    // ========================================================================
    always_comb begin
        for (int i = 0; i < NEURONS; i++) begin
            popcounts_out[i*COUNT_WIDTH+:COUNT_WIDTH] = popcounts_buffer_r[i];
        end
    end

endmodule
`default_nettype wire
