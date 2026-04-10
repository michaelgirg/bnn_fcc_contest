// ============================================================================
// Module: neuron_processor
// ============================================================================
// 3-STAGE PIPELINE - LATENCY OPTIMIZED
//   Stage A: Register inputs + XNOR                      (cycle 1)
//   Stage B: Popcount + register                         (cycle 2)  
//   Stage C: Accumulate + compare + output (SINGLE CYCLE)(cycle 3)
//
// Total latency: N_beats + 3 cycles
// 
// Key latency features:
// - Stage C combines accumulation + threshold comparison in ONE cycle
// - No extra adder pipeline stage (combinational add feeds comparator directly)
// - Operand isolation in Stage A (gates compute with valid_in) [1]
// - Control/data alignment: accumulator uses stage_b_* with stage_b_partial_r
//
// Critical path: accumulator + stage_b_partial → compare → output
// Only split Stage C if static timing analysis fails [5]
// ============================================================================
module neuron_processor #(
    parameter int PW                = 8,
    parameter int MAX_NEURON_INPUTS = 784,
    parameter bit OUTPUT_LAYER      = 0
) (
    input  logic                                   clk,
    input  logic                                   rst,
    input  logic                                   valid_in,
    input  logic                                   last,
    input  logic [                         PW-1:0] x,
    input  logic [                         PW-1:0] w,
    input  logic [$clog2(MAX_NEURON_INPUTS+1)-1:0] threshold,
    output logic                                   valid_out,
    output logic                                   activation,
    output logic [$clog2(MAX_NEURON_INPUTS+1)-1:0] popcount_out
);
    localparam int ACCUM_WIDTH = $clog2(MAX_NEURON_INPUTS + 1);
    localparam int PARTIAL_WIDTH = $clog2(PW + 1);

    // =========================================================================
    // Stage A: Register inputs and compute XNOR
    // Operand isolation: gate compute with valid_in to reduce power [1]
    // =========================================================================
    logic                   stage_a_valid_r;
    logic                   stage_a_last_r;
    logic [         PW-1:0] stage_a_xnor_r;
    logic [ACCUM_WIDTH-1:0] stage_a_threshold_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            stage_a_valid_r     <= '0;
            stage_a_last_r      <= '0;
            stage_a_xnor_r      <= '0;
            stage_a_threshold_r <= '0;
        end else begin
            stage_a_valid_r <= valid_in;
            stage_a_last_r  <= last;

            // Operand isolation: only compute when valid_in asserted [1]
            if (valid_in) begin
                stage_a_xnor_r      <= x ~^ w;  // Counts matches
                stage_a_threshold_r <= threshold;
            end
            // else: hold previous values (reduces unnecessary switching)
        end
    end

    // =========================================================================
    // Stage B: Popcount and register
    // Simple loop - synthesis tools optimize carry chains automatically [1]
    // =========================================================================
    logic [PARTIAL_WIDTH-1:0] partial_popcount_comb;
    logic [PARTIAL_WIDTH-1:0] stage_b_partial_r;
    logic                     stage_b_valid_r;
    logic                     stage_b_last_r;
    logic [  ACCUM_WIDTH-1:0] stage_b_threshold_r;

    always_comb begin
        partial_popcount_comb = '0;
        for (int i = 0; i < PW; i++) partial_popcount_comb += PARTIAL_WIDTH'(stage_a_xnor_r[i]);
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            stage_b_partial_r   <= '0;
            stage_b_valid_r     <= '0;
            stage_b_last_r      <= '0;
            stage_b_threshold_r <= '0;
        end else begin
            stage_b_partial_r   <= partial_popcount_comb;
            stage_b_valid_r     <= stage_a_valid_r;
            stage_b_last_r      <= stage_a_last_r;
            stage_b_threshold_r <= stage_a_threshold_r;
        end
    end

    // =========================================================================
    // Stage C: SINGLE-CYCLE accumulate + compare + output (LATENCY CRITICAL)
    // 
    // This is the longest combinational path in the design:
    //   accumulator_r + stage_b_partial_r → comparison → output
    // 
    // Splitting this would add a 4th pipeline stage (worse latency).
    // Only split if static timing analysis shows timing failure [5].
    // 
    // CRITICAL: Accumulator control uses Stage B signals (stage_b_valid_r,
    // stage_b_last_r) with Stage B data (stage_b_partial_r) to maintain
    // proper pipeline alignment.
    // =========================================================================
    logic [ACCUM_WIDTH-1:0] accumulator_r;
    logic [ACCUM_WIDTH-1:0] final_sum_comb;

    // Combinational sum: accumulator + current partial (critical path start)
    assign final_sum_comb = accumulator_r + ACCUM_WIDTH'(stage_b_partial_r);

    // Accumulator maintenance - uses Stage B control with Stage B data
    always_ff @(posedge clk) begin
        if (rst) begin
            accumulator_r <= '0;
        end else begin
            if (stage_b_valid_r && stage_b_last_r) begin
                // Last beat: clear for next neuron
                accumulator_r <= '0;
            end else if (stage_b_valid_r) begin
                // Non-last beat: save running sum
                accumulator_r <= final_sum_comb;
            end
        end
    end

    // Output registers
    logic                   valid_out_r;
    logic                   activation_r;
    logic [ACCUM_WIDTH-1:0] popcount_out_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_out_r    <= '0;
            activation_r   <= '0;
            popcount_out_r <= '0;
        end else begin
            valid_out_r <= stage_b_valid_r && stage_b_last_r;

            if (stage_b_valid_r && stage_b_last_r) begin
                // Capture final sum (includes current partial via final_sum_comb)
                popcount_out_r <= final_sum_comb;

                // Hidden layers: fire if popcount >= threshold
                // Output layer: activation unused (always 0)
                activation_r   <= OUTPUT_LAYER ? 1'b0 : (final_sum_comb >= stage_b_threshold_r);
            end
        end
    end

    assign valid_out    = valid_out_r;
    assign activation   = activation_r;
    assign popcount_out = popcount_out_r;

endmodule
