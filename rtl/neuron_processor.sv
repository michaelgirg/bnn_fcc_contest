// neuron_processor.sv - FULLY FIXED VERSION
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

    // Stage A: Register inputs and compute XNOR
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
            stage_a_valid_r     <= valid_in;
            stage_a_last_r      <= last;
            stage_a_xnor_r      <= x ~^ w;  // Always compute, will be ignored if !valid
            stage_a_threshold_r <= threshold;
        end
    end

    // Stage B: Popcount and register
    logic [PARTIAL_WIDTH-1:0] partial_popcount_comb;
    logic [PARTIAL_WIDTH-1:0] stage_b_partial_r;
    logic                     stage_b_valid_r;
    logic                     stage_b_last_r;
    logic [  ACCUM_WIDTH-1:0] stage_b_threshold_r;

    always_comb begin
        partial_popcount_comb = '0;
        for (int i = 0; i < PW; i++)
        partial_popcount_comb = partial_popcount_comb + PARTIAL_WIDTH'(stage_a_xnor_r[i]);
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

    // Stage C: Accumulate + compare + output
    logic [ACCUM_WIDTH-1:0] accumulator_r;
    logic [ACCUM_WIDTH-1:0] final_sum_comb;

    assign final_sum_comb = accumulator_r + ACCUM_WIDTH'(stage_b_partial_r);

    always_ff @(posedge clk) begin
        if (rst) begin
            accumulator_r <= '0;
        end else begin
            if (stage_b_valid_r) begin
                if (stage_b_last_r) begin
                    accumulator_r <= '0;
                end else begin
                    accumulator_r <= final_sum_comb;
                end
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
                popcount_out_r <= final_sum_comb;
                activation_r   <= OUTPUT_LAYER ? 1'b0 : (final_sum_comb >= stage_b_threshold_r);
            end
        end
    end

    assign valid_out    = valid_out_r;
    assign activation   = activation_r;
    assign popcount_out = popcount_out_r;

endmodule
