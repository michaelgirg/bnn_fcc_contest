//==============================================================================
// Argmax Module - Pipelined tree-based comparator for output layer
//==============================================================================
module argmax #(
    parameter  int OUTPUTS = 10,
    parameter  int COUNT_W = 9,
    localparam int CLASS_W = (OUTPUTS > 1) ? $clog2(OUTPUTS) : 1
) (
    input  logic                       clk,
    input  logic                       rst,
    input  logic                       valid_in,
    input  logic [OUTPUTS*COUNT_W-1:0] popcounts_in,
    output logic                       valid_out,
    output logic [        CLASS_W-1:0] class_idx,
    output logic [        COUNT_W-1:0] max_value
);

    //==========================================================================
    // Unpack inputs for easier access
    //==========================================================================
    logic [COUNT_W-1:0] count[OUTPUTS];
    always_comb begin
        for (int i = 0; i < OUTPUTS; i++) begin
            count[i] = popcounts_in[i*COUNT_W+:COUNT_W];
        end
    end

    //==========================================================================
    // Pipeline Stage 1: Pairwise comparisons + register results
    //==========================================================================
    typedef struct packed {
        logic [COUNT_W-1:0] value;
        logic [CLASS_W-1:0] index;
    } winner_t;

    winner_t stage1_winners_r[5];
    logic    valid_s1_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_s1_r <= 1'b0;
        end else begin
            // FIXED: Compare INCOMING data, register the results
            for (int i = 0; i < 5; i++) begin
                if (count[2*i] >= count[2*i+1]) begin
                    stage1_winners_r[i].value <= count[2*i];
                    stage1_winners_r[i].index <= CLASS_W'(2 * i);
                end else begin
                    stage1_winners_r[i].value <= count[2*i+1];
                    stage1_winners_r[i].index <= CLASS_W'(2 * i + 1);
                end
            end
            valid_s1_r <= valid_in;
        end
    end

    //==========================================================================
    // Pipeline Stage 2: Reduce 5 winners to 3
    //==========================================================================
    winner_t stage2_winners_r[3];
    logic    valid_s2_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            valid_s2_r <= 1'b0;
        end else begin
            // Compare winners [0] vs [1]
            if (stage1_winners_r[0].value >= stage1_winners_r[1].value) begin
                stage2_winners_r[0] <= stage1_winners_r[0];
            end else begin
                stage2_winners_r[0] <= stage1_winners_r[1];
            end

            // Compare winners [2] vs [3]
            if (stage1_winners_r[2].value >= stage1_winners_r[3].value) begin
                stage2_winners_r[1] <= stage1_winners_r[2];
            end else begin
                stage2_winners_r[1] <= stage1_winners_r[3];
            end

            // Pass through winner [4]
            stage2_winners_r[2] <= stage1_winners_r[4];

            valid_s2_r <= valid_s1_r;
        end
    end

    //==========================================================================
    // Pipeline Stage 3: Final comparison (3 inputs -> 1 winner)
    //==========================================================================
    logic [COUNT_W-1:0] max_value_r;
    logic [CLASS_W-1:0] class_idx_r;
    logic               valid_s3_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            max_value_r <= '0;
            class_idx_r <= '0;
            valid_s3_r  <= 1'b0;
        end else begin
            // Compare stage2[0] vs stage2[1]
            winner_t temp_winner;
            if (stage2_winners_r[0].value >= stage2_winners_r[1].value) begin
                temp_winner = stage2_winners_r[0];
            end else begin
                temp_winner = stage2_winners_r[1];
            end

            // Compare temp_winner vs stage2[2]
            if (temp_winner.value >= stage2_winners_r[2].value) begin
                max_value_r <= temp_winner.value;
                class_idx_r <= temp_winner.index;
            end else begin
                max_value_r <= stage2_winners_r[2].value;
                class_idx_r <= stage2_winners_r[2].index;
            end

            valid_s3_r <= valid_s2_r;
        end
    end

    //==========================================================================
    // Output assignments
    //==========================================================================
    assign max_value = max_value_r;
    assign class_idx = class_idx_r;
    assign valid_out = valid_s3_r;

endmodule
