//==============================================================================
// Argmax Module - Tree-based comparator for output layer
// Finds the neuron with maximum popcount value
//==============================================================================
module argmax #(
    parameter  int OUTPUTS = 10,
    parameter  int COUNT_W = 9,
    localparam int CLASS_W = (OUTPUTS > 1) ? $clog2(OUTPUTS) : 1
) (
    input  logic                       valid_in,
    input  logic [OUTPUTS*COUNT_W-1:0] popcounts_in,
    output logic                       valid_out,
    output logic [        CLASS_W-1:0] class_idx,
    output logic [        COUNT_W-1:0] max_value
);

    //==========================================================================
    // Stage 0: Unpack input popcounts
    //==========================================================================
    logic [COUNT_W-1:0] count[OUTPUTS];

    always_comb begin
        for (int i = 0; i < OUTPUTS; i++) begin
            count[i] = popcounts_in[i*COUNT_W+:COUNT_W];
        end
    end

    //==========================================================================
    // Stage 1: Pairwise comparison (5 comparisons in parallel)
    // Compares: (0,1), (2,3), (4,5), (6,7), (8,9)
    //==========================================================================
    logic [COUNT_W-1:0] stage1_max[5];
    logic [CLASS_W-1:0] stage1_idx[5];

    always_comb begin
        for (int i = 0; i < 5; i++) begin
            if (count[2*i] >= count[2*i+1]) begin
                stage1_max[i] = count[2*i];
                stage1_idx[i] = CLASS_W'(2 * i);
            end else begin
                stage1_max[i] = count[2*i+1];
                stage1_idx[i] = CLASS_W'(2 * i + 1);
            end
        end
    end

    //==========================================================================
    // Stage 2: Reduce 5 winners to 3
    // Compares: (stage1[0], stage1[1]), (stage1[2], stage1[3])
    // Passes through: stage1[4]
    //==========================================================================
    logic [COUNT_W-1:0] stage2_max[3];
    logic [CLASS_W-1:0] stage2_idx[3];

    always_comb begin
        // Compare first two stage1 winners
        if (stage1_max[0] >= stage1_max[1]) begin
            stage2_max[0] = stage1_max[0];
            stage2_idx[0] = stage1_idx[0];
        end else begin
            stage2_max[0] = stage1_max[1];
            stage2_idx[0] = stage1_idx[1];
        end

        // Compare next two stage1 winners
        if (stage1_max[2] >= stage1_max[3]) begin
            stage2_max[1] = stage1_max[2];
            stage2_idx[1] = stage1_idx[2];
        end else begin
            stage2_max[1] = stage1_max[3];
            stage2_idx[1] = stage1_idx[3];
        end

        // Pass through the odd one
        stage2_max[2] = stage1_max[4];
        stage2_idx[2] = stage1_idx[4];
    end

    //==========================================================================
    // Stage 3: Final reduction (find maximum of 3 values)
    //==========================================================================
    logic [COUNT_W-1:0] stage3_max_temp;
    logic [CLASS_W-1:0] stage3_idx_temp;

    always_comb begin
        // First compare stage2[0] vs stage2[1]
        if (stage2_max[0] >= stage2_max[1]) begin
            stage3_max_temp = stage2_max[0];
            stage3_idx_temp = stage2_idx[0];
        end else begin
            stage3_max_temp = stage2_max[1];
            stage3_idx_temp = stage2_idx[1];
        end

        // Then compare winner against stage2[2]
        if (stage3_max_temp >= stage2_max[2]) begin
            max_value = stage3_max_temp;
            class_idx = stage3_idx_temp;
        end else begin
            max_value = stage2_max[2];
            class_idx = stage2_idx[2];
        end

        // Valid passthrough
        valid_out = valid_in;
    end

endmodule
