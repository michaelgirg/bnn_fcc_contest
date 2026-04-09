module argmax #(
    parameter int OUTPUTS   = 10,
    parameter int COUNT_W   = 9,
    localparam int CLASS_W  = (OUTPUTS > 1) ? $clog2(OUTPUTS) : 1
) (
    input  logic                       valid_in,
    input  logic [OUTPUTS*COUNT_W-1:0] popcounts_in,
    output logic                       valid_out,
    output logic [CLASS_W-1:0]         class_idx,
    output logic [COUNT_W-1:0]         max_value
);

    logic [COUNT_W-1:0] popcount   [OUTPUTS];

    always_comb begin
        for (int i = 0; i < OUTPUTS; i++) begin
            popcount[i] = popcounts_in[i*COUNT_W +: COUNT_W];
        end

        class_idx = '0;
        max_value = popcount[0];

        for (int i = 1; i < OUTPUTS; i++) begin
            if (popcount[i] > max_value) begin
                max_value = popcount[i];
                class_idx = CLASS_W'(i);
            end
        end

        valid_out = valid_in;
    end

endmodule