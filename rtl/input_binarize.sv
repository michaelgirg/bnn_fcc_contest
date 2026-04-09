module input_binarize #(
    parameter int PIXELS    = 784,
    parameter int PIXEL_W   = 8,
    parameter logic [PIXEL_W-1:0] THRESHOLD = 8'd128
) (
    input  logic [PIXELS*PIXEL_W-1:0] pixels_in,
    output logic [PIXELS-1:0]         bits_out
);

    always_comb begin
        for (int i = 0; i < PIXELS; i++) begin
            bits_out[i] = (pixels_in[i*PIXEL_W +: PIXEL_W] >= THRESHOLD);
        end
    end

endmodule