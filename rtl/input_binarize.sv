//==============================================================================
// Incremental Input Binarization Module
// Converts streaming 8-bit pixels to binary immediately as they arrive
//==============================================================================
module input_binarize #(
    parameter int                 PIXELS    = 784,
    parameter int                 PIXEL_W   = 8,
    parameter int                 BUS_WIDTH = 64,
    parameter logic [PIXEL_W-1:0] THRESHOLD = 8'd128
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic [BUS_WIDTH-1:0] data_in_data,
    input  logic                 data_in_valid,
    output logic                 data_in_ready,
    input  logic                 data_in_last,
    output logic [   PIXELS-1:0] binarized_out,
    output logic                 binarized_valid,
    input  logic                 binarized_ready
);

    //==========================================================================
    // Local Parameters
    //==========================================================================
    localparam int PIXELS_PER_BEAT = BUS_WIDTH / PIXEL_W;
    localparam int NUM_BEATS = (PIXELS + PIXELS_PER_BEAT - 1) / PIXELS_PER_BEAT;
    localparam int BEAT_WIDTH = (NUM_BEATS > 1) ? $clog2(NUM_BEATS) : 1;
    localparam int PIXEL_IDX_WIDTH = (PIXELS > 1) ? $clog2(PIXELS) : 1;

    //==========================================================================
    // State and Signals
    //==========================================================================
    logic [    PIXELS-1:0] binarized_buffer_r;
    logic [BEAT_WIDTH-1:0] beat_count_r;
    logic                  binarized_valid_r;
    logic [BEAT_WIDTH+2:0] base_idx;  // Declared outside always block

    //==========================================================================
    // Main FSM and Binarization Logic
    //==========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            binarized_buffer_r <= '0;
            beat_count_r       <= '0;
            binarized_valid_r  <= 1'b0;
        end else begin
            if (binarized_valid_r) begin
                if (binarized_ready) begin
                    binarized_valid_r <= 1'b0;
                    beat_count_r      <= '0;
                end
            end else if (data_in_valid && data_in_ready) begin
                base_idx = {beat_count_r, 3'b000};

                for (int i = 0; i < PIXELS_PER_BEAT; i++) begin
                    automatic logic [PIXEL_IDX_WIDTH-1:0] pixel_idx = base_idx + PIXEL_IDX_WIDTH'(i);
                    if (pixel_idx < PIXELS) begin
                        automatic logic [PIXEL_W-1:0] pixel = data_in_data[i*PIXEL_W+:PIXEL_W];
                        binarized_buffer_r[pixel_idx] <= (pixel >= THRESHOLD);
                    end
                end

                if (data_in_last || (beat_count_r == BEAT_WIDTH'(NUM_BEATS - 1))) begin
                    binarized_valid_r <= 1'b1;
                end else begin
                    beat_count_r <= beat_count_r + 1'b1;
                end
            end
        end
    end

    //==========================================================================
    // Output Assignments
    //==========================================================================
    assign data_in_ready   = !binarized_valid_r;
    assign binarized_out   = binarized_buffer_r;
    assign binarized_valid = binarized_valid_r;

endmodule
