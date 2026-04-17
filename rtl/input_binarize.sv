//==============================================================================
// Incremental Input Binarization Module
// Converts streaming 8-bit pixels to binary immediately as they arrive
//
// Timing-oriented version:
// - Keeps the same external interface and flat PIXELS-bit output
// - Adds a 1-cycle local capture stage for incoming beat/data
// - Uses 4 duplicated write-enable registers to reduce high fanout
// - Blocks a new image from starting once the last beat has been accepted
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
    localparam int NUM_BEATS       = (PIXELS + PIXELS_PER_BEAT - 1) / PIXELS_PER_BEAT;
    localparam int BEAT_WIDTH      = (NUM_BEATS > 1) ? $clog2(NUM_BEATS) : 1;

    // 4-way static partition of the output storage
    localparam int Q0_START = 0;
    localparam int Q1_START = (PIXELS * 1) / 4;
    localparam int Q2_START = (PIXELS * 2) / 4;
    localparam int Q3_START = (PIXELS * 3) / 4;
    localparam int Q4_START = PIXELS;

    localparam int Q0_W = Q1_START - Q0_START;
    localparam int Q1_W = Q2_START - Q1_START;
    localparam int Q2_W = Q3_START - Q2_START;
    localparam int Q3_W = Q4_START - Q3_START;

    //==========================================================================
    // State and Signals
    //==========================================================================
    logic [Q0_W-1:0] bin_buf_q0_r;
    logic [Q1_W-1:0] bin_buf_q1_r;
    logic [Q2_W-1:0] bin_buf_q2_r;
    logic [Q3_W-1:0] bin_buf_q3_r;

    logic [BEAT_WIDTH-1:0] beat_count_r;
    logic                  binarized_valid_r;
    logic                  capture_closed_r;

    // Stage-1 pipeline registers
    logic [BUS_WIDTH-1:0]  data_in_data_r;
    logic [BEAT_WIDTH-1:0] beat_count_pipe_r;
    logic                  last_pipe_r;

    // 4 duplicated write-enable registers
    logic [3:0] write_en_dup_r;

    logic accept_input;
    logic last_accepting_beat;

    assign last_accepting_beat = data_in_last || (beat_count_r == BEAT_WIDTH'(NUM_BEATS - 1));
    assign accept_input        = data_in_valid && data_in_ready;

    // Ready is high only while collecting one image and before final output is pending.
    assign data_in_ready = !binarized_valid_r && !capture_closed_r;

    //==========================================================================
    // Main sequential logic
    //==========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            bin_buf_q0_r       <= '0;
            bin_buf_q1_r       <= '0;
            bin_buf_q2_r       <= '0;
            bin_buf_q3_r       <= '0;
            beat_count_r       <= '0;
            binarized_valid_r  <= 1'b0;
            capture_closed_r   <= 1'b0;

            data_in_data_r     <= '0;
            beat_count_pipe_r  <= '0;
            last_pipe_r        <= 1'b0;
            write_en_dup_r     <= '0;
        end else begin
            // Default: duplicated enables are 1-cycle pulses
            write_en_dup_r <= '0;

            // Consume completed output image
            if (binarized_valid_r && binarized_ready) begin
                binarized_valid_r <= 1'b0;
                beat_count_r      <= '0;
                capture_closed_r  <= 1'b0;
            end

            // Stage 0: capture incoming beat and generate duplicated enables
            if (accept_input) begin
                data_in_data_r    <= data_in_data;
                beat_count_pipe_r <= beat_count_r;
                last_pipe_r       <= last_accepting_beat;
                write_en_dup_r    <= 4'b1111;

                if (last_accepting_beat) begin
                    capture_closed_r <= 1'b1;
                end else begin
                    beat_count_r <= beat_count_r + BEAT_WIDTH'(1);
                end
            end

            // Stage 1: update partitioned storage from duplicated enables
            if (|write_en_dup_r) begin
                for (int i = 0; i < PIXELS_PER_BEAT; i++) begin
                    int  flat_idx;
                    logic pixel_bit;

                    flat_idx  = beat_count_pipe_r * PIXELS_PER_BEAT + i;
                    pixel_bit = (data_in_data_r[i*PIXEL_W +: PIXEL_W] >= THRESHOLD);

                    if (flat_idx < PIXELS) begin
                        if ((flat_idx >= Q0_START) && (flat_idx < Q1_START) && write_en_dup_r[0]) begin
                            bin_buf_q0_r[flat_idx - Q0_START] <= pixel_bit;
                        end
                        else if ((flat_idx >= Q1_START) && (flat_idx < Q2_START) && write_en_dup_r[1]) begin
                            bin_buf_q1_r[flat_idx - Q1_START] <= pixel_bit;
                        end
                        else if ((flat_idx >= Q2_START) && (flat_idx < Q3_START) && write_en_dup_r[2]) begin
                            bin_buf_q2_r[flat_idx - Q2_START] <= pixel_bit;
                        end
                        else if ((flat_idx >= Q3_START) && (flat_idx < Q4_START) && write_en_dup_r[3]) begin
                            bin_buf_q3_r[flat_idx - Q3_START] <= pixel_bit;
                        end
                    end
                end

                if (last_pipe_r) begin
                    binarized_valid_r <= 1'b1;
                end
            end
        end
    end

    //==========================================================================
    // Output flattening
    //==========================================================================
    always_comb begin
        binarized_out = '0;

        for (int i = 0; i < Q0_W; i++) begin
            binarized_out[Q0_START + i] = bin_buf_q0_r[i];
        end
        for (int i = 0; i < Q1_W; i++) begin
            binarized_out[Q1_START + i] = bin_buf_q1_r[i];
        end
        for (int i = 0; i < Q2_W; i++) begin
            binarized_out[Q2_START + i] = bin_buf_q2_r[i];
        end
        for (int i = 0; i < Q3_W; i++) begin
            binarized_out[Q3_START + i] = bin_buf_q3_r[i];
        end
    end

    assign binarized_valid = binarized_valid_r;

endmodule