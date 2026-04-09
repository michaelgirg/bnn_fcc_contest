`default_nettype none

module input_binarize_tb;

    localparam int PIXELS  = 16;
    localparam int PIXEL_W = 8;

    logic [PIXELS*PIXEL_W-1:0] pixels_in;
    logic [PIXELS-1:0]         bits_out;

    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    input_binarize #(
        .PIXELS   (PIXELS),
        .PIXEL_W  (PIXEL_W),
        .THRESHOLD(8'd128)
    ) dut (
        .pixels_in(pixels_in),
        .bits_out (bits_out)
    );

    // -------------------------------------------------------------------------
    // Scoreboard
    // -------------------------------------------------------------------------
    class Scoreboard;
        int tests_run, tests_passed, tests_failed;

        function void reset_stats();
            tests_run    = 0;
            tests_passed = 0;
            tests_failed = 0;
        endfunction

        function void report();
            $display("\n============================================================");
            $display("INPUT_BINARIZE_TB RESULTS");
            $display("============================================================");
            $display("Tests Run:    %0d", tests_run);
            $display("Tests Passed: %0d", tests_passed);
            $display("Tests Failed: %0d", tests_failed);
            if (tests_failed == 0) $display("*** ALL TESTS PASSED ***");
            else                   $display("*** FAILURES DETECTED ***");
            $display("============================================================");
        endfunction
    endclass

    Scoreboard sb;

    // -------------------------------------------------------------------------
    // Golden model
    // -------------------------------------------------------------------------
    class GoldenModel;
        function automatic bit verify(
            input  logic [PIXELS*PIXEL_W-1:0] actual_pixels,
            input  logic [PIXELS-1:0]         actual_bits,
            output string                     error_msg
        );
            logic [PIXELS-1:0] expected_bits;
            bit pass;

            pass      = 1'b1;
            error_msg = "";

            for (int i = 0; i < PIXELS; i++) begin
                logic [PIXEL_W-1:0] px;
                px = actual_pixels[i*PIXEL_W +: PIXEL_W];
                expected_bits[i] = (px >= 8'd128);
            end

            for (int i = 0; i < PIXELS; i++) begin
                if (actual_bits[i] !== expected_bits[i]) begin
                    pass = 1'b0;
                    error_msg = {
                        error_msg,
                        $sformatf("idx=%0d got=%0b exp=%0b ", i, actual_bits[i], expected_bits[i])
                    };
                end
            end

            return pass;
        endfunction
    endclass

    GoldenModel golden;

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    task automatic run_case(
        input logic [PIXELS*PIXEL_W-1:0] vec,
        input string                     name
    );
        string err;

        sb.tests_run++;
        pixels_in = vec;
        #1;

        if (golden.verify(pixels_in, bits_out, err)) begin
            sb.tests_passed++;
            $display("[PASS] %s", name);
        end else begin
            sb.tests_failed++;
            $error("[FAIL] %s : %s", name, err);
        end
    endtask

    task automatic build_threshold_sweep(
        output logic [PIXELS*PIXEL_W-1:0] vec
    );
        vec = '0;

        // Covers below, equal, and above 128
        for (int i = 0; i < PIXELS; i++) begin
            logic [7:0] value;
            case (i)
                0:  value = 8'd0;
                1:  value = 8'd1;
                2:  value = 8'd63;
                3:  value = 8'd64;
                4:  value = 8'd127;
                5:  value = 8'd128;
                6:  value = 8'd129;
                7:  value = 8'd255;
                default: value = 8'(i * 17);
            endcase
            vec[i*PIXEL_W +: PIXEL_W] = value;
        end
    endtask

    task automatic build_random_vec(
        output logic [PIXELS*PIXEL_W-1:0] vec
    );
        vec = '0;
        for (int i = 0; i < PIXELS; i++) begin
            vec[i*PIXEL_W +: PIXEL_W] = $urandom_range(0,255);
        end
    endtask

    // -------------------------------------------------------------------------
    // Main
    // -------------------------------------------------------------------------
    initial begin
        logic [PIXELS*PIXEL_W-1:0] vec;

        sb     = new();
        golden = new();
        sb.reset_stats();

        pixels_in = '0;

        // all zeros
        vec = '0;
        run_case(vec, "all_zero_pixels");

        // all 255s
        vec = '0;
        for (int i = 0; i < PIXELS; i++) begin
            vec[i*PIXEL_W +: PIXEL_W] = 8'hFF;
        end
        run_case(vec, "all_fullscale_pixels");

        // threshold boundary coverage
        build_threshold_sweep(vec);
        run_case(vec, "threshold_boundary_sweep");

        // random tests
        for (int t = 0; t < 5; t++) begin
            build_random_vec(vec);
            run_case(vec, $sformatf("random_%0d", t));
        end

        sb.report();
        $finish;
    end

endmodule

`default_nettype wire