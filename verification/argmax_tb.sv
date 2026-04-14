`default_nettype none
module argmax_tb;
    localparam int OUTPUTS = 10;
    localparam int COUNT_W = 9;
    localparam int CLASS_W = (OUTPUTS > 1) ? $clog2(OUTPUTS) : 1;
    logic                       valid_in;
    logic [OUTPUTS*COUNT_W-1:0] popcounts_in;
    logic                       valid_out;
    logic [        CLASS_W-1:0] class_idx;
    logic [        COUNT_W-1:0] max_value;
    // -------------------------------------------------------------------------
    // DUT
    // -------------------------------------------------------------------------
    argmax #(
        .OUTPUTS(OUTPUTS),
        .COUNT_W(COUNT_W)
    ) dut (
        .valid_in    (valid_in),
        .popcounts_in(popcounts_in),
        .valid_out   (valid_out),
        .class_idx   (class_idx),
        .max_value   (max_value)
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
            $display("ARGMAX_TB RESULTS");
            $display("============================================================");
            $display("Tests Run:    %0d", tests_run);
            $display("Tests Passed: %0d", tests_passed);
            $display("Tests Failed: %0d", tests_failed);
            if (tests_failed == 0) $display("*** ALL TESTS PASSED ***");
            else $display("*** FAILURES DETECTED ***");
            $display("============================================================");
        endfunction
    endclass
    Scoreboard sb;
    // -------------------------------------------------------------------------
    // Golden model
    // -------------------------------------------------------------------------
    class GoldenModel;
        function automatic bit verify(input logic actual_valid_in,
                                      input logic [OUTPUTS*COUNT_W-1:0] actual_packed,
                                      input logic actual_valid_out, input logic [CLASS_W-1:0] actual_class,
                                      input logic [COUNT_W-1:0] actual_max, output string error_msg);
            logic [COUNT_W-1:0] vals      [OUTPUTS];
            logic [COUNT_W-1:0] exp_max;
            logic [CLASS_W-1:0] exp_class;
            logic               exp_valid;
            bit                 pass;
            pass      = 1'b1;
            error_msg = "";
            for (int i = 0; i < OUTPUTS; i++) begin
                vals[i] = actual_packed[i*COUNT_W+:COUNT_W];
            end
            exp_valid = actual_valid_in;
            exp_class = '0;
            exp_max   = vals[0];
            // Tie-break: first max wins, matching ">="
            for (int i = 1; i < OUTPUTS; i++) begin
                if (vals[i] > exp_max) begin
                    exp_max   = vals[i];
                    exp_class = CLASS_W'(i);
                end
            end
            if (actual_valid_out !== exp_valid) begin
                pass = 1'b0;
                error_msg = {error_msg, $sformatf("valid got=%0b exp=%0b ", actual_valid_out, exp_valid)};
            end
            if (actual_class !== exp_class) begin
                pass = 1'b0;
                error_msg = {error_msg, $sformatf("class got=%0d exp=%0d ", actual_class, exp_class)};
            end
            if (actual_max !== exp_max) begin
                pass = 1'b0;
                error_msg = {error_msg, $sformatf("max got=%0d exp=%0d ", actual_max, exp_max)};
            end
            return pass;
        endfunction
    endclass
    GoldenModel golden;
    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    task automatic run_case(input logic vin, input logic [OUTPUTS*COUNT_W-1:0] pack, input string name);
        string err;
        sb.tests_run++;
        valid_in     = vin;
        popcounts_in = pack;
        #1;
        if (golden.verify(valid_in, popcounts_in, valid_out, class_idx, max_value, err)) begin
            sb.tests_passed++;
            $display("[PASS] %s", name);
        end else begin
            sb.tests_failed++;
            $error("[FAIL] %s : %s", name, err);
        end
    endtask
    task automatic set_vals(output logic [OUTPUTS*COUNT_W-1:0] pack, input int unsigned vals[OUTPUTS]);
        pack = '0;
        for (int i = 0; i < OUTPUTS; i++) begin
            pack[i*COUNT_W+:COUNT_W] = COUNT_W'(vals[i]);
        end
    endtask
    // -------------------------------------------------------------------------
    // Main
    // -------------------------------------------------------------------------
    initial begin
        int unsigned vals[OUTPUTS];
        logic [OUTPUTS*COUNT_W-1:0] pack;
        sb     = new();
        golden = new();
        sb.reset_stats();
        $display("\n============================================================");
        $display("Starting ARGMAX Testbench");
        $display("OUTPUTS=%0d, COUNT_W=%0d, CLASS_W=%0d", OUTPUTS, COUNT_W, CLASS_W);
        $display("============================================================\n");
        valid_in     = 1'b0;
        popcounts_in = '0;
        // case 1: max at index 0 (beginning)
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 10;
        vals[0] = 400;
        set_vals(pack, vals);
        run_case(1'b1, pack, "max_at_index_0");
        // case 2: max at middle index
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 50;
        vals[5] = 255;
        set_vals(pack, vals);
        run_case(1'b1, pack, "max_at_index_5");
        // case 3: max at last index
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 20;
        vals[9] = 500;
        set_vals(pack, vals);
        run_case(1'b1, pack, "max_at_index_9");
        // case 4: increasing sequence -> last index wins
        for (int i = 0; i < OUTPUTS; i++) vals[i] = i * 10;
        set_vals(pack, vals);
        run_case(1'b1, pack, "increasing_sequence");
        // case 5: decreasing sequence -> first index wins
        for (int i = 0; i < OUTPUTS; i++) vals[i] = (OUTPUTS - i) * 10;
        set_vals(pack, vals);
        run_case(1'b1, pack, "decreasing_sequence");
        // case 6: single obvious max
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 10;
        vals[3] = 200;
        set_vals(pack, vals);
        run_case(1'b1, pack, "single_max_index_3");
        // case 7: tie at multiple indices -> lowest index should win
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 5;
        vals[2] = 100;
        vals[7] = 100;
        set_vals(pack, vals);
        run_case(1'b1, pack, "tie_indices_2_and_7");
        // case 8: tie at first two indices
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 15;
        vals[0] = 150;
        vals[1] = 150;
        set_vals(pack, vals);
        run_case(1'b1, pack, "tie_indices_0_and_1");
        // case 9: all same value -> index 0 should win
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 77;
        set_vals(pack, vals);
        run_case(1'b1, pack, "all_same_value");
        // case 10: all zero
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 0;
        set_vals(pack, vals);
        run_case(1'b1, pack, "all_zero");
        // case 11: maximum possible value
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 0;
        vals[4] = (1 << COUNT_W) - 1;
        set_vals(pack, vals);
        run_case(1'b1, pack, "max_possible_value");
        // case 12: alternating high/low
        for (int i = 0; i < OUTPUTS; i++) begin
            vals[i] = (i % 2 == 0) ? 10 : 200;
        end
        set_vals(pack, vals);
        run_case(1'b1, pack, "alternating_pattern");
        // case 13: valid passthrough low
        for (int i = 0; i < OUTPUTS; i++) vals[i] = $urandom_range(0, 255);
        set_vals(pack, vals);
        run_case(1'b0, pack, "valid_passthrough_low");
        // case 14: valid toggle test
        for (int i = 0; i < OUTPUTS; i++) vals[i] = i * 5;
        set_vals(pack, vals);
        run_case(1'b0, pack, "valid_low_before_high");
        run_case(1'b1, pack, "valid_high_after_low");
        // Random tests with detailed seeding
        $display("\n--- Starting Random Tests ---");
        for (int t = 0; t < 15; t++) begin
            for (int i = 0; i < OUTPUTS; i++) begin
                vals[i] = $urandom_range(0, (1 << COUNT_W) - 1);
            end
            set_vals(pack, vals);
            run_case(1'b1, pack, $sformatf("random_%0d", t));
        end
        // Edge case: sparse values
        for (int i = 0; i < OUTPUTS; i++) vals[i] = 0;
        vals[1] = 1;
        vals[8] = 2;
        set_vals(pack, vals);
        run_case(1'b1, pack, "sparse_nonzero");
        sb.report();
        $finish;
    end
endmodule
`default_nettype wire
