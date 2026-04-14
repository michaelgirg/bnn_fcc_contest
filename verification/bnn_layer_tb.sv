`default_nettype none
module bnn_layer_tb;
    //==========================================================================
    // Parameters
    //==========================================================================
    localparam int INPUTS = 784;
    localparam int NEURONS = 256;
    localparam int PW = 8;
    localparam int PN = 8;
    localparam int BEATS = (INPUTS + PW - 1) / PW;
    localparam int COUNT_W = $clog2(INPUTS + 1);
    localparam int NEURON_GROUPS = (NEURONS + PN - 1) / PN;
    localparam int WEIGHT_DEPTH = BEATS * NEURON_GROUPS;
    localparam int CFG_NIDX_W = (NEURONS > 1) ? $clog2(NEURONS) : 1;
    localparam int CFG_ADDR_W = (BEATS > 1) ? $clog2(BEATS) : 1;
    localparam int CLK_PERIOD_NS = 10;
    localparam int RANDOM_SEED = 12345;

    //==========================================================================
    // DUT Signals
    //==========================================================================
    logic clk, rst, start, done, busy;
    logic                       input_load;
    logic [         INPUTS-1:0] input_vector;
    logic                       output_valid;
    logic [        NEURONS-1:0] activations_out;
    logic [NEURONS*COUNT_W-1:0] popcounts_out;
    logic cfg_write_en, cfg_threshold_write, cfg_ready;
    logic [CFG_NIDX_W-1:0] cfg_neuron_idx;
    logic [CFG_ADDR_W-1:0] cfg_weight_addr;
    logic [        PW-1:0] cfg_weight_data;
    logic [   COUNT_W-1:0] cfg_threshold_data;

    //==========================================================================
    // Configuration Storage
    //==========================================================================
    class ConfigData;
        logic [     PW-1:0] weights   [NEURONS] [BEATS];
        logic [COUNT_W-1:0] thresholds[NEURONS];

        function void set_weight(int neuron, int beat, logic [PW-1:0] data);
            weights[neuron][beat] = data;
        endfunction

        function void set_threshold(int neuron, logic [COUNT_W-1:0] data);
            thresholds[neuron] = data;
        endfunction
    endclass

    //==========================================================================
    // Scoreboard
    //==========================================================================
    class Scoreboard;
        int     tests_run,   tests_passed, tests_failed,  x_errors;
        longint latency_sum;
        int     latency_min, latency_max,  latency_count;

        function void reset_stats();
            tests_run     = 0;
            tests_passed  = 0;
            tests_failed  = 0;
            x_errors      = 0;
            latency_sum   = 0;
            latency_min   = 32'h7FFF_FFFF;
            latency_max   = 0;
            latency_count = 0;
        endfunction

        function void record_latency(int cycles);
            latency_sum += cycles;
            latency_count++;
            if (cycles < latency_min) latency_min = cycles;
            if (cycles > latency_max) latency_max = cycles;
        endfunction

        function void report();
            real avg_lat, avg_lat_ns;
            $display("\n================================================================================");
            $display("FINAL RESULTS");
            $display("================================================================================");
            $display("Tests Run:      %0d", tests_run);
            $display("Tests Passed:   %0d (%.1f%%)", tests_passed,
                     tests_run ? 100.0 * tests_passed / tests_run : 0.0);
            $display("Tests Failed:   %0d", tests_failed);
            $display("X-State Errors: %0d", x_errors);
            $display("================================================================================");
            if (tests_failed == 0) $display("*** ALL TESTS PASSED ***");
            else $display("*** FAILURES DETECTED ***");
            $display("\n================================================================================");
            $display("LATENCY REPORT");
            $display("================================================================================");
            $display("Config:  INPUTS=%0d  NEURONS=%0d  PW=%0d  PN=%0d", INPUTS, NEURONS, PW, PN);
            $display("Theoretical min cycles: ~%0d (BEATS=%0d x GROUPS=%0d + drain)", BEATS * NEURON_GROUPS,
                     BEATS, NEURON_GROUPS);
            if (latency_count > 0) begin
                avg_lat    = real'(latency_sum) / real'(latency_count);
                avg_lat_ns = avg_lat * CLK_PERIOD_NS;
                $display("Measured (over %0d inferences):", latency_count);
                $display("  Min: %0d cycles (%.1f ns)", latency_min, real'(latency_min) * CLK_PERIOD_NS);
                $display("  Max: %0d cycles (%.1f ns)", latency_max, real'(latency_max) * CLK_PERIOD_NS);
                $display("  Avg: %.1f cycles (%.1f ns)", avg_lat, avg_lat_ns);
                $display("  Throughput: %.2f inferences/us", 1000.0 / avg_lat_ns);
            end else $display("  No passing inferences recorded.");
            $display("================================================================================");
        endfunction
    endclass

    //==========================================================================
    // Golden Model - FIXED to accept cfg_data as parameter
    //==========================================================================
    class GoldenModel;
        function automatic bit verify_outputs(
            input logic [INPUTS-1:0] input_vec, input logic [NEURONS-1:0] actual_act,
            input logic [NEURONS*COUNT_W-1:0] actual_pop_packed, input ConfigData cfg,  // PASS CONFIG DATA
            output string error_msg, output bit has_x_state);
            logic [COUNT_W-1:0] expected_pop[NEURONS];
            logic [COUNT_W-1:0] actual_pop[NEURONS];
            logic expected_act;
            bit pass = 1;
            int error_count = 0;
            has_x_state = 0;

            // Unpack actual popcounts
            for (int n = 0; n < NEURONS; n++) actual_pop[n] = actual_pop_packed[n*COUNT_W+:COUNT_W];

            // Calculate expected outputs
            for (int n = 0; n < NEURONS; n++) begin
                int popcount = 0;
                for (int b = 0; b < BEATS; b++) begin
                    for (int p = 0; p < PW; p++) begin
                        int inp_idx = b * PW + p;
                        if (inp_idx < INPUTS) popcount += (input_vec[inp_idx] == cfg.weights[n][b][p]);
                    end
                end
                expected_pop[n] = popcount;
            end

            // Check for X states
            for (int n = 0; n < NEURONS && error_count < 5; n++) begin
                if ($isunknown(actual_act[n])) begin
                    error_msg = {error_msg, $sformatf("N%0d:act=X ", n)};
                    pass = 0;
                    has_x_state = 1;
                    error_count++;
                end
                if ($isunknown(actual_pop[n])) begin
                    error_msg = {error_msg, $sformatf("N%0d:pop=X ", n)};
                    pass = 0;
                    has_x_state = 1;
                    error_count++;
                end
            end

            // Check values if no X states
            if (!has_x_state) begin
                for (int n = 0; n < NEURONS && error_count < 5; n++) begin
                    expected_act = (expected_pop[n] >= cfg.thresholds[n]);
                    if (actual_pop[n] !== expected_pop[n]) begin
                        error_msg = {
                            error_msg, $sformatf("N%0d:pop=%0d exp=%0d ", n, actual_pop[n], expected_pop[n])
                        };
                        pass = 0;
                        error_count++;
                    end
                    if (actual_act[n] !== expected_act) begin
                        error_msg = {
                            error_msg, $sformatf("N%0d:act=%0b exp=%0b ", n, actual_act[n], expected_act)
                        };
                        pass = 0;
                        error_count++;
                    end
                end
            end
            return pass;
        endfunction
    endclass

    ConfigData  cfg_data;
    Scoreboard  sb;
    GoldenModel golden;

    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    bnn_layer #(
        .INPUTS      (INPUTS),
        .NEURONS     (NEURONS),
        .PW          (PW),
        .PN          (PN),
        .OUTPUT_LAYER(0)
    ) dut (
        .*
    );

    //==========================================================================
    // Clock & Cycle Counter
    //==========================================================================
    always #(CLK_PERIOD_NS / 2) clk = ~clk;

    longint cycle_count;
    always_ff @(posedge clk) begin
        if (rst) cycle_count <= 0;
        else cycle_count <= cycle_count + 1;
    end

    //==========================================================================
    // Assertions
    //==========================================================================
    // synthesis translate_off
    property p_cfg_ready_when_idle;
        @(posedge clk) disable iff (rst) !busy |-> cfg_ready;
    endproperty
    assert property (p_cfg_ready_when_idle)
    else $error("[ASSERT] cfg_ready must be high when not busy");

    property p_done_pulse;
        @(posedge clk) disable iff (rst) done |=> !done;
    endproperty
    assert property (p_done_pulse)
    else $error("[ASSERT] done must be a single-cycle pulse");
    // synthesis translate_on

    //==========================================================================
    // Tasks
    //==========================================================================
    task reset_and_init_dut();
        $display("[%0t] Asserting reset...", $time);
        rst = 1;
        start = 0;
        input_load = 0;
        cfg_write_en = 0;
        cfg_threshold_write = 0;
        repeat (5) @(posedge clk);
        rst = 0;
        repeat (10) @(posedge clk);
        $display("[%0t] Reset released, DUT ready", $time);
    endtask

    task warm_reset();
        $display("[%0t] >>> WARM RESET (memories persist per spec) <<<", $time);
        rst = 1;
        start = 0;
        input_load = 0;
        cfg_write_en = 0;
        cfg_threshold_write = 0;
        repeat (5) @(posedge clk);
        rst = 0;
        repeat (5) @(posedge clk);
        $display("[%0t] >>> Warm reset complete <<<", $time);
    endtask

    task configure_weights();
        $display("[%0t] Configuring %0d weights...", $time, NEURONS * BEATS);
        while (!cfg_ready) @(posedge clk);
        for (int n = 0; n < NEURONS; n++) begin
            for (int b = 0; b < BEATS; b++) begin
                @(posedge clk);
                cfg_write_en = 1;
                cfg_threshold_write = 0;
                cfg_neuron_idx = n;
                cfg_weight_addr = b;
                cfg_weight_data = cfg_data.weights[n][b];
            end
        end
        @(posedge clk);
        cfg_write_en = 0;
        repeat (10) @(posedge clk);
        $display("[%0t] Weight configuration complete", $time);
    endtask

    task configure_thresholds();
        $display("[%0t] Configuring %0d thresholds...", $time, NEURONS);
        while (!cfg_ready) @(posedge clk);
        for (int n = 0; n < NEURONS; n++) begin
            @(posedge clk);
            cfg_write_en = 0;
            cfg_threshold_write = 1;
            cfg_neuron_idx = n;
            cfg_threshold_data = cfg_data.thresholds[n];
        end
        @(posedge clk);
        cfg_threshold_write = 0;
        repeat (10) @(posedge clk);
        $display("[%0t] Threshold configuration complete", $time);
    endtask

    task run_test(input logic [INPUTS-1:0] input_vec, input string test_name);
        string error_msg;
        bit timed_out, has_x;
        longint start_cycle, done_cycle;
        int elapsed;

        sb.tests_run++;
        $display("[%0t] Test: %s (ones=%0d/%0d)", $time, test_name, $countones(input_vec), INPUTS);

        // Load input
        @(posedge clk);
        input_vector = input_vec;
        input_load   = 1;
        @(posedge clk);
        input_load = 0;
        repeat (5) @(posedge clk);

        // Start inference with timeout
        fork
            begin
                @(posedge clk);
                start = 1;
                start_cycle = cycle_count;
                @(posedge clk);
                start = 0;
                wait (done);
                done_cycle = cycle_count;
                @(posedge clk);
            end
            begin
                repeat (100000) @(posedge clk);
                timed_out = 1;
            end
        join_any
        disable fork;

        if (timed_out) begin
            $error("[TIMEOUT] %s", test_name);
            sb.tests_failed++;
            return;
        end

        if (!output_valid) begin
            $error("[PROTOCOL] %s - done high but output_valid=0", test_name);
            sb.tests_failed++;
            return;
        end

        elapsed   = int'(done_cycle - start_cycle);
        error_msg = "";

        // PASS cfg_data to golden model
        if (golden.verify_outputs(
                input_vec, activations_out, popcounts_out, cfg_data, error_msg, has_x
            )) begin
            $display("[PASS] %s  latency=%0d cyc (%.1f ns)", test_name, elapsed,
                     real'(elapsed) * CLK_PERIOD_NS);
            sb.tests_passed++;
            sb.record_latency(elapsed);
        end else begin
            $error("[FAIL] %s: %s", test_name, error_msg);
            if (has_x) sb.x_errors++;
            sb.tests_failed++;
        end
    endtask

    task check_busy_protocol(input logic [INPUTS-1:0] input_vec, output bit pass);
        bit busy_seen, done_seen;
        int timeout_cnt;

        pass = 1;
        busy_seen = 0;
        done_seen = 0;
        timeout_cnt = 0;

        @(posedge clk);
        input_vector = input_vec;
        input_load   = 1;
        @(posedge clk);
        input_load = 0;
        repeat (3) @(posedge clk);

        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        // Wait for busy
        while (!busy && timeout_cnt < 20) begin
            @(posedge clk);
            timeout_cnt++;
        end
        if (!busy) begin
            $error("[PROTOCOL] busy never asserted");
            pass = 0;
            return;
        end
        busy_seen   = 1;

        // Wait for done
        timeout_cnt = 0;
        while (!done && timeout_cnt < 100000) begin
            @(posedge clk);
            timeout_cnt++;
        end
        if (timeout_cnt >= 100000) begin
            $error("[PROTOCOL] timeout waiting for done");
            pass = 0;
            return;
        end
        done_seen = 1;
        @(posedge clk);
        if (!busy_seen || !done_seen) pass = 0;
    endtask

    //==========================================================================
    // Main Test Sequence
    //==========================================================================
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        {input_load, start, cfg_write_en, cfg_threshold_write} = '0;
        {input_vector, cfg_neuron_idx, cfg_weight_addr, cfg_weight_data, cfg_threshold_data} = '0;

        cfg_data = new();
        golden = new();
        sb = new();
        sb.reset_stats();
        $srandom(RANDOM_SEED);

        $display("\n=== BNN Layer Testbench ===");
        $display("INPUTS=%0d, NEURONS=%0d, PW=%0d, PN=%0d", INPUTS, NEURONS, PW, PN);
        $display("Random seed: %0d\n", RANDOM_SEED);

        reset_and_init_dut();

        // Generate random config
        for (int n = 0; n < NEURONS; n++) begin
            for (int b = 0; b < BEATS; b++) cfg_data.weights[n][b] = $urandom();
            cfg_data.thresholds[n] = $urandom_range(INPUTS / 4, 3 * INPUTS / 4);
        end

        configure_weights();
        configure_thresholds();

        // TEST GROUP 1: Basic
        $display("\n=== TEST GROUP 1: Basic Functionality ===");
        for (int i = 0; i < 5; i++) run_test($urandom(), $sformatf("basic_%0d", i));

        // TEST GROUP 2: Corners
        $display("\n=== TEST GROUP 2: Corner Cases ===");
        run_test('0, "all_zeros");
        run_test('1, "all_ones");
        begin
            logic [INPUTS-1:0] v;
            for (int i = 0; i < INPUTS; i++) v[i] = i[0];
            run_test(v, "alternating_01");
        end

        // TEST GROUP 3: Back-to-back
        $display("\n=== TEST GROUP 3: Back-to-Back ===");
        for (int i = 0; i < 10; i++) run_test($urandom(), $sformatf("b2b_%0d", i));

        // TEST GROUP 4: Reconfig
        $display("\n=== TEST GROUP 4: Reconfiguration ===");
        run_test($urandom(), "pre_reconfig");
        $display("[%0t] Reconfiguring model...", $time);
        for (int n = 0; n < NEURONS; n++) begin
            for (int b = 0; b < BEATS; b++) cfg_data.weights[n][b] = $urandom();
            cfg_data.thresholds[n] = $urandom_range(INPUTS / 4, 3 * INPUTS / 4);
        end
        configure_weights();
        configure_thresholds();
        for (int i = 0; i < 3; i++) run_test($urandom(), $sformatf("post_reconfig_%0d", i));

        // TEST GROUP 5: Warm reset (IDLE)
        $display("\n=== TEST GROUP 5: Warm Reset (IDLE) ===");
        warm_reset();
        for (int i = 0; i < 3; i++) run_test($urandom(), $sformatf("warm_idle_%0d", i));

        // TEST GROUP 6: Warm reset (mid-inference)
        $display("\n=== TEST GROUP 6: Warm Reset (mid-inference) ===");
        begin
            logic [INPUTS-1:0] v = $urandom();
            @(posedge clk);
            input_vector = v;
            input_load   = 1;
            @(posedge clk);
            input_load = 0;
            repeat (3) @(posedge clk);
            @(posedge clk);
            start = 1;
            @(posedge clk);
            start = 0;
            repeat (10) @(posedge clk);
            warm_reset();
            sb.tests_run++;
            if (busy) begin
                $error("[WARM-RESET] busy still high after reset");
                sb.tests_failed++;
            end else begin
                $display("[PASS] busy correctly cleared");
                sb.tests_passed++;
            end
            for (int i = 0; i < 3; i++) run_test($urandom(), $sformatf("warm_mid_%0d", i));
        end

        // TEST GROUP 7: Protocol
        $display("\n=== TEST GROUP 7: Protocol Checks ===");
        for (int i = 0; i < 3; i++) begin
            bit proto_ok;
            check_busy_protocol($urandom(), proto_ok);
            sb.tests_run++;
            if (proto_ok) begin
                $display("[PASS] protocol_%0d", i);
                sb.tests_passed++;
            end else begin
                $error("[FAIL] protocol_%0d", i);
                sb.tests_failed++;
            end
        end

        // Final report
        repeat (50) @(posedge clk);
        sb.report();
        $display(sb.tests_failed == 0 ? "\n*** PASS ***\n" : "\n*** FAIL ***\n");
        $finish;
    end
endmodule
`default_nettype wire
