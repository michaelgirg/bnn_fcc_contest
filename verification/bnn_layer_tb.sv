module bnn_layer_tb;
    localparam int INPUTS = 32;
    localparam int NEURONS = 32;
    localparam int PW = 16;
    localparam int PN = 4;
    localparam bit OUTPUT_LAYER = 0;
    localparam int NUM_TESTS = 100;

    localparam int COUNT_WIDTH = $clog2(INPUTS + 1);
    localparam int INPUT_BEATS = (INPUTS + PW - 1) / PW;
    localparam int NEURON_GROUPS = (NEURONS + PN - 1) / PN;
    localparam int CFG_NEURON_WIDTH = (NEURONS > 1) ? $clog2(NEURONS) : 1;
    localparam int CFG_WEIGHT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1;

    logic clk = 0;
    logic rst = 1;
    logic input_load;
    logic [INPUTS-1:0] input_vector;
    logic start;
    logic done;
    logic busy;
    logic [NEURONS-1:0] activations_out;
    logic [NEURONS*COUNT_WIDTH-1:0] popcounts_out;
    logic output_valid;
    logic cfg_write_en;
    logic [CFG_NEURON_WIDTH-1:0] cfg_neuron_idx;
    logic [CFG_WEIGHT_ADDR_WIDTH-1:0] cfg_weight_addr;
    logic [PW-1:0] cfg_weight_data;
    logic [COUNT_WIDTH-1:0] cfg_threshold_data;
    logic cfg_threshold_write;
    logic cfg_ready;

    logic [PW-1:0] weights[NEURONS][INPUT_BEATS];
    logic [COUNT_WIDTH-1:0] thresholds[NEURONS];
    logic [INPUTS-1:0] input_data;
    int passed = 0;
    int failed = 0;

    bnn_layer #(
        .INPUTS      (INPUTS),
        .NEURONS     (NEURONS),
        .PW          (PW),
        .PN          (PN),
        .OUTPUT_LAYER(OUTPUT_LAYER)
    ) dut (
        .*
    );

    always #5 clk = ~clk;

    function automatic int calc_popcount(int n);
        automatic int count = 0;
        for (int b = 0; b < INPUT_BEATS; b++) begin
            for (int bit_pos = 0; bit_pos < PW; bit_pos++) begin
                automatic int idx = b * PW + bit_pos;
                if (idx < INPUTS) begin
                    if ((input_data[idx] ~^ weights[n][b][bit_pos]) == 1'b1) count++;
                end
            end
        end
        return count;
    endfunction

    function automatic logic calc_activation(int n);
        return (calc_popcount(n) >= thresholds[n]) ? 1'b1 : 1'b0;
    endfunction

    task reset_dut();
        rst = 1;
        input_load = 0;
        input_vector = '0;
        start = 0;
        cfg_write_en = 0;
        cfg_neuron_idx = '0;
        cfg_weight_addr = '0;
        cfg_weight_data = '0;
        cfg_threshold_data = '0;
        cfg_threshold_write = 0;
        repeat (10) @(posedge clk);
        rst = 0;
        repeat (5) @(posedge clk);
    endtask

    task write_config();
        for (int n = 0; n < NEURONS; n++) begin
            for (int b = 0; b < INPUT_BEATS; b++) begin
                @(posedge clk);
                cfg_write_en = 1;
                cfg_threshold_write = 0;
                cfg_neuron_idx = n;
                cfg_weight_addr = b;
                cfg_weight_data = weights[n][b];
            end
        end

        for (int n = 0; n < NEURONS; n++) begin
            @(posedge clk);
            cfg_write_en = 1;
            cfg_threshold_write = 1;
            cfg_neuron_idx = n;
            cfg_threshold_data = thresholds[n];
        end

        @(posedge clk);
        cfg_write_en = 0;
        cfg_threshold_write = 0;
    endtask

    task run_test();
        @(posedge clk);
        input_load   = 1;
        input_vector = input_data;
        @(posedge clk);
        input_load = 0;

        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;

        wait (done);
        @(posedge clk);
    endtask

    task check_results();
        automatic logic test_pass = 1;

        if (!output_valid) begin
            $display("ERROR: output_valid not asserted when done");
            test_pass = 0;
        end

        for (int n = 0; n < NEURONS; n++) begin
            automatic int   expected_pop = calc_popcount(n);
            automatic logic expected_act = calc_activation(n);

            if (OUTPUT_LAYER) begin
                automatic int actual_pop = 0;
                for (int b = 0; b < COUNT_WIDTH; b++) begin
                    actual_pop[b] = popcounts_out[n*COUNT_WIDTH+b];
                end
                if (actual_pop != expected_pop) begin
                    $display("ERROR: Neuron %0d popcount mismatch. Expected=%0d, Got=%0d", n, expected_pop,
                             actual_pop);
                    test_pass = 0;
                end
            end else begin
                if (activations_out[n] != expected_act) begin
                    $display(
                        "ERROR: Neuron %0d activation mismatch. Expected=%0b, Got=%0b, Popcount=%0d, Threshold=%0d",
                        n, expected_act, activations_out[n], expected_pop, thresholds[n]);
                    test_pass = 0;
                end
            end
        end

        if (test_pass) passed++;
        else failed++;
    endtask

    task randomize_test();
        input_data = $urandom();
        for (int n = 0; n < NEURONS; n++) begin
            for (int b = 0; b < INPUT_BEATS; b++) begin
                weights[n][b] = $urandom();
            end
            thresholds[n] = $urandom_range(0, INPUTS);
        end
    endtask

    initial begin
        $display("========================================");
        $display("BNN Layer Testbench");
        $display("========================================");

        reset_dut();

        for (int t = 0; t < NUM_TESTS; t++) begin
            randomize_test();
            write_config();
            run_test();
            check_results();

            if ((t + 1) % 10 == 0) begin
                $display("Completed %0d/%0d tests (Passed: %0d, Failed: %0d)", t + 1, NUM_TESTS, passed,
                         failed);
            end
        end

        $display("========================================");
        $display("RESULTS: %0d PASSED, %0d FAILED", passed, failed);
        if (failed == 0) $display("SUCCESS!");
        else $display("FAILURE!");
        $display("========================================");
        $finish;
    end

    initial begin
        #10000000;
        $display("TIMEOUT!");
        $finish;
    end

endmodule
