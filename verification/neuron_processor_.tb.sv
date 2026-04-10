`default_nettype none
// ============================================================================
// Module: neuron_processor_tb
// Purpose: Comprehensive testbench for neuron_processor with coverage
// Added:   per-neuron latency tracking and summary report
// ============================================================================
module neuron_processor_tb #(
    parameter int PW                         = 8,
    parameter int MAX_NEURON_INPUTS          = 784,
    parameter bit OUTPUT_LAYER               = 0,
    parameter int NUM_TESTS                  = 500,
    parameter int MIN_BEATS                  = 1,
    parameter int MAX_BEATS                  = 10,
    parameter bit TOGGLE_INPUTS_WHILE_ACTIVE = 1'b1,
    parameter int MIN_CYCLES_BETWEEN_NEURONS = 0,
    parameter int MAX_CYCLES_BETWEEN_NEURONS = 5,
    parameter bit LOG_START_MONITOR          = 1'b0,
    parameter bit LOG_DONE_MONITOR           = 1'b0,
    parameter int CLK_PERIOD_NS              = 10
);
    localparam int ACCUM_WIDTH = $clog2(MAX_NEURON_INPUTS + 1);

    logic                   clk = 0;
    logic                   rst;
    logic                   valid_in;
    logic                   last;
    logic [         PW-1:0] x;
    logic [         PW-1:0] w;
    logic [ACCUM_WIDTH-1:0] threshold;
    logic                   valid_out;
    logic                   activation;
    logic [ACCUM_WIDTH-1:0] popcount_out;

    int passed, failed;

    event   start_event;
    event   done_event;

    mailbox driver_mailbox = new;
    mailbox scoreboard_input_mailbox = new;
    mailbox scoreboard_result_mailbox = new;

    // Latency mailbox: driver sends (start_cycle, num_beats) per neuron
    // scoreboard reads it to compute latency per test
    typedef struct {
        longint start_cycle;
        int     num_beats;
    } timing_item_t;
    mailbox #(timing_item_t) timing_mailbox = new;

    // =========================================================================
    // Free-running cycle counter
    // =========================================================================
    longint cycle_count;
    always_ff @(posedge clk) begin
        if (rst) cycle_count <= 0;
        else cycle_count <= cycle_count + 1;
    end

    // =========================================================================
    // DUT
    // =========================================================================
    neuron_processor #(
        .PW               (PW),
        .MAX_NEURON_INPUTS(MAX_NEURON_INPUTS),
        .OUTPUT_LAYER     (OUTPUT_LAYER)
    ) DUT (
        .*
    );

    // =========================================================================
    // Transaction class
    // =========================================================================
    class neuron_item;
        rand int                   num_beats;
        rand bit [         PW-1:0] x_data    [];
        rand bit [         PW-1:0] w_data    [];
        rand bit [ACCUM_WIDTH-1:0] threshold;

        constraint valid_beats {num_beats inside {[MIN_BEATS : MAX_BEATS]};}
        constraint array_sizes {
            x_data.size() == num_beats;
            w_data.size() == num_beats;
        }

        function int calc_expected_popcount();
            int total = 0;
            for (int i = 0; i < num_beats; i++) begin
                bit [PW-1:0] xnor_result = x_data[i] ~^ w_data[i];
                total += $countones(xnor_result);
            end
            return total;
        endfunction

        function bit calc_expected_activation();
            return (calc_expected_popcount() >= threshold);
        endfunction
    endclass

    // =========================================================================
    // Latency statistics class
    // =========================================================================
    class LatencyStats;
        longint total_cycles;
        int     count;
        int     min_cycles;
        int     max_cycles;

        function new();
            total_cycles = 0;
            count        = 0;
            min_cycles   = int'(32'h7FFF_FFFF);
            max_cycles   = 0;
        endfunction

        function void record(int cycles);
            total_cycles += cycles;
            count++;
            if (cycles < min_cycles) min_cycles = cycles;
            if (cycles > max_cycles) max_cycles = cycles;
        endfunction

        function void report();
            real avg_cycles, avg_ns;
            $display("\n================================================================");
            $display("LATENCY REPORT  (first valid_in -> valid_out, passing tests only)");
            $display("================================================================");
            $display("Config: PW=%0d  MAX_NEURON_INPUTS=%0d  OUTPUT_LAYER=%0d", PW, MAX_NEURON_INPUTS,
                     OUTPUT_LAYER);
            $display("NP pipeline depth: 3 stages (Stage A: XNOR, Stage B: popcount, Stage C: accum)");
            $display("Expected min latency for N-beat neuron: N + 3 cycles");
            $display("  Single-beat : %0d cycles", 1 + 3);
            $display("  Max-beat(%0d): %0d cycles", MAX_BEATS, MAX_BEATS + 3);
            if (count > 0) begin
                avg_cycles = real'(total_cycles) / real'(count);
                avg_ns     = avg_cycles * CLK_PERIOD_NS;
                $display("Measured (%0d passing neurons):", count);
                $display("  Min latency : %0d cycles  (%0.1f ns)", min_cycles,
                         real'(min_cycles) * CLK_PERIOD_NS);
                $display("  Max latency : %0d cycles  (%0.1f ns)", max_cycles,
                         real'(max_cycles) * CLK_PERIOD_NS);
                $display("  Avg latency : %.2f cycles  (%.1f ns)", avg_cycles, avg_ns);
            end else begin
                $display("  No passing neurons recorded.");
            end
            $display("================================================================\n");
        endfunction
    endclass

    LatencyStats lat_stats;

    // =========================================================================
    // Reference model functions
    // =========================================================================
    function int model_popcount(neuron_item item);
        return item.calc_expected_popcount();
    endfunction

    function bit model_activation(neuron_item item);
        return item.calc_expected_activation();
    endfunction

    // =========================================================================
    // Clock
    // =========================================================================
    initial begin : generate_clock
        forever #(CLK_PERIOD_NS / 2) clk <= ~clk;
    end

    // =========================================================================
    // Initialization
    // =========================================================================
    initial begin : initialization
        $timeformat(-9, 0, " ns");
        lat_stats = new();
        rst       <= 1;
        valid_in  <= 0;
        last      <= 0;
        x         <= '0;
        w         <= '0;
        threshold <= '0;
        repeat (5) @(posedge clk);
        rst <= 0;
    end

    // =========================================================================
    // Generator
    // =========================================================================
    initial begin : generator
        neuron_item test;

        for (int i = 0; i < NUM_TESTS; i++) begin
            test = new();
            assert (test.randomize());
            driver_mailbox.put(test);
        end

        // ------------------------------------------------------------------
        // Directed corner cases
        // ------------------------------------------------------------------
        test = new();
        test.num_beats = 2;
        test.x_data = new[2];
        test.w_data = new[2];
        test.x_data[0] = 8'hAA;
        test.w_data[0] = 8'h55;
        test.x_data[1] = 8'hFF;
        test.w_data[1] = 8'hFF;
        test.threshold = PW;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = 3;
        test.x_data = new[3];
        test.w_data = new[3];
        for (int i = 0; i < 3; i++) begin
            test.x_data[i] = $urandom();
            test.w_data[i] = $urandom();
        end
        test.threshold = (3 * PW) / 2;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = 4;
        test.x_data = new[4];
        test.w_data = new[4];
        for (int i = 0; i < 4; i++) begin
            test.x_data[i] = 8'hAA;
            test.w_data[i] = 8'h55;
        end
        test.threshold = PW * 2;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = 5;
        test.x_data = new[5];
        test.w_data = new[5];
        for (int i = 0; i < 5; i++) begin
            test.x_data[i] = $urandom();
            test.w_data[i] = $urandom();
        end
        test.threshold = (5 * PW) / 2;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = 8;
        test.x_data = new[8];
        test.w_data = new[8];
        for (int i = 0; i < 8; i++) begin
            test.x_data[i] = 8'hAA;
            test.w_data[i] = 8'h55;
        end
        test.threshold = 20;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = MAX_BEATS;
        test.x_data = new[MAX_BEATS];
        test.w_data = new[MAX_BEATS];
        for (int i = 0; i < MAX_BEATS; i++) begin
            test.x_data[i] = 8'hFF;
            test.w_data[i] = 8'hFF;
        end
        test.threshold = (MAX_BEATS * PW) / 2;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = 2;
        test.x_data = new[2];
        test.w_data = new[2];
        test.x_data[0] = 8'b11110000;
        test.w_data[0] = 8'b11110000;
        test.x_data[1] = 8'b11000000;
        test.w_data[1] = 8'b11000000;
        test.threshold = 6;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = 1;
        test.x_data = new[1];
        test.w_data = new[1];
        test.x_data[0] = 8'b11111110;
        test.w_data[0] = 8'b11111110;
        test.threshold = 6;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = 1;
        test.x_data = new[1];
        test.w_data = new[1];
        test.x_data[0] = 8'b11111000;
        test.w_data[0] = 8'b11111000;
        test.threshold = 6;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = 1;
        test.x_data = new[1];
        test.w_data = new[1];
        test.x_data[0] = 8'h00;
        test.w_data[0] = 8'h00;
        test.threshold = 0;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = MAX_BEATS;
        test.x_data = new[MAX_BEATS];
        test.w_data = new[MAX_BEATS];
        for (int i = 0; i < MAX_BEATS; i++) begin
            test.x_data[i] = 8'hFF;
            test.w_data[i] = 8'hFF;
        end
        test.threshold = MAX_BEATS * PW;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = MAX_BEATS / 2;
        test.x_data = new[MAX_BEATS / 2];
        test.w_data = new[MAX_BEATS / 2];
        for (int i = 0; i < MAX_BEATS / 2; i++) begin
            test.x_data[i] = 8'hFF;
            test.w_data[i] = 8'hFF;
        end
        test.threshold = (MAX_BEATS * PW) / 4;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = MAX_BEATS;
        test.x_data = new[MAX_BEATS];
        test.w_data = new[MAX_BEATS];
        for (int i = 0; i < MAX_BEATS - 1; i++) begin
            test.x_data[i] = 8'hFF;
            test.w_data[i] = 8'hFF;
        end
        test.x_data[MAX_BEATS-1] = 8'hFE;
        test.w_data[MAX_BEATS-1] = 8'hFE;
        test.threshold = MAX_BEATS * PW - 5;
        driver_mailbox.put(test);

        test = new();
        test.num_beats = MAX_BEATS;
        test.x_data = new[MAX_BEATS];
        test.w_data = new[MAX_BEATS];
        for (int i = 0; i < MAX_BEATS; i++) begin
            test.x_data[i] = 8'hFF;
            test.w_data[i] = 8'hFF;
        end
        test.threshold = 1;
        driver_mailbox.put(test);

        for (int i = 0; i < 5; i++) begin
            test = new();
            test.num_beats = 1;
            test.x_data = new[1];
            test.w_data = new[1];
            test.x_data[0] = 8'hFF;
            test.w_data[0] = 8'hFF;
            test.threshold = 4;
            driver_mailbox.put(test);
        end

        for (int i = 0; i < 5; i++) begin
            test = new();
            test.num_beats = 1;
            test.x_data = new[1];
            test.w_data = new[1];
            test.x_data[0] = $urandom();
            test.w_data[0] = $urandom();
            test.threshold = PW / 2;
            driver_mailbox.put(test);
        end
    end

    // =========================================================================
    // Start Monitor
    // =========================================================================
    initial begin : start_monitor
        forever begin
            @(posedge clk iff !rst && valid_in);
            if (LOG_START_MONITOR) $display("[%0t] Start monitor: neuron processing started", $realtime);
            ->start_event;
        end
    end

    // =========================================================================
    // Done Monitor
    // =========================================================================
    initial begin : done_monitor
        forever begin
            @(posedge clk iff !rst && valid_out);
            if (LOG_DONE_MONITOR)
                $display(
                    "[%0t] Done monitor: result = %0d, activation = %0b", $realtime, popcount_out, activation
                );
            ->done_event;
        end
    end

    // =========================================================================
    // Driver
    // =========================================================================
    initial begin : driver
        neuron_item   item;
        timing_item_t timing;
        automatic bit first_test = 1'b1;
        automatic int test_count = 0;

        @(posedge clk iff !rst);

        forever begin
            driver_mailbox.get(item);
            test_count++;

            if (!first_test) begin
                if (test_count > NUM_TESTS + 10) begin
                    // No delay for rapid sequence tests
                end else begin
                    repeat ($urandom_range(MIN_CYCLES_BETWEEN_NEURONS, MAX_CYCLES_BETWEEN_NEURONS))
                        @(posedge clk);
                end
            end
            first_test = 1'b0;

            scoreboard_input_mailbox.put(item);

            // Record the cycle on which first valid_in fires
            timing.start_cycle = cycle_count;
            timing.num_beats   = item.num_beats;
            timing_mailbox.put(timing);

            for (int beat = 0; beat < item.num_beats; beat++) begin
                valid_in  <= 1;
                last      <= (beat == item.num_beats - 1);
                x         <= item.x_data[beat];
                w         <= item.w_data[beat];
                threshold <= item.threshold;
                @(posedge clk);

                if (TOGGLE_INPUTS_WHILE_ACTIVE &&
                    beat < item.num_beats - 1 &&
                    test_count != (NUM_TESTS + 6) &&
                    test_count != (NUM_TESTS + 15)) begin
                    valid_in <= 0;
                    x <= $urandom();
                    w <= $urandom();
                    @(posedge clk);
                end
            end

            valid_in <= 0;
            last     <= 0;
            @(posedge clk iff valid_out);
            scoreboard_result_mailbox.put({popcount_out, activation});
        end
    end

    // =========================================================================
    // Scoreboard — checks correctness and records latency
    // =========================================================================
    initial begin : scoreboard
        neuron_item                     item;
        timing_item_t                   timing;
        logic         [  ACCUM_WIDTH:0] result_packet;
        int                             expected_popcount;
        bit                             expected_activation;
        logic         [ACCUM_WIDTH-1:0] actual_popcount;
        logic                           actual_activation;
        int                             elapsed;
        longint                         done_cycle;

        passed = 0;
        failed = 0;

        for (int i = 0; i < NUM_TESTS + 24; i++) begin
            scoreboard_input_mailbox.get(item);
            scoreboard_result_mailbox.get(result_packet);
            timing_mailbox.get(timing);

            // Cycle when valid_out fired — approximate from current time
            // (valid_out was sampled by driver before putting result_packet)
            done_cycle = cycle_count;
            elapsed    = int'(done_cycle - timing.start_cycle);

            actual_popcount   = result_packet[ACCUM_WIDTH:1];
            actual_activation = result_packet[0];
            expected_popcount = model_popcount(item);
            expected_activation = model_activation(item);

            if (OUTPUT_LAYER) begin
                if (actual_popcount == expected_popcount) begin
                    $display("Test %0d passed (time %0t): popcount=%0d  latency=%0d cycles", i, $time,
                             actual_popcount, elapsed);
                    passed++;
                    lat_stats.record(elapsed);
                end else begin
                    $display("Test %0d FAILED (time %0t): popcount=%0d expected=%0d", i, $time,
                             actual_popcount, expected_popcount);
                    failed++;
                end
            end else begin
                if (actual_popcount == expected_popcount && actual_activation == expected_activation) begin
                    $display("Test %0d passed (time %0t): popcount=%0d act=%0b  latency=%0d cycles", i,
                             $time, actual_popcount, actual_activation, elapsed);
                    passed++;
                    lat_stats.record(elapsed);
                end else begin
                    $display("Test %0d FAILED (time %0t): popcount=%0d/%0d act=%0b/%0b", i, $time,
                             actual_popcount, expected_popcount, actual_activation, expected_activation);
                    failed++;
                end
            end
        end

        $display("========================================");
        $display("Tests completed: %0d passed, %0d failed", passed, failed);
        $display("========================================");

        lat_stats.report();

        $finish;
    end

    // =========================================================================
    // Assertions
    // =========================================================================
    property p_valid_out_timing;
        @(posedge clk) disable iff (rst) valid_in && last |-> ##3 valid_out;
    endproperty
    assert property (p_valid_out_timing)
    else $error("valid_out not asserted 3 cycles after last");

    property p_accumulator_clear;
        @(posedge clk) disable iff (rst) valid_in && last |-> ##4 (DUT.accumulator_r == 0);
    endproperty
    assert property (p_accumulator_clear)
    else $error("Accumulator not cleared");

    property p_valid_out_pulse;
        @(posedge clk) disable iff (rst) valid_out |=> !valid_out;
    endproperty
    assert property (p_valid_out_pulse)
    else $error("valid_out held high too long");

    // =========================================================================
    // Coverage
    // =========================================================================
    covergroup cg_neuron @(done_event);
        option.per_instance = 1;
        coverpoint popcount_out {
            bins zero = {0};
            bins low = {[1 : PW - 1]};
            bins mid = {[PW : 2 * PW - 1]};
            bins high = {[2 * PW : MAX_BEATS * PW]};
        }
        coverpoint activation {bins inactive = {0}; bins active = {1};}
    endgroup
    cg_neuron cov_neuron = new();

    single_beat_neuron :
    cover property (@(posedge clk) disable iff (rst) valid_in && last);
    multi_beat_neuron :
    cover property (@(posedge clk) disable iff (rst) valid_in && !last ##1 valid_in && last);
    three_beat_neuron :
    cover property (@(posedge clk) disable iff (rst)
        valid_in && !last ##1 valid_in && !last ##1 valid_in && last);
    max_beat_neuron :
    cover property (@(posedge clk) disable iff (rst)
        valid_in && !last [*MAX_BEATS-1] ##1 valid_in && last);
    back_to_back_neurons :
    cover property (@(posedge clk) disable iff (rst) valid_out ##1 valid_in);
    activation_high :
    cover property (@(posedge clk) disable iff (rst) valid_out && activation);
    activation_low :
    cover property (@(posedge clk) disable iff (rst) valid_out && !activation);
    popcount_zero :
    cover property (@(posedge clk) disable iff (rst) valid_out && (popcount_out == 0));
    popcount_max_single_beat :
    cover property (@(posedge clk) disable iff (rst) valid_out && (popcount_out == PW));
    popcount_half_max :
    cover property (@(posedge clk) disable iff (rst) valid_out && (popcount_out == (MAX_BEATS * PW) / 2));
    popcount_near_max :
    cover property (@(posedge clk) disable iff (rst) valid_out && (popcount_out >= (MAX_BEATS * PW - 2)));
    threshold_exact_match :
    cover property (@(posedge clk) disable iff (rst) valid_out && (popcount_out == threshold));
    threshold_one_below :
    cover property (@(posedge clk) disable iff (rst) valid_out && (popcount_out == threshold - 1));
    threshold_one_above :
    cover property (@(posedge clk) disable iff (rst) valid_out && (popcount_out == threshold + 1));
    threshold_at_zero :
    cover property (@(posedge clk) disable iff (rst) valid_in && (threshold == 0));
    threshold_at_max :
    cover property (@(posedge clk) disable iff (rst) valid_in && (threshold == MAX_BEATS * PW));
    all_inputs_match_weights :
    cover property (@(posedge clk) disable iff (rst) valid_in && (x == w));
    no_inputs_match_weights :
    cover property (@(posedge clk) disable iff (rst) valid_in && (x == ~w));
    all_ones_input :
    cover property (@(posedge clk) disable iff (rst) valid_in && (x == '1));
    all_zeros_input :
    cover property (@(posedge clk) disable iff (rst) valid_in && (x == '0));
    accumulator_nonzero_before_clear :
    cover property (@(posedge clk) disable iff (rst)
        (DUT.accumulator_r > 0) && DUT.stage_b_valid_r && DUT.stage_b_last_r);
    accumulator_at_max_before_clear :
    cover property (@(posedge clk) disable iff (rst)
        (DUT.accumulator_r >= (MAX_BEATS - 1) * PW) && DUT.stage_b_valid_r && DUT.stage_b_last_r);
    accumulator_cleared :
    cover property (@(posedge clk) disable iff (rst) valid_out ##1 (DUT.accumulator_r == 0));
    full_pipeline_propagation :
    cover property (@(posedge clk) disable iff (rst) valid_in && last ##3 valid_out);
    continuous_operation :
    cover property (@(posedge clk) disable iff (rst)
        (valid_in && !last) ##1 (valid_in && !last) ##1
        (valid_in && !last) ##1 (valid_in && !last) ##1 valid_in);
    rapid_neuron_sequence :
    cover property (@(posedge clk) disable iff (rst) valid_out ##[0:1] valid_out ##[0:1] valid_out);

    if (!OUTPUT_LAYER) begin : hidden_layer_coverage
        hidden_layer_activate_at_threshold :
        cover property (@(posedge clk) disable iff (rst)
            valid_out && activation && (popcount_out == threshold));
        hidden_layer_activate_above_threshold :
        cover property (@(posedge clk) disable iff (rst)
            valid_out && activation && (popcount_out > threshold));
        hidden_layer_no_activate_below_threshold :
        cover property (@(posedge clk) disable iff (rst)
            valid_out && !activation && (popcount_out < threshold));
    end : hidden_layer_coverage

endmodule
`default_nettype wire
