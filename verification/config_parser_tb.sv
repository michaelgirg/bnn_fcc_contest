// verification/config_parser_tb.sv
//
// Unit testbench for config_parser Phase 1.
// Drives AXI4-Stream config messages and checks header decoding.
//
// Run with:
//   vlog -sv rtl/config_parser.sv verification/config_parser_tb.sv
//   vsim work.config_parser_tb
//   run -all

`timescale 1ns/1ps

module config_parser_tb;

    // -----------------------------------------------------------------------
    // Parameters — match your DUT
    // -----------------------------------------------------------------------
    localparam int CLK_PERIOD       = 10;   // ns
    localparam int CONFIG_BUS_WIDTH = 64;
    localparam int BYTES_PER_BEAT   = CONFIG_BUS_WIDTH / 8;

    // -----------------------------------------------------------------------
    // DUT signals
    // -----------------------------------------------------------------------
    logic clk = 0, rst;

    logic                          config_valid;
    logic                          config_ready;
    logic [CONFIG_BUS_WIDTH-1:0]   config_data;
    logic [BYTES_PER_BEAT-1:0]     config_keep;
    logic                          config_last;

    logic        header_valid;
    logic [7:0]  o_msg_type;
    logic [7:0]  o_layer_id;
    logic [15:0] o_layer_inputs;
    logic [15:0] o_num_neurons;
    logic [15:0] o_bytes_per_neuron;
    logic [31:0] o_total_bytes;
    logic [31:0] payload_byte_count;
    logic        config_done;

    // -----------------------------------------------------------------------
    // DUT instantiation
    // -----------------------------------------------------------------------
    config_parser #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH)
    ) dut (.*);

    // -----------------------------------------------------------------------
    // Clock
    // -----------------------------------------------------------------------
    always #(CLK_PERIOD/2) clk = ~clk;

    // -----------------------------------------------------------------------
    // Test bookkeeping
    // -----------------------------------------------------------------------
    int pass_count = 0, fail_count = 0;

    task automatic expect32(
        input string       name,
        input logic [31:0] got,
        input logic [31:0] expected
    );
        if (got === expected) begin
            $display("  [PASS] %-30s got=0x%08X", name, got);
            pass_count++;
        end else begin
            $display("  [FAIL] %-30s expected=0x%08X  got=0x%08X",
                     name, expected, got);
            fail_count++;
        end
    endtask

    // -----------------------------------------------------------------------
    // AXI-Stream beat driver
    // Drives signals after negedge (setup time before next posedge).
    // Handles backpressure via config_ready poll.
    // -----------------------------------------------------------------------
    task automatic send_beat(
        input logic [CONFIG_BUS_WIDTH-1:0] data,
        input logic [BYTES_PER_BEAT-1:0]   keep,
        input logic                        last
    );
        @(negedge clk);
        config_data  = data;
        config_keep  = keep;
        config_last  = last;
        config_valid = 1'b1;

        // Spin until DUT accepts (config_ready at posedge)
        @(posedge clk);
        while (!config_ready) @(posedge clk);

        // Transaction committed — deassert after falling edge
        @(negedge clk);
        config_valid = 1'b0;
        config_last  = 1'b0;
    endtask

    // -----------------------------------------------------------------------
    // Full config message sender
    //   - builds the two header beats
    //   - packs payload bytes into CONFIG_BUS_WIDTH beats
    //   - sets TKEEP and TLAST correctly on the final beat
    // -----------------------------------------------------------------------
    task automatic send_message(
        input logic [7:0]  msg_type,
        input logic [7:0]  layer_id,
        input logic [15:0] layer_inputs_p,
        input logic [15:0] num_neurons_p,
        input logic [15:0] bytes_per_neuron_p,
        input logic [31:0] total_bytes_p,
        input logic [7:0]  payload [],  // dynamic array of payload bytes
        input int          payload_len
    );
        logic [CONFIG_BUS_WIDTH-1:0] word;
        logic [BYTES_PER_BEAT-1:0]   keep;
        int bi;         // byte index into payload
        int beat_bytes; // valid bytes in this beat

        // --- Header Word 0 ---
        word = {bytes_per_neuron_p, num_neurons_p, layer_inputs_p,
                layer_id, msg_type};
        send_beat(word, {BYTES_PER_BEAT{1'b1}}, 1'b0);

        // --- Header Word 1 ---
        //   [31:0]  = total_bytes
        //   [63:32] = reserved
        word = {32'h0, total_bytes_p};
        send_beat(word, {BYTES_PER_BEAT{1'b1}}, 1'b0);

        // --- Payload beats ---
        bi = 0;
        while (bi < payload_len) begin
            word      = '0;
            keep      = '0;
            beat_bytes = ((payload_len - bi) >= BYTES_PER_BEAT)
                         ? BYTES_PER_BEAT : (payload_len - bi);

            for (int i = 0; i < beat_bytes; i++) begin
                word[i*8 +: 8] = payload[bi + i];
                keep[i]        = 1'b1;
            end
            bi += beat_bytes;
            send_beat(word, keep, (bi >= payload_len)); // TLAST on final beat
        end
    endtask

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------
    initial begin
        rst          = 1;
        config_valid = 0;
        config_data  = '0;
        config_keep  = '0;
        config_last  = 0;

        repeat(4) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        // ==================================================================
        // TEST 1 — Layer 1 weights (SFC: 784 inputs, 256 neurons)
        //   bytes_per_neuron = ceil(784/8) = 98
        //   total_bytes      = 256 * 98   = 25088
        // ==================================================================
        $display("\n=== TEST 1: Layer 1 weight message header ===");
        begin : t1
            automatic logic [7:0] payload[];
            payload = new[25088];
            foreach (payload[i]) payload[i] = 8'hAB; // dummy data

            fork
                send_message(8'd0, 8'd1, 16'd784, 16'd256,
                             16'd98, 32'd25088, payload, 25088);
                begin
                    @(posedge clk iff header_valid);
                    #1; // let combinational outputs settle
                    expect32("msg_type",         {24'b0, o_msg_type},        32'd0);
                    expect32("layer_id",         {24'b0, o_layer_id},        32'd1);
                    expect32("layer_inputs",     {16'b0, o_layer_inputs},    32'd784);
                    expect32("num_neurons",      {16'b0, o_num_neurons},     32'd256);
                    expect32("bytes_per_neuron", {16'b0, o_bytes_per_neuron},32'd98);
                    expect32("total_bytes",      o_total_bytes,              32'd25088);
                end
            join

            repeat(2) @(posedge clk);
            expect32("payload_byte_count", payload_byte_count, 32'd25088);
        end

        // ==================================================================
        // TEST 2 — Layer 1 thresholds (256 neurons × 4 bytes each)
        // ==================================================================
        $display("\n=== TEST 2: Layer 1 threshold message header ===");
        begin : t2
            automatic logic [7:0] payload[];
            payload = new[1024];
            foreach (payload[i]) payload[i] = $urandom_range(0, 255);

            fork
                send_message(8'd1, 8'd1, 16'd0, 16'd256,
                             16'd4, 32'd1024, payload, 1024);
                begin
                    @(posedge clk iff header_valid);
                    #1;
                    expect32("msg_type",    {24'b0, o_msg_type},    32'd1);
                    expect32("layer_id",    {24'b0, o_layer_id},    32'd1);
                    expect32("num_neurons", {16'b0, o_num_neurons}, 32'd256);
                    expect32("total_bytes", o_total_bytes,          32'd1024);
                end
            join

            repeat(2) @(posedge clk);
            expect32("payload_byte_count", payload_byte_count, 32'd1024);
        end

        // ==================================================================
        // TEST 3 — Tiny 2-neuron/4-input topology (easy to trace by hand)
        //   bytes_per_neuron=1, total_bytes=2, payload fits in 1 beat
        // ==================================================================
        $display("\n=== TEST 3: Tiny 2-neuron weight message ===");
        begin : t3
            // Neuron 0: weights = 4'b1011 (padded to 8'b1111_1011)
            // Neuron 1: weights = 4'b0010 (padded to 8'b1111_0010)
            automatic logic [7:0] payload[] = '{8'hFB, 8'hF2};

            fork
                send_message(8'd0, 8'd1, 16'd4, 16'd2,
                             16'd1, 32'd2, payload, 2);
                begin
                    @(posedge clk iff header_valid);
                    #1;
                    expect32("num_neurons",      {16'b0, o_num_neurons},     32'd2);
                    expect32("bytes_per_neuron", {16'b0, o_bytes_per_neuron},32'd1);
                    expect32("layer_inputs",     {16'b0, o_layer_inputs},    32'd4);
                    expect32("total_bytes",      o_total_bytes,              32'd2);
                end
            join

            repeat(2) @(posedge clk);
            expect32("payload_byte_count", payload_byte_count, 32'd2);
        end

        // ==================================================================
        // TEST 4 — Back-to-back messages (no gap between them)
        //          Parser should immediately accept next header after TLAST
        // ==================================================================
        $display("\n=== TEST 4: Back-to-back messages ===");
        begin : t4
            automatic logic [7:0] p1[] = new[8];
            automatic logic [7:0] p2[] = new[16];
            foreach (p1[i]) p1[i] = 8'hAA;
            foreach (p2[i]) p2[i] = 8'hBB;

            // Send msg 1, check header
            fork
                send_message(8'd0, 8'd1, 16'd8, 16'd8,
                             16'd1, 32'd8, p1, 8);
                @(posedge clk iff header_valid);
            join
            // Immediately send msg 2 with no extra idle cycles
            fork
                send_message(8'd1, 8'd2, 16'd0, 16'd4,
                             16'd4, 32'd16, p2, 16);
                begin
                    @(posedge clk iff header_valid);
                    #1;
                    expect32("msg2 msg_type",  {24'b0, o_msg_type},  32'd1);
                    expect32("msg2 layer_id",  {24'b0, o_layer_id},  32'd2);
                    expect32("msg2 total_bytes", o_total_bytes,      32'd16);
                end
            join
            repeat(2) @(posedge clk);
            expect32("msg2 byte_count", payload_byte_count, 32'd16);
        end

        // ==================================================================
        // Summary
        // ==================================================================
        repeat(4) @(posedge clk);
        $display("\n==========================================");
        $display("  %0d passed  |  %0d failed", pass_count, fail_count);
        $display("==========================================\n");
        if (fail_count > 0) $display("*** SOME TESTS FAILED ***");
        else                $display("All tests passed — ready for Phase 2");
        $finish;
    end

    initial begin : timeout_watchdog
        #5_000_000;
        $display("TIMEOUT — simulation ran too long");
        $finish;
    end

endmodule