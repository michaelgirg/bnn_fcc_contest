`default_nettype none

module config_manager_multi_tb;

    parameter int CONFIG_BUS_WIDTH = 64;
    parameter int PW               = 8;

    parameter int L0_INPUTS        = 16;
    parameter int L0_NEURONS       = 2;
    parameter int L0_COUNT_WIDTH   = $clog2(L0_INPUTS + 1);
    parameter int L0_INPUT_BEATS   = (L0_INPUTS + PW - 1) / PW;
    parameter int L0_NIDX_W        = (L0_NEURONS > 1) ? $clog2(L0_NEURONS) : 1;
    parameter int L0_ADDR_W        = (L0_INPUT_BEATS > 1) ? $clog2(L0_INPUT_BEATS) : 1;

    parameter int L1_INPUTS        = 8;
    parameter int L1_NEURONS       = 3;
    parameter int L1_COUNT_WIDTH   = $clog2(L1_INPUTS + 1);
    parameter int L1_INPUT_BEATS   = (L1_INPUTS + PW - 1) / PW;
    parameter int L1_NIDX_W        = (L1_NEURONS > 1) ? $clog2(L1_NEURONS) : 1;
    parameter int L1_ADDR_W        = (L1_INPUT_BEATS > 1) ? $clog2(L1_INPUT_BEATS) : 1;

    parameter int L2_INPUTS        = 8;
    parameter int L2_NEURONS       = 2;
    parameter int L2_COUNT_WIDTH   = $clog2(L2_INPUTS + 1);
    parameter int L2_INPUT_BEATS   = (L2_INPUTS + PW - 1) / PW;
    parameter int L2_NIDX_W        = (L2_NEURONS > 1) ? $clog2(L2_NEURONS) : 1;
    parameter int L2_ADDR_W        = (L2_INPUT_BEATS > 1) ? $clog2(L2_INPUT_BEATS) : 1;

    localparam int MAX_NEURONS =
        (L0_NEURONS >= L1_NEURONS) ?
            ((L0_NEURONS >= L2_NEURONS) ? L0_NEURONS : L2_NEURONS) :
            ((L1_NEURONS >= L2_NEURONS) ? L1_NEURONS : L2_NEURONS);

    localparam int MAX_BEATS =
        (L0_INPUT_BEATS >= L1_INPUT_BEATS) ?
            ((L0_INPUT_BEATS >= L2_INPUT_BEATS) ? L0_INPUT_BEATS : L2_INPUT_BEATS) :
            ((L1_INPUT_BEATS >= L2_INPUT_BEATS) ? L1_INPUT_BEATS : L2_INPUT_BEATS);

    localparam int CFG_NEURON_W      = (MAX_NEURONS > 1) ? $clog2(MAX_NEURONS) : 1;
    localparam int CFG_WEIGHT_ADDR_W = (MAX_BEATS > 1) ? $clog2(MAX_BEATS) : 1;
    localparam int THRESHOLD_W =
        (L0_COUNT_WIDTH > L1_COUNT_WIDTH) ?
            ((L0_COUNT_WIDTH > L2_COUNT_WIDTH) ? L0_COUNT_WIDTH : L2_COUNT_WIDTH) :
            ((L1_COUNT_WIDTH > L2_COUNT_WIDTH) ? L1_COUNT_WIDTH : L2_COUNT_WIDTH);

    logic                         clk = 0;
    logic                         rst;

    logic                         cfg_valid;
    logic                         cfg_ready;
    logic [CONFIG_BUS_WIDTH-1:0]  cfg_data;
    logic [CONFIG_BUS_WIDTH/8-1:0] cfg_keep;
    logic                         cfg_last;

    logic                         out_cfg_write_en;
    logic [1:0]                   out_cfg_layer_sel;
    logic [CFG_NEURON_W-1:0]      out_cfg_neuron_idx;
    logic [CFG_WEIGHT_ADDR_W-1:0] out_cfg_weight_addr;
    logic [PW-1:0]                out_cfg_weight_data;
    logic [THRESHOLD_W-1:0]       out_cfg_threshold_data;
    logic                         out_cfg_threshold_write;
    logic                         out_cfg_ready;

    int passed, failed;

    typedef struct packed {
        logic                         is_thresh;
        logic [1:0]                   layer_sel;
        logic [CFG_NEURON_W-1:0]      neuron_idx;
        logic [CFG_WEIGHT_ADDR_W-1:0] weight_addr;
        logic [PW-1:0]                weight_data;
        logic [THRESHOLD_W-1:0]       threshold_data;
    } cfg_txn_t;

    cfg_txn_t seen_q[$];

    config_manager_multi #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .PW              (PW),

        .L0_INPUTS       (L0_INPUTS),
        .L0_NEURONS      (L0_NEURONS),
        .L0_COUNT_WIDTH  (L0_COUNT_WIDTH),
        .L0_INPUT_BEATS  (L0_INPUT_BEATS),
        .L0_NIDX_W       (L0_NIDX_W),
        .L0_ADDR_W       (L0_ADDR_W),

        .L1_INPUTS       (L1_INPUTS),
        .L1_NEURONS      (L1_NEURONS),
        .L1_COUNT_WIDTH  (L1_COUNT_WIDTH),
        .L1_INPUT_BEATS  (L1_INPUT_BEATS),
        .L1_NIDX_W       (L1_NIDX_W),
        .L1_ADDR_W       (L1_ADDR_W),

        .L2_INPUTS       (L2_INPUTS),
        .L2_NEURONS      (L2_NEURONS),
        .L2_COUNT_WIDTH  (L2_COUNT_WIDTH),
        .L2_INPUT_BEATS  (L2_INPUT_BEATS),
        .L2_NIDX_W       (L2_NIDX_W),
        .L2_ADDR_W       (L2_ADDR_W)
    ) DUT (
        .clk                 (clk),
        .rst                 (rst),
        .cfg_valid           (cfg_valid),
        .cfg_ready           (cfg_ready),
        .cfg_data            (cfg_data),
        .cfg_keep            (cfg_keep),
        .cfg_last            (cfg_last),

        .out_cfg_write_en       (out_cfg_write_en),
        .out_cfg_layer_sel      (out_cfg_layer_sel),
        .out_cfg_neuron_idx     (out_cfg_neuron_idx),
        .out_cfg_weight_addr    (out_cfg_weight_addr),
        .out_cfg_weight_data    (out_cfg_weight_data),
        .out_cfg_threshold_data (out_cfg_threshold_data),
        .out_cfg_threshold_write(out_cfg_threshold_write),
        .out_cfg_ready          (out_cfg_ready)
    );

    initial forever #5 clk = ~clk;

    function automatic logic [15:0] layer_inputs(input logic [1:0] lid);
        case (lid)
            2'd0: layer_inputs = L0_INPUTS[15:0];
            2'd1: layer_inputs = L1_INPUTS[15:0];
            2'd2: layer_inputs = L2_INPUTS[15:0];
            default: layer_inputs = 16'd0;
        endcase
    endfunction

    function automatic logic [15:0] layer_neurons(input logic [1:0] lid);
        case (lid)
            2'd0: layer_neurons = L0_NEURONS[15:0];
            2'd1: layer_neurons = L1_NEURONS[15:0];
            2'd2: layer_neurons = L2_NEURONS[15:0];
            default: layer_neurons = 16'd0;
        endcase
    endfunction

    function automatic logic [15:0] layer_beats(input logic [1:0] lid);
        case (lid)
            2'd0: layer_beats = L0_INPUT_BEATS[15:0];
            2'd1: layer_beats = L1_INPUT_BEATS[15:0];
            2'd2: layer_beats = L2_INPUT_BEATS[15:0];
            default: layer_beats = 16'd0;
        endcase
    endfunction

    function automatic logic [5:0] layer_countw(input logic [1:0] lid);
        case (lid)
            2'd0: layer_countw = L0_COUNT_WIDTH[5:0];
            2'd1: layer_countw = L1_COUNT_WIDTH[5:0];
            2'd2: layer_countw = L2_COUNT_WIDTH[5:0];
            default: layer_countw = 6'd1;
        endcase
    endfunction

    function automatic logic [THRESHOLD_W-1:0] trunc_threshold(
        input logic [31:0] raw,
        input logic [1:0]  lid
    );
        logic [31:0] mask32;
        logic [5:0]  cw;
        begin
            cw = layer_countw(lid);
            if (cw >= 32)
                mask32 = 32'hFFFF_FFFF;
            else
                mask32 = (32'h0000_0001 << cw) - 1;
            trunc_threshold = (raw & mask32)[THRESHOLD_W-1:0];
        end
    endfunction

    function automatic logic [63:0] pack8(
        input logic [7:0] b0,
        input logic [7:0] b1,
        input logic [7:0] b2,
        input logic [7:0] b3,
        input logic [7:0] b4,
        input logic [7:0] b5,
        input logic [7:0] b6,
        input logic [7:0] b7
    );
        pack8 = {b7,b6,b5,b4,b3,b2,b1,b0};
    endfunction

    task automatic send_beat(
        input logic [63:0] data_i,
        input logic [7:0]  keep_i,
        input logic        last_i
    );
        @(posedge clk iff cfg_ready);
        cfg_valid <= 1'b1;
        cfg_data  <= data_i;
        cfg_keep  <= keep_i;
        cfg_last  <= last_i;
        @(posedge clk);
        cfg_valid <= 1'b0;
        cfg_data  <= '0;
        cfg_keep  <= '0;
        cfg_last  <= 1'b0;
    endtask

    task automatic send_message(
        input logic [7:0] msgtype,
        input logic [7:0] layerid,
        input logic [15:0] layerinputs_i,
        input logic [15:0] numneurons_i,
        input logic [15:0] bytesperneuron_i,
        input logic [7:0] payload[]
    );
        logic [31:0] totalbytes;
        logic [63:0] beat_data;
        logic [7:0]  beat_keep;
        int idx;
        int j;

        totalbytes = payload.size();

        send_beat(
            pack8(
                msgtype,
                layerid,
                layerinputs_i[7:0],
                layerinputs_i[15:8],
                numneurons_i[7:0],
                numneurons_i[15:8],
                bytesperneuron_i[7:0],
                bytesperneuron_i[15:8]
            ),
            8'hFF,
            1'b0
        );

        send_beat(
            pack8(
                totalbytes[7:0],
                totalbytes[15:8],
                totalbytes[23:16],
                totalbytes[31:24],
                8'h00,
                8'h00,
                8'h00,
                8'h00
            ),
            8'hFF,
            (payload.size() == 0)
        );

        idx = 0;
        while (idx < payload.size()) begin
            beat_data = '0;
            beat_keep = '0;
            for (j = 0; j < 8; j++) begin
                if (idx < payload.size()) begin
                    beat_data[j*8 +: 8] = payload[idx];
                    beat_keep[j] = 1'b1;
                    idx++;
                end
            end
            send_beat(beat_data, beat_keep, (idx == payload.size()));
        end
    endtask

    task automatic check_segment(
        input string name,
        input int base_idx,
        input cfg_txn_t exp[]
    );
        bit local_fail;
        local_fail = 1'b0;

        if ((seen_q.size() - base_idx) != exp.size()) begin
            $display("[FAIL] %s: saw %0d txns, expected %0d",
                     name, seen_q.size() - base_idx, exp.size());
            failed++;
            return;
        end

        for (int i = 0; i < exp.size(); i++) begin
            if (seen_q[base_idx + i] !== exp[i]) begin
                $display("[FAIL] %s: mismatch at txn %0d", name, i);
                $display("        got is_thresh=%0b layer=%0d neuron=%0d addr=%0d w=%0h t=%0h",
                         seen_q[base_idx + i].is_thresh,
                         seen_q[base_idx + i].layer_sel,
                         seen_q[base_idx + i].neuron_idx,
                         seen_q[base_idx + i].weight_addr,
                         seen_q[base_idx + i].weight_data,
                         seen_q[base_idx + i].threshold_data);
                $display("        exp is_thresh=%0b layer=%0d neuron=%0d addr=%0d w=%0h t=%0h",
                         exp[i].is_thresh,
                         exp[i].layer_sel,
                         exp[i].neuron_idx,
                         exp[i].weight_addr,
                         exp[i].weight_data,
                         exp[i].threshold_data);
                local_fail = 1'b1;
            end
        end

        if (local_fail) failed++;
        else begin
            $display("[PASS] %s", name);
            passed++;
        end
    endtask

    always_ff @(posedge clk) begin
        cfg_txn_t txn;
        if (!rst && out_cfg_ready && out_cfg_write_en) begin
            txn.is_thresh      = 1'b0;
            txn.layer_sel      = out_cfg_layer_sel;
            txn.neuron_idx     = out_cfg_neuron_idx;
            txn.weight_addr    = out_cfg_weight_addr;
            txn.weight_data    = out_cfg_weight_data;
            txn.threshold_data = '0;
            seen_q.push_back(txn);
        end

        if (!rst && out_cfg_ready && out_cfg_threshold_write) begin
            txn.is_thresh      = 1'b1;
            txn.layer_sel      = out_cfg_layer_sel;
            txn.neuron_idx     = out_cfg_neuron_idx;
            txn.weight_addr    = '0;
            txn.weight_data    = '0;
            txn.threshold_data = out_cfg_threshold_data;
            seen_q.push_back(txn);
        end
    end

    initial begin
        passed        = 0;
        failed        = 0;
        rst           = 1'b1;
        cfg_valid     = 1'b0;
        cfg_data      = '0;
        cfg_keep      = '0;
        cfg_last      = 1'b0;
        out_cfg_ready = 1'b1;

        repeat (5) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // ------------------------------------------------------------
        // TEST 1: Layer 0 weights routing/order
        // ------------------------------------------------------------
        begin
            logic [7:0] payload[];
            cfg_txn_t   exp[];
            int         base;
            int         idx;

            base    = seen_q.size();
            payload = new[L0_INPUT_BEATS * L0_NEURONS];
            payload[0] = 8'h11;
            payload[1] = 8'h22;
            payload[2] = 8'h33;
            payload[3] = 8'h44;

            exp = new[payload.size()];
            idx = 0;
            for (int n = 0; n < L0_NEURONS; n++) begin
                for (int b = 0; b < L0_INPUT_BEATS; b++) begin
                    exp[idx].is_thresh      = 1'b0;
                    exp[idx].layer_sel      = 2'd0;
                    exp[idx].neuron_idx     = CFG_NEURON_W'(n);
                    exp[idx].weight_addr    = CFG_WEIGHT_ADDR_W'(b);
                    exp[idx].weight_data    = payload[idx];
                    exp[idx].threshold_data = '0;
                    idx++;
                end
            end

            send_message(
                8'd0,
                8'd0,
                layer_inputs(2'd0),
                layer_neurons(2'd0),
                layer_beats(2'd0),
                payload
            );

            repeat (10) @(posedge clk);
            check_segment("Test1_layer0_weights", base, exp);
        end

        // ------------------------------------------------------------
        // TEST 2: Layer 1 thresholds routing + truncation
        // ------------------------------------------------------------
        begin
            logic [7:0] payload[];
            cfg_txn_t   exp[];
            logic [31:0] raw_thresholds[0:L1_NEURONS-1];
            int         base;
            int         idx;

            base = seen_q.size();
            payload = new[4 * L1_NEURONS];

            raw_thresholds[0] = 32'd3;
            raw_thresholds[1] = 32'd9;
            raw_thresholds[2] = 32'd255;

            for (int n = 0; n < L1_NEURONS; n++) begin
                payload[n*4 + 0] = raw_thresholds[n][7:0];
                payload[n*4 + 1] = raw_thresholds[n][15:8];
                payload[n*4 + 2] = raw_thresholds[n][23:16];
                payload[n*4 + 3] = raw_thresholds[n][31:24];
            end

            exp = new[L1_NEURONS];
            idx = 0;
            for (int n = 0; n < L1_NEURONS; n++) begin
                exp[idx].is_thresh      = 1'b1;
                exp[idx].layer_sel      = 2'd1;
                exp[idx].neuron_idx     = CFG_NEURON_W'(n);
                exp[idx].weight_addr    = '0;
                exp[idx].weight_data    = '0;
                exp[idx].threshold_data = trunc_threshold(raw_thresholds[n], 2'd1);
                idx++;
            end

            send_message(
                8'd1,
                8'd1,
                16'd0,
                layer_neurons(2'd1),
                16'd4,
                payload
            );

            repeat (10) @(posedge clk);
            check_segment("Test2_layer1_thresholds", base, exp);
        end

        // ------------------------------------------------------------
        // TEST 3: Unknown layer id skipped
        // ------------------------------------------------------------
        begin
            logic [7:0] payload[];
            int base;

            base    = seen_q.size();
            payload = new[3];
            payload[0] = 8'hAA;
            payload[1] = 8'hBB;
            payload[2] = 8'hCC;

            send_message(
                8'd0,
                8'd9,
                16'd8,
                16'd1,
                16'd1,
                payload
            );

            repeat (10) @(posedge clk);
            if (seen_q.size() == base) begin
                $display("[PASS] Test3_unknown_layer_skipped");
                passed++;
            end else begin
                $display("[FAIL] Test3_unknown_layer_skipped: saw %0d unexpected txns",
                         seen_q.size() - base);
                failed++;
            end
        end

        // ------------------------------------------------------------
        // TEST 4: Layer 2 weights with downstream backpressure
        // ------------------------------------------------------------
        begin
            logic [7:0] payload[];
            cfg_txn_t   exp[];
            int         base;
            int         idx;

            base    = seen_q.size();
            payload = new[L2_INPUT_BEATS * L2_NEURONS];
            payload[0] = 8'h5A;
            payload[1] = 8'hC3;

            exp = new[payload.size()];
            idx = 0;
            for (int n = 0; n < L2_NEURONS; n++) begin
                for (int b = 0; b < L2_INPUT_BEATS; b++) begin
                    exp[idx].is_thresh      = 1'b0;
                    exp[idx].layer_sel      = 2'd2;
                    exp[idx].neuron_idx     = CFG_NEURON_W'(n);
                    exp[idx].weight_addr    = CFG_WEIGHT_ADDR_W'(b);
                    exp[idx].weight_data    = payload[idx];
                    exp[idx].threshold_data = '0;
                    idx++;
                end
            end

            out_cfg_ready <= 1'b0;
            fork
                begin
                    send_message(
                        8'd0,
                        8'd2,
                        layer_inputs(2'd2),
                        layer_neurons(2'd2),
                        layer_beats(2'd2),
                        payload
                    );
                end
                begin
                    repeat (8) @(posedge clk);
                    out_cfg_ready <= 1'b1;
                end
            join

            repeat (12) @(posedge clk);
            check_segment("Test4_layer2_backpressure", base, exp);
        end

        // ------------------------------------------------------------
        // TEST 5: cfg_ready only when input beat buffer empty
        // ------------------------------------------------------------
        begin
            if (cfg_ready) begin
                $display("[PASS] Test5_cfg_ready_idle_high");
                passed++;
            end else begin
                $display("[FAIL] Test5_cfg_ready_idle_high");
                failed++;
            end
        end

        @(posedge clk);
        $display("\n========================================");
        $display("config_manager_multi TB Complete");
        $display("Tests passed: %0d", passed);
        $display("Tests failed: %0d", failed);
        $display("========================================");
        $finish;
    end

    p_no_dual_write :
    assert property (@(posedge clk) disable iff (rst) !(out_cfg_write_en && out_cfg_threshold_write))
    else $error("weight and threshold write asserted together");

    p_layer_sel_range :
    assert property (@(posedge clk) disable iff (rst)
        (out_cfg_write_en || out_cfg_threshold_write) |-> (out_cfg_layer_sel inside {2'd0,2'd1,2'd2}))
    else $error("invalid out_cfg_layer_sel");

    p_threshold_write_addr_zero :
    assert property (@(posedge clk) disable iff (rst)
        out_cfg_threshold_write |-> (out_cfg_weight_addr == '0))
    else $error("threshold write carried non-zero address");

endmodule

`default_nettype wire