`timescale 1ns / 1ps

module config_manager_multi_tb;
    import bnn_config_pkg::*;

    localparam int CONFIG_BUS_WIDTH = 64;
    localparam int PW               = 8;

    localparam int L0_INPUTS        = 784;
    localparam int L0_NEURONS       = 256;
    localparam int L0_COUNTW        = $clog2(L0_INPUTS + 1);
    localparam int L0_BEATS         = (L0_INPUTS + PW - 1) / PW;
    localparam int L0_NIDX_W        = (L0_NEURONS > 1) ? $clog2(L0_NEURONS) : 1;
    localparam int L0_ADDR_W        = (L0_BEATS > 1) ? $clog2(L0_BEATS) : 1;

    localparam int L1_INPUTS        = 256;
    localparam int L1_NEURONS       = 256;
    localparam int L1_COUNTW        = $clog2(L1_INPUTS + 1);
    localparam int L1_BEATS         = (L1_INPUTS + PW - 1) / PW;
    localparam int L1_NIDX_W        = (L1_NEURONS > 1) ? $clog2(L1_NEURONS) : 1;
    localparam int L1_ADDR_W        = (L1_BEATS > 1) ? $clog2(L1_BEATS) : 1;

    localparam int L2_INPUTS        = 256;
    localparam int L2_NEURONS       = 10;
    localparam int L2_COUNTW        = $clog2(L2_INPUTS + 1);
    localparam int L2_BEATS         = (L2_INPUTS + PW - 1) / PW;
    localparam int L2_NIDX_W        = (L2_NEURONS > 1) ? $clog2(L2_NEURONS) : 1;
    localparam int L2_ADDR_W        = (L2_BEATS > 1) ? $clog2(L2_BEATS) : 1;

    logic clk, rst;

    logic                           cfg_valid;
    logic                           cfg_ready;
    logic [CONFIG_BUS_WIDTH-1:0]    cfg_data;
    logic [CONFIG_BUS_WIDTH/8-1:0]  cfg_keep;
    logic                           cfg_last;

    logic                           l0_cfg_we, l0_cfg_tw, l0_cfg_rdy;
    logic [L0_NIDX_W-1:0]           l0_cfg_nidx;
    logic [L0_ADDR_W-1:0]           l0_cfg_addr;
    logic [PW-1:0]                  l0_cfg_wdata;
    logic [L0_COUNTW-1:0]           l0_cfg_tdata;

    logic                           l1_cfg_we, l1_cfg_tw, l1_cfg_rdy;
    logic [L1_NIDX_W-1:0]           l1_cfg_nidx;
    logic [L1_ADDR_W-1:0]           l1_cfg_addr;
    logic [PW-1:0]                  l1_cfg_wdata;
    logic [L1_COUNTW-1:0]           l1_cfg_tdata;

    logic                           l2_cfg_we, l2_cfg_tw, l2_cfg_rdy;
    logic [L2_NIDX_W-1:0]           l2_cfg_nidx;
    logic [L2_ADDR_W-1:0]           l2_cfg_addr;
    logic [PW-1:0]                  l2_cfg_wdata;
    logic [L2_COUNTW-1:0]           l2_cfg_tdata;

    bit          weights0[L0_NEURONS][L0_INPUTS];
    bit          weights1[L1_NEURONS][L1_INPUTS];
    bit          weights2[L2_NEURONS][L2_INPUTS];
    int unsigned thresholds0[L0_NEURONS];
    int unsigned thresholds1[L1_NEURONS];
    int unsigned thresholds2[L2_NEURONS];

    byte_q_t       stream_bytes;
    axi64_beat_t   axi_beats_q[$];

    cfg_header_t   meta0, meta1, meta2;
    bit            meta0_valid, meta1_valid, meta2_valid;

    weight_write_t exp_w0_q[$], exp_w1_q[$], exp_w2_q[$];
    thresh_write_t exp_t0_q[$], exp_t1_q[$], exp_t2_q[$];

    logic [CONFIG_BUS_WIDTH-1:0]   beat_data_da[];
    logic [CONFIG_BUS_WIDTH/8-1:0] beat_keep_da[];
    logic                          beat_last_da[];
    integer                        beat_count;

    logic [L0_NIDX_W-1:0]          exp_w_nidx0[];
    logic [L0_ADDR_W-1:0]          exp_w_addr0[];
    logic [PW-1:0]                 exp_w_word0[];
    logic [L0_NIDX_W-1:0]          exp_t_nidx0[];
    logic [L0_COUNTW-1:0]          exp_t_val0[];
    integer                        exp_w_count0, exp_t_count0;

    logic [L1_NIDX_W-1:0]          exp_w_nidx1[];
    logic [L1_ADDR_W-1:0]          exp_w_addr1[];
    logic [PW-1:0]                 exp_w_word1[];
    logic [L1_NIDX_W-1:0]          exp_t_nidx1[];
    logic [L1_COUNTW-1:0]          exp_t_val1[];
    integer                        exp_w_count1, exp_t_count1;

    logic [L2_NIDX_W-1:0]          exp_w_nidx2[];
    logic [L2_ADDR_W-1:0]          exp_w_addr2[];
    logic [PW-1:0]                 exp_w_word2[];
    logic [L2_NIDX_W-1:0]          exp_t_nidx2[];
    logic [L2_COUNTW-1:0]          exp_t_val2[];
    integer                        exp_w_count2, exp_t_count2;

    integer                        w_seen0, w_seen1, w_seen2;
    integer                        t_seen0, t_seen1, t_seen2;

    bit gaps_enable;
    bit stalls_enable;

    config_manager_multi #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .PW(PW),
        .L0_INPUTS(L0_INPUTS), .L0_NEURONS(L0_NEURONS),
        .L1_INPUTS(L1_INPUTS), .L1_NEURONS(L1_NEURONS),
        .L2_INPUTS(L2_INPUTS), .L2_NEURONS(L2_NEURONS)
    ) dut (
        .clk(clk),
        .rst(rst),
        .cfg_valid(cfg_valid),
        .cfg_ready(cfg_ready),
        .cfg_data(cfg_data),
        .cfg_keep(cfg_keep),
        .cfg_last(cfg_last),

        .l0_cfg_we(l0_cfg_we),
        .l0_cfg_tw(l0_cfg_tw),
        .l0_cfg_nidx(l0_cfg_nidx),
        .l0_cfg_addr(l0_cfg_addr),
        .l0_cfg_wdata(l0_cfg_wdata),
        .l0_cfg_tdata(l0_cfg_tdata),
        .l0_cfg_rdy(l0_cfg_rdy),

        .l1_cfg_we(l1_cfg_we),
        .l1_cfg_tw(l1_cfg_tw),
        .l1_cfg_nidx(l1_cfg_nidx),
        .l1_cfg_addr(l1_cfg_addr),
        .l1_cfg_wdata(l1_cfg_wdata),
        .l1_cfg_tdata(l1_cfg_tdata),
        .l1_cfg_rdy(l1_cfg_rdy),

        .l2_cfg_we(l2_cfg_we),
        .l2_cfg_tw(l2_cfg_tw),
        .l2_cfg_nidx(l2_cfg_nidx),
        .l2_cfg_addr(l2_cfg_addr),
        .l2_cfg_wdata(l2_cfg_wdata),
        .l2_cfg_tdata(l2_cfg_tdata),
        .l2_cfg_rdy(l2_cfg_rdy)
    );

    always #5 clk = ~clk;

    task automatic do_reset();
        begin
            rst        = 1'b1;
            cfg_valid  = 1'b0;
            cfg_data   = '0;
            cfg_keep   = '0;
            cfg_last   = 1'b0;
            l0_cfg_rdy = 1'b1;
            l1_cfg_rdy = 1'b1;
            l2_cfg_rdy = 1'b1;
            repeat (5) @(posedge clk);
            rst = 1'b0;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic fill_random_model(input int unsigned seed);
        integer n, i;
        begin
            void'($urandom(seed));

            for (n = 0; n < L0_NEURONS; n = n + 1) begin
                for (i = 0; i < L0_INPUTS; i = i + 1)
                    weights0[n][i] = $urandom_range(0,1);
                thresholds0[n] = $urandom_range(0, L0_INPUTS);
            end

            for (n = 0; n < L1_NEURONS; n = n + 1) begin
                for (i = 0; i < L1_INPUTS; i = i + 1)
                    weights1[n][i] = $urandom_range(0,1);
                thresholds1[n] = $urandom_range(0, L1_INPUTS);
            end

            for (n = 0; n < L2_NEURONS; n = n + 1) begin
                for (i = 0; i < L2_INPUTS; i = i + 1)
                    weights2[n][i] = $urandom_range(0,1);
                thresholds2[n] = $urandom_range(0, L2_INPUTS);
            end
        end
    endtask

    task automatic flatten_beats();
        integer i;
        begin
            beat_count   = axi_beats_q.size();
            beat_data_da = new[beat_count];
            beat_keep_da = new[beat_count];
            beat_last_da = new[beat_count];

            for (i = 0; i < beat_count; i = i + 1) begin
                beat_data_da[i] = axi_beats_q[i].data;
                beat_keep_da[i] = axi_beats_q[i].keep;
                beat_last_da[i] = axi_beats_q[i].last;
            end

            axi_beats_q.delete();
        end
    endtask

    task automatic flatten_expected();
        integer i;
        begin
            exp_w_count0 = exp_w0_q.size();
            exp_t_count0 = exp_t0_q.size();
            exp_w_count1 = exp_w1_q.size();
            exp_t_count1 = exp_t1_q.size();
            exp_w_count2 = exp_w2_q.size();
            exp_t_count2 = exp_t2_q.size();

            exp_w_nidx0 = new[exp_w_count0];
            exp_w_addr0 = new[exp_w_count0];
            exp_w_word0 = new[exp_w_count0];
            exp_t_nidx0 = new[exp_t_count0];
            exp_t_val0  = new[exp_t_count0];

            exp_w_nidx1 = new[exp_w_count1];
            exp_w_addr1 = new[exp_w_count1];
            exp_w_word1 = new[exp_w_count1];
            exp_t_nidx1 = new[exp_t_count1];
            exp_t_val1  = new[exp_t_count1];

            exp_w_nidx2 = new[exp_w_count2];
            exp_w_addr2 = new[exp_w_count2];
            exp_w_word2 = new[exp_w_count2];
            exp_t_nidx2 = new[exp_t_count2];
            exp_t_val2  = new[exp_t_count2];

            for (i = 0; i < exp_w_count0; i = i + 1) begin
                exp_w_nidx0[i] = exp_w0_q[i].neuron[L0_NIDX_W-1:0];
                exp_w_addr0[i] = exp_w0_q[i].beat[L0_ADDR_W-1:0];
                exp_w_word0[i] = exp_w0_q[i].word[PW-1:0];
            end
            for (i = 0; i < exp_t_count0; i = i + 1) begin
                exp_t_nidx0[i] = exp_t0_q[i].neuron[L0_NIDX_W-1:0];
                exp_t_val0[i]  = exp_t0_q[i].thresh[L0_COUNTW-1:0];
            end

            for (i = 0; i < exp_w_count1; i = i + 1) begin
                exp_w_nidx1[i] = exp_w1_q[i].neuron[L1_NIDX_W-1:0];
                exp_w_addr1[i] = exp_w1_q[i].beat[L1_ADDR_W-1:0];
                exp_w_word1[i] = exp_w1_q[i].word[PW-1:0];
            end
            for (i = 0; i < exp_t_count1; i = i + 1) begin
                exp_t_nidx1[i] = exp_t1_q[i].neuron[L1_NIDX_W-1:0];
                exp_t_val1[i]  = exp_t1_q[i].thresh[L1_COUNTW-1:0];
            end

            for (i = 0; i < exp_w_count2; i = i + 1) begin
                exp_w_nidx2[i] = exp_w2_q[i].neuron[L2_NIDX_W-1:0];
                exp_w_addr2[i] = exp_w2_q[i].beat[L2_ADDR_W-1:0];
                exp_w_word2[i] = exp_w2_q[i].word[PW-1:0];
            end
            for (i = 0; i < exp_t_count2; i = i + 1) begin
                exp_t_nidx2[i] = exp_t2_q[i].neuron[L2_NIDX_W-1:0];
                exp_t_val2[i]  = exp_t2_q[i].thresh[L2_COUNTW-1:0];
            end

            exp_w0_q.delete(); exp_t0_q.delete();
            exp_w1_q.delete(); exp_t1_q.delete();
            exp_w2_q.delete(); exp_t2_q.delete();
        end
    endtask

    task automatic build_case();
        begin
            stream_bytes.delete();
            axi_beats_q.delete();
            exp_w0_q.delete(); exp_t0_q.delete();
            exp_w1_q.delete(); exp_t1_q.delete();
            exp_w2_q.delete(); exp_t2_q.delete();

            build_layer_stream(stream_bytes, 0, L0_INPUTS, weights0, thresholds0);
            build_layer_stream(stream_bytes, 1, L1_INPUTS, weights1, thresholds1);
            build_layer_stream(stream_bytes, 2, L2_INPUTS, weights2, thresholds2);

            pack_bytes_to_axi64(stream_bytes, axi_beats_q);

            process_stream_for_layer(stream_bytes, 0, PW, meta0, meta0_valid, exp_w0_q, exp_t0_q);
            process_stream_for_layer(stream_bytes, 1, PW, meta1, meta1_valid, exp_w1_q, exp_t1_q);
            process_stream_for_layer(stream_bytes, 2, PW, meta2, meta2_valid, exp_w2_q, exp_t2_q);

            if (!meta0_valid || !meta1_valid || !meta2_valid)
                $fatal(1, "Missing weight header for one or more layers");

            flatten_beats();
            flatten_expected();
        end
    endtask

    task automatic drive_stream();
        integer i;
        begin
            for (i = 0; i < beat_count; i = i + 1) begin
                do begin
                    @(posedge clk);
                    cfg_valid <= 1'b1;
                    cfg_data  <= beat_data_da[i];
                    cfg_keep  <= beat_keep_da[i];
                    cfg_last  <= beat_last_da[i];
                end while (!cfg_ready);

                @(posedge clk);
                cfg_valid <= 1'b0;
                cfg_data  <= '0;
                cfg_keep  <= '0;
                cfg_last  <= 1'b0;

                if (gaps_enable)
                    repeat ($urandom_range(0, 3)) @(posedge clk);
            end
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            w_seen0 <= 0; t_seen0 <= 0;
            w_seen1 <= 0; t_seen1 <= 0;
            w_seen2 <= 0; t_seen2 <= 0;
        end else begin
            if (l0_cfg_we && l0_cfg_rdy) begin
                if (w_seen0 >= exp_w_count0) $fatal(1, "Extra L0 weight write");
                if (l0_cfg_nidx !== exp_w_nidx0[w_seen0]) $fatal(1, "L0 weight neuron mismatch");
                if (l0_cfg_addr !== exp_w_addr0[w_seen0]) $fatal(1, "L0 weight addr mismatch");
                if (l0_cfg_wdata !== exp_w_word0[w_seen0]) $fatal(1, "L0 weight data mismatch");
                w_seen0 <= w_seen0 + 1;
            end

            if (l0_cfg_tw && l0_cfg_rdy) begin
                if (t_seen0 >= exp_t_count0) $fatal(1, "Extra L0 threshold write");
                if (l0_cfg_nidx !== exp_t_nidx0[t_seen0]) $fatal(1, "L0 threshold neuron mismatch");
                if (l0_cfg_tdata !== exp_t_val0[t_seen0]) $fatal(1, "L0 threshold data mismatch");
                t_seen0 <= t_seen0 + 1;
            end

            if (l1_cfg_we && l1_cfg_rdy) begin
                if (w_seen1 >= exp_w_count1) $fatal(1, "Extra L1 weight write");
                if (l1_cfg_nidx !== exp_w_nidx1[w_seen1]) $fatal(1, "L1 weight neuron mismatch");
                if (l1_cfg_addr !== exp_w_addr1[w_seen1]) $fatal(1, "L1 weight addr mismatch");
                if (l1_cfg_wdata !== exp_w_word1[w_seen1]) $fatal(1, "L1 weight data mismatch");
                w_seen1 <= w_seen1 + 1;
            end

            if (l1_cfg_tw && l1_cfg_rdy) begin
                if (t_seen1 >= exp_t_count1) $fatal(1, "Extra L1 threshold write");
                if (l1_cfg_nidx !== exp_t_nidx1[t_seen1]) $fatal(1, "L1 threshold neuron mismatch");
                if (l1_cfg_tdata !== exp_t_val1[t_seen1]) $fatal(1, "L1 threshold data mismatch");
                t_seen1 <= t_seen1 + 1;
            end

            if (l2_cfg_we && l2_cfg_rdy) begin
                if (w_seen2 >= exp_w_count2) $fatal(1, "Extra L2 weight write");
                if (l2_cfg_nidx !== exp_w_nidx2[w_seen2]) $fatal(1, "L2 weight neuron mismatch");
                if (l2_cfg_addr !== exp_w_addr2[w_seen2]) $fatal(1, "L2 weight addr mismatch");
                if (l2_cfg_wdata !== exp_w_word2[w_seen2]) $fatal(1, "L2 weight data mismatch");
                w_seen2 <= w_seen2 + 1;
            end

            if (l2_cfg_tw && l2_cfg_rdy) begin
                if (t_seen2 >= exp_t_count2) $fatal(1, "Extra L2 threshold write");
                if (l2_cfg_nidx !== exp_t_nidx2[t_seen2]) $fatal(1, "L2 threshold neuron mismatch");
                if (l2_cfg_tdata !== exp_t_val2[t_seen2]) $fatal(1, "L2 threshold data mismatch");
                t_seen2 <= t_seen2 + 1;
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            l0_cfg_rdy <= 1'b1;
            l1_cfg_rdy <= 1'b1;
            l2_cfg_rdy <= 1'b1;
        end else if (stalls_enable) begin
            l0_cfg_rdy <= ($urandom_range(0,3) != 0);
            l1_cfg_rdy <= ($urandom_range(0,3) != 0);
            l2_cfg_rdy <= ($urandom_range(0,3) != 0);
        end else begin
            l0_cfg_rdy <= 1'b1;
            l1_cfg_rdy <= 1'b1;
            l2_cfg_rdy <= 1'b1;
        end
    end

    task automatic run_test(
        input string name,
        input int unsigned seed,
        input bit use_gaps,
        input bit use_stalls
    );
        begin
            $display("---- %s ----", name);

            gaps_enable   = use_gaps;
            stalls_enable = use_stalls;

            w_seen0 = 0; t_seen0 = 0;
            w_seen1 = 0; t_seen1 = 0;
            w_seen2 = 0; t_seen2 = 0;

            do_reset();
            fill_random_model(seed);
            build_case();

            fork
                drive_stream();
            join

            wait ((w_seen0 == exp_w_count0) && (t_seen0 == exp_t_count0) &&
                  (w_seen1 == exp_w_count1) && (t_seen1 == exp_t_count1) &&
                  (w_seen2 == exp_w_count2) && (t_seen2 == exp_t_count2));

            repeat (10) @(posedge clk);
            $display("%s PASS", name);
        end
    endtask

    initial begin
        clk = 1'b0;
        run_test("basic",          32'd42,   1'b0, 1'b0);
        run_test("with_gaps",      32'd99,   1'b1, 1'b0);
        run_test("with_backpress", 32'd2026, 1'b0, 1'b1);
        run_test("gaps_backpress", 32'd7,    1'b1, 1'b1);
        $display("ALL TESTS PASSED");
        $finish;
    end

    initial begin
        #20ms;
        $fatal(1,
            "TB timeout: L0 w=%0d/%0d t=%0d/%0d | L1 w=%0d/%0d t=%0d/%0d | L2 w=%0d/%0d t=%0d/%0d",
            w_seen0, exp_w_count0, t_seen0, exp_t_count0,
            w_seen1, exp_w_count1, t_seen1, exp_t_count1,
            w_seen2, exp_w_count2, t_seen2, exp_t_count2
        );
    end

endmodule