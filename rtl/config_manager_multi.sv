`timescale 1ns / 1ps

module config_manager_multi #(
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int PW               = 8,

    parameter int L0_INPUTS        = 784,
    parameter int L0_NEURONS       = 256,
    parameter int L0_COUNT_WIDTH   = $clog2(L0_INPUTS + 1),
    parameter int L0_INPUT_BEATS   = (L0_INPUTS + PW - 1) / PW,
    parameter int L0_NIDX_W        = (L0_NEURONS > 1) ? $clog2(L0_NEURONS) : 1,
    parameter int L0_ADDR_W        = (L0_INPUT_BEATS > 1) ? $clog2(L0_INPUT_BEATS) : 1,

    parameter int L1_INPUTS        = 256,
    parameter int L1_NEURONS       = 256,
    parameter int L1_COUNT_WIDTH   = $clog2(L1_INPUTS + 1),
    parameter int L1_INPUT_BEATS   = (L1_INPUTS + PW - 1) / PW,
    parameter int L1_NIDX_W        = (L1_NEURONS > 1) ? $clog2(L1_NEURONS) : 1,
    parameter int L1_ADDR_W        = (L1_INPUT_BEATS > 1) ? $clog2(L1_INPUT_BEATS) : 1,

    parameter int L2_INPUTS        = 256,
    parameter int L2_NEURONS       = 10,
    parameter int L2_COUNT_WIDTH   = $clog2(L2_INPUTS + 1),
    parameter int L2_INPUT_BEATS   = (L2_INPUTS + PW - 1) / PW,
    parameter int L2_NIDX_W        = (L2_NEURONS > 1) ? $clog2(L2_NEURONS) : 1,
    parameter int L2_ADDR_W        = (L2_INPUT_BEATS > 1) ? $clog2(L2_INPUT_BEATS) : 1
)(
    input  logic                             clk,
    input  logic                             rst,

    input  logic                             cfg_valid,
    output logic                             cfg_ready,
    input  logic [CONFIG_BUS_WIDTH-1:0]      cfg_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0]    cfg_keep,
    input  logic                             cfg_last,

    output logic                             l0_cfg_we,
    output logic                             l0_cfg_tw,
    output logic [L0_NIDX_W-1:0]             l0_cfg_nidx,
    output logic [L0_ADDR_W-1:0]             l0_cfg_addr,
    output logic [PW-1:0]                    l0_cfg_wdata,
    output logic [L0_COUNT_WIDTH-1:0]        l0_cfg_tdata,
    input  logic                             l0_cfg_rdy,

    output logic                             l1_cfg_we,
    output logic                             l1_cfg_tw,
    output logic [L1_NIDX_W-1:0]             l1_cfg_nidx,
    output logic [L1_ADDR_W-1:0]             l1_cfg_addr,
    output logic [PW-1:0]                    l1_cfg_wdata,
    output logic [L1_COUNT_WIDTH-1:0]        l1_cfg_tdata,
    input  logic                             l1_cfg_rdy,

    output logic                             l2_cfg_we,
    output logic                             l2_cfg_tw,
    output logic [L2_NIDX_W-1:0]             l2_cfg_nidx,
    output logic [L2_ADDR_W-1:0]             l2_cfg_addr,
    output logic [PW-1:0]                    l2_cfg_wdata,
    output logic [L2_COUNT_WIDTH-1:0]        l2_cfg_tdata,
    input  logic                             l2_cfg_rdy
);

    localparam int MAX_NEURONS = (L0_NEURONS >= L1_NEURONS) ?
                                 ((L0_NEURONS >= L2_NEURONS) ? L0_NEURONS : L2_NEURONS) :
                                 ((L1_NEURONS >= L2_NEURONS) ? L1_NEURONS : L2_NEURONS);

    localparam int MAX_BEATS   = (L0_INPUT_BEATS >= L1_INPUT_BEATS) ?
                                 ((L0_INPUT_BEATS >= L2_INPUT_BEATS) ? L0_INPUT_BEATS : L2_INPUT_BEATS) :
                                 ((L1_INPUT_BEATS >= L2_INPUT_BEATS) ? L1_INPUT_BEATS : L2_INPUT_BEATS);

    localparam int MAX_NIDX_W  = (MAX_NEURONS > 1) ? $clog2(MAX_NEURONS) : 1;
    localparam int MAX_ADDR_W  = (MAX_BEATS > 1) ? $clog2(MAX_BEATS) : 1;

    typedef enum logic [2:0] {
        IDLE         = 3'd0,
        RECV_HEADER  = 3'd1,
        DECODE       = 3'd2,
        SKIP_PAYLOAD = 3'd3,
        RECV_W_BYTE  = 3'd4,
        WRITE_W      = 3'd5,
        RECV_T_BYTES = 3'd6,
        WRITE_T      = 3'd7
    } state_t;

    state_t state, next;

    function automatic logic layer_known(input logic [7:0] lid);
        begin
            case (lid)
                8'd0, 8'd1, 8'd2: layer_known = 1'b1;
                default:          layer_known = 1'b0;
            endcase
        end
    endfunction

    function automatic [15:0] layer_input_beats(input logic [7:0] lid);
        begin
            case (lid)
                8'd0:    layer_input_beats = L0_INPUT_BEATS[15:0];
                8'd1:    layer_input_beats = L1_INPUT_BEATS[15:0];
                8'd2:    layer_input_beats = L2_INPUT_BEATS[15:0];
                default: layer_input_beats = 16'd0;
            endcase
        end
    endfunction

    function automatic [5:0] layer_countw(input logic [7:0] lid);
        begin
            case (lid)
                8'd0:    layer_countw = L0_COUNT_WIDTH[5:0];
                8'd1:    layer_countw = L1_COUNT_WIDTH[5:0];
                8'd2:    layer_countw = L2_COUNT_WIDTH[5:0];
                default: layer_countw = 6'd1;
            endcase
        end
    endfunction

    function automatic [31:0] trunc32(
        input logic [31:0] raw,
        input logic [5:0]  cw
    );
        logic [31:0] mask32;
        begin
            if (cw >= 32)
                mask32 = 32'hFFFF_FFFF;
            else
                mask32 = (32'd1 << cw) - 1;
            trunc32 = raw & mask32;
        end
    endfunction

    logic [63:0] beat_data_q;
    logic [7:0]  beat_keep_q;
    logic [2:0]  beat_ptr_q;
    logic        beat_empty_q;

    logic [7:0]  byte_out;
    logic        byte_valid;
    logic        byte_rd_en;

    assign byte_out   = beat_data_q[beat_ptr_q*8 +: 8];
    assign byte_valid = (!beat_empty_q) && beat_keep_q[beat_ptr_q];
    assign cfg_ready  = beat_empty_q;

    always_ff @(posedge clk) begin
        if (rst) begin
            beat_data_q  <= '0;
            beat_keep_q  <= '0;
            beat_ptr_q   <= '0;
            beat_empty_q <= 1'b1;
        end else begin
            if (beat_empty_q) begin
                if (cfg_valid) begin
                    beat_data_q  <= cfg_data;
                    beat_keep_q  <= cfg_keep;
                    beat_ptr_q   <= 3'd0;
                    beat_empty_q <= (cfg_keep == 8'h00);
                end
            end else begin
                if (!beat_keep_q[beat_ptr_q] || byte_rd_en) begin
                    if (beat_ptr_q == 3'd7)
                        beat_empty_q <= 1'b1;
                    else
                        beat_ptr_q <= beat_ptr_q + 3'd1;
                end
            end
        end
    end

    logic [7:0]  hdr_msgtype, hdr_layerid;
    logic [15:0] hdr_layerinputs, hdr_numneurons, hdr_bytesperneu;
    logic [31:0] hdr_totalbytes;
    logic [3:0]  hdr_byte_cnt;

    logic [31:0] payload_bytes_left;
    logic [7:0]  active_layerid;
    logic [15:0] active_beats_per_neu;
    logic [5:0]  active_countw;
    logic        active_layer_valid;

    logic [MAX_NIDX_W-1:0] cur_neuron;
    logic [MAX_ADDR_W-1:0] cur_beat;

    logic [7:0]  weight_byte_q;
    logic [31:0] thresh_reg;
    logic [1:0]  thresh_byte_cnt;

    logic        active_cfg_rdy;
    logic        dec_layer_valid;
    logic [15:0] dec_beats_per_neu;
    logic [5:0]  dec_countw;

    assign dec_layer_valid   = layer_known(hdr_layerid);
    assign dec_beats_per_neu = layer_input_beats(hdr_layerid);
    assign dec_countw        = layer_countw(hdr_layerid);

    always_comb begin
        case (active_layerid)
            8'd0:    active_cfg_rdy = l0_cfg_rdy;
            8'd1:    active_cfg_rdy = l1_cfg_rdy;
            8'd2:    active_cfg_rdy = l2_cfg_rdy;
            default: active_cfg_rdy = 1'b1;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state              <= IDLE;
            hdr_msgtype        <= '0;
            hdr_layerid        <= '0;
            hdr_layerinputs    <= '0;
            hdr_numneurons     <= '0;
            hdr_bytesperneu    <= '0;
            hdr_totalbytes     <= '0;
            hdr_byte_cnt       <= '0;
            payload_bytes_left <= '0;
            active_layerid     <= '0;
            active_beats_per_neu <= '0;
            active_countw      <= '0;
            active_layer_valid <= 1'b0;
            cur_neuron         <= '0;
            cur_beat           <= '0;
            weight_byte_q      <= '0;
            thresh_reg         <= '0;
            thresh_byte_cnt    <= '0;
        end else begin
            state <= next;

            case (state)
                IDLE: begin
                    hdr_byte_cnt       <= '0;
                    payload_bytes_left <= '0;
                    cur_neuron         <= '0;
                    cur_beat           <= '0;
                    thresh_reg         <= '0;
                    thresh_byte_cnt    <= '0;
                end

                RECV_HEADER: begin
                    if (byte_valid && byte_rd_en) begin
                        case (hdr_byte_cnt)
                            4'd0:  hdr_msgtype           <= byte_out;
                            4'd1:  hdr_layerid           <= byte_out;
                            4'd2:  hdr_layerinputs[7:0]  <= byte_out;
                            4'd3:  hdr_layerinputs[15:8] <= byte_out;
                            4'd4:  hdr_numneurons[7:0]   <= byte_out;
                            4'd5:  hdr_numneurons[15:8]  <= byte_out;
                            4'd6:  hdr_bytesperneu[7:0]  <= byte_out;
                            4'd7:  hdr_bytesperneu[15:8] <= byte_out;
                            4'd8:  hdr_totalbytes[7:0]   <= byte_out;
                            4'd9:  hdr_totalbytes[15:8]  <= byte_out;
                            4'd10: hdr_totalbytes[23:16] <= byte_out;
                            4'd11: hdr_totalbytes[31:24] <= byte_out;
                            default: ;
                        endcase
                        hdr_byte_cnt <= hdr_byte_cnt + 4'd1;
                    end
                end

                DECODE: begin
                    payload_bytes_left  <= hdr_totalbytes;
                    active_layerid      <= hdr_layerid;
                    active_beats_per_neu <= dec_beats_per_neu;
                    active_countw       <= dec_countw;
                    active_layer_valid  <= dec_layer_valid;
                    cur_neuron          <= '0;
                    cur_beat            <= '0;
                    thresh_reg          <= '0;
                    thresh_byte_cnt     <= '0;
                end

                SKIP_PAYLOAD: begin
                    if (byte_valid && byte_rd_en)
                        payload_bytes_left <= payload_bytes_left - 32'd1;
                end

                RECV_W_BYTE: begin
                    if (byte_valid && byte_rd_en) begin
                        weight_byte_q      <= byte_out;
                        payload_bytes_left <= payload_bytes_left - 32'd1;
                    end
                end

                WRITE_W: begin
                    if (active_cfg_rdy) begin
                        if (cur_beat == active_beats_per_neu - 1) begin
                            cur_beat   <= '0;
                            cur_neuron <= cur_neuron + 1'b1;
                        end else begin
                            cur_beat <= cur_beat + 1'b1;
                        end
                    end
                end

                RECV_T_BYTES: begin
                    if (byte_valid && byte_rd_en) begin
                        case (thresh_byte_cnt)
                            2'd0: thresh_reg[7:0]   <= byte_out;
                            2'd1: thresh_reg[15:8]  <= byte_out;
                            2'd2: thresh_reg[23:16] <= byte_out;
                            2'd3: thresh_reg[31:24] <= byte_out;
                        endcase
                        thresh_byte_cnt    <= thresh_byte_cnt + 2'd1;
                        payload_bytes_left <= payload_bytes_left - 32'd1;
                    end
                end

                WRITE_T: begin
                    if (active_cfg_rdy) begin
                        cur_neuron      <= cur_neuron + 1'b1;
                        thresh_reg      <= '0;
                        thresh_byte_cnt <= '0;
                    end
                end

                default: ;
            endcase
        end
    end

    always_comb begin
        next       = state;
        byte_rd_en = 1'b0;

        l0_cfg_we    = 1'b0;
        l0_cfg_tw    = 1'b0;
        l0_cfg_nidx  = '0;
        l0_cfg_addr  = '0;
        l0_cfg_wdata = '0;
        l0_cfg_tdata = '0;

        l1_cfg_we    = 1'b0;
        l1_cfg_tw    = 1'b0;
        l1_cfg_nidx  = '0;
        l1_cfg_addr  = '0;
        l1_cfg_wdata = '0;
        l1_cfg_tdata = '0;

        l2_cfg_we    = 1'b0;
        l2_cfg_tw    = 1'b0;
        l2_cfg_nidx  = '0;
        l2_cfg_addr  = '0;
        l2_cfg_wdata = '0;
        l2_cfg_tdata = '0;

        case (state)
            IDLE: begin
                if (byte_valid)
                    next = RECV_HEADER;
            end

            RECV_HEADER: begin
                if (byte_valid) begin
                    byte_rd_en = 1'b1;
                    if (hdr_byte_cnt == 4'd15)
                        next = DECODE;
                end
            end

            DECODE: begin
                if (!dec_layer_valid)
                    next = SKIP_PAYLOAD;
                else if (hdr_msgtype == 8'd0)
                    next = RECV_W_BYTE;
                else
                    next = RECV_T_BYTES;
            end

            SKIP_PAYLOAD: begin
                if (payload_bytes_left == 0)
                    next = IDLE;
                else if (byte_valid)
                    byte_rd_en = 1'b1;
            end

            RECV_W_BYTE: begin
                if (payload_bytes_left == 0)
                    next = IDLE;
                else if (byte_valid) begin
                    byte_rd_en = 1'b1;
                    next = WRITE_W;
                end
            end

            WRITE_W: begin
                case (active_layerid)
                    8'd0: begin
                        l0_cfg_we    = active_layer_valid;
                        l0_cfg_nidx  = cur_neuron[L0_NIDX_W-1:0];
                        l0_cfg_addr  = cur_beat[L0_ADDR_W-1:0];
                        l0_cfg_wdata = weight_byte_q[PW-1:0];
                    end
                    8'd1: begin
                        l1_cfg_we    = active_layer_valid;
                        l1_cfg_nidx  = cur_neuron[L1_NIDX_W-1:0];
                        l1_cfg_addr  = cur_beat[L1_ADDR_W-1:0];
                        l1_cfg_wdata = weight_byte_q[PW-1:0];
                    end
                    8'd2: begin
                        l2_cfg_we    = active_layer_valid;
                        l2_cfg_nidx  = cur_neuron[L2_NIDX_W-1:0];
                        l2_cfg_addr  = cur_beat[L2_ADDR_W-1:0];
                        l2_cfg_wdata = weight_byte_q[PW-1:0];
                    end
                    default: ;
                endcase

                if (active_cfg_rdy) begin
                    if (payload_bytes_left == 0)
                        next = IDLE;
                    else
                        next = RECV_W_BYTE;
                end
            end

            RECV_T_BYTES: begin
                if (payload_bytes_left == 0)
                    next = IDLE;
                else if (byte_valid) begin
                    byte_rd_en = 1'b1;
                    if (thresh_byte_cnt == 2'd3)
                        next = WRITE_T;
                end
            end

            WRITE_T: begin
                case (active_layerid)
                    8'd0: begin
                        l0_cfg_tw    = active_layer_valid;
                        l0_cfg_nidx  = cur_neuron[L0_NIDX_W-1:0];
                        l0_cfg_tdata = trunc32(thresh_reg, active_countw)[L0_COUNT_WIDTH-1:0];
                    end
                    8'd1: begin
                        l1_cfg_tw    = active_layer_valid;
                        l1_cfg_nidx  = cur_neuron[L1_NIDX_W-1:0];
                        l1_cfg_tdata = trunc32(thresh_reg, active_countw)[L1_COUNT_WIDTH-1:0];
                    end
                    8'd2: begin
                        l2_cfg_tw    = active_layer_valid;
                        l2_cfg_nidx  = cur_neuron[L2_NIDX_W-1:0];
                        l2_cfg_tdata = trunc32(thresh_reg, active_countw)[L2_COUNT_WIDTH-1:0];
                    end
                    default: ;
                endcase

                if (active_cfg_rdy) begin
                    if (payload_bytes_left == 0)
                        next = IDLE;
                    else
                        next = RECV_T_BYTES;
                end
            end

            default: next = IDLE;
        endcase
    end

endmodule