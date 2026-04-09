`timescale 1ns / 1ps

module config_manager_multi #(
    parameter int CONFIG_BUS_WIDTH = 64,

    parameter int L0_INPUTS  = 784,
    parameter int L0_NEURONS = 256,
    parameter int L0_PW      = 8,

    parameter int L1_INPUTS  = 256,
    parameter int L1_NEURONS = 256,
    parameter int L1_PW      = 8,

    parameter int L2_INPUTS  = 256,
    parameter int L2_NEURONS = 10,
    parameter int L2_PW      = 8
)(
    input  logic clk,
    input  logic rst,

    input  logic                        cfg_valid,
    output logic                        cfg_ready,
    input  logic [CONFIG_BUS_WIDTH-1:0] cfg_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0] cfg_keep,
    input  logic                        cfg_last,

    // Layer 0 config port
    output logic                                       l0_cfg_we,
    output logic                                       l0_cfg_tw,
    output logic[$clog2((L0_NEURONS > 1) ? L0_NEURONS : 2)-1:0] l0_cfg_nidx,
    output logic[$clog2((((L0_INPUTS + L0_PW - 1)/L0_PW) > 1) ? ((L0_INPUTS + L0_PW - 1)/L0_PW) : 2)-1:0] l0_cfg_addr,
    output logic[L0_PW-1:0]                            l0_cfg_wdata,
    output logic[$clog2(L0_INPUTS+1)-1:0]             l0_cfg_tdata,
    input  logic                                       l0_cfg_rdy,

    // Layer 1 config port
    output logic                                       l1_cfg_we,
    output logic                                       l1_cfg_tw,
    output logic[$clog2((L1_NEURONS > 1) ? L1_NEURONS : 2)-1:0] l1_cfg_nidx,
    output logic[$clog2((((L1_INPUTS + L1_PW - 1)/L1_PW) > 1) ? ((L1_INPUTS + L1_PW - 1)/L1_PW) : 2)-1:0] l1_cfg_addr,
    output logic[L1_PW-1:0]                            l1_cfg_wdata,
    output logic[$clog2(L1_INPUTS+1)-1:0]             l1_cfg_tdata,
    input  logic                                       l1_cfg_rdy,

    // Layer 2 config port
    output logic                                       l2_cfg_we,
    output logic                                       l2_cfg_tw,
    output logic[$clog2((L2_NEURONS > 1) ? L2_NEURONS : 2)-1:0] l2_cfg_nidx,
    output logic[$clog2((((L2_INPUTS + L2_PW - 1)/L2_PW) > 1) ? ((L2_INPUTS + L2_PW - 1)/L2_PW) : 2)-1:0] l2_cfg_addr,
    output logic[L2_PW-1:0]                            l2_cfg_wdata,
    output logic[$clog2(L2_INPUTS+1)-1:0]             l2_cfg_tdata,
    input  logic                                       l2_cfg_rdy
);

    logic p0_cfg_ready, p1_cfg_ready, p2_cfg_ready;

    assign cfg_ready = p0_cfg_ready & p1_cfg_ready & p2_cfg_ready;

    config_parser #(
        .TARGET_LAYER_ID(0),
        .INPUTS(L0_INPUTS),
        .NEURONS(L0_NEURONS),
        .PW(L0_PW)
    ) u_parser_l0 (
        .clk(clk),
        .rst(rst),
        .cfg_valid(cfg_valid),
        .cfg_ready(p0_cfg_ready),
        .cfg_data(cfg_data),
        .cfg_keep(cfg_keep),
        .cfg_last(cfg_last),
        .cfg_we(l0_cfg_we),
        .cfg_tw(l0_cfg_tw),
        .cfg_nidx(l0_cfg_nidx),
        .cfg_addr(l0_cfg_addr),
        .cfg_wdata(l0_cfg_wdata),
        .cfg_tdata(l0_cfg_tdata),
        .cfg_rdy(l0_cfg_rdy)
    );

    config_parser #(
        .TARGET_LAYER_ID(1),
        .INPUTS(L1_INPUTS),
        .NEURONS(L1_NEURONS),
        .PW(L1_PW)
    ) u_parser_l1 (
        .clk(clk),
        .rst(rst),
        .cfg_valid(cfg_valid),
        .cfg_ready(p1_cfg_ready),
        .cfg_data(cfg_data),
        .cfg_keep(cfg_keep),
        .cfg_last(cfg_last),
        .cfg_we(l1_cfg_we),
        .cfg_tw(l1_cfg_tw),
        .cfg_nidx(l1_cfg_nidx),
        .cfg_addr(l1_cfg_addr),
        .cfg_wdata(l1_cfg_wdata),
        .cfg_tdata(l1_cfg_tdata),
        .cfg_rdy(l1_cfg_rdy)
    );

    config_parser #(
        .TARGET_LAYER_ID(2),
        .INPUTS(L2_INPUTS),
        .NEURONS(L2_NEURONS),
        .PW(L2_PW)
    ) u_parser_l2 (
        .clk(clk),
        .rst(rst),
        .cfg_valid(cfg_valid),
        .cfg_ready(p2_cfg_ready),
        .cfg_data(cfg_data),
        .cfg_keep(cfg_keep),
        .cfg_last(cfg_last),
        .cfg_we(l2_cfg_we),
        .cfg_tw(l2_cfg_tw),
        .cfg_nidx(l2_cfg_nidx),
        .cfg_addr(l2_cfg_addr),
        .cfg_wdata(l2_cfg_wdata),
        .cfg_tdata(l2_cfg_tdata),
        .cfg_rdy(l2_cfg_rdy)
    );

endmodule