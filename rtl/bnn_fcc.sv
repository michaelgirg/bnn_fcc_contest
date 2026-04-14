module bnn_fcc #(
    parameter int INPUT_DATA_WIDTH = 8,
    parameter int INPUT_BUS_WIDTH = 64,
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int OUTPUT_DATA_WIDTH = 4,
    parameter int OUTPUT_BUS_WIDTH = 8,
    parameter int TOTAL_LAYERS = 4,
    parameter int TOPOLOGY[TOTAL_LAYERS] = '{0: 784, 1: 256, 2: 256, 3: 10, default: 0},
    parameter int PARALLEL_INPUTS = 8,
    parameter int PARALLEL_NEURONS[TOTAL_LAYERS-1] = '{default: 8}
) (
    input  logic                          clk,
    input  logic                          rst,
    input  logic                          config_valid,
    output logic                          config_ready,
    input  logic [  CONFIG_BUS_WIDTH-1:0] config_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0] config_keep,
    input  logic                          config_last,
    input  logic                          data_in_valid,
    output logic                          data_in_ready,
    input  logic [   INPUT_BUS_WIDTH-1:0] data_in_data,
    input  logic [ INPUT_BUS_WIDTH/8-1:0] data_in_keep,
    input  logic                          data_in_last,
    output logic                          data_out_valid,
    input  logic                          data_out_ready,
    output logic [  OUTPUT_BUS_WIDTH-1:0] data_out_data,
    output logic [OUTPUT_BUS_WIDTH/8-1:0] data_out_keep,
    output logic                          data_out_last
);
    localparam int INPUTS = TOPOLOGY[0];
    localparam int HIDDEN1 = TOPOLOGY[1];
    localparam int HIDDEN2 = TOPOLOGY[2];
    localparam int OUTPUTS = TOPOLOGY[3];
    localparam int OUTPUT_KEEP_W = OUTPUT_BUS_WIDTH / 8;
    localparam int OUT_BYTES_USED = (OUTPUT_DATA_WIDTH + 7) / 8;
    localparam int L1_COUNT_W = $clog2(INPUTS + 1);
    localparam int L2_COUNT_W = $clog2(HIDDEN1 + 1);
    localparam int L3_COUNT_W = $clog2(HIDDEN2 + 1);
    localparam int THRESHOLD_W = (L1_COUNT_W > L2_COUNT_W) ?
                                 (L1_COUNT_W > L3_COUNT_W ? L1_COUNT_W : L3_COUNT_W) :
                                 (L2_COUNT_W > L3_COUNT_W ? L2_COUNT_W : L3_COUNT_W);
    localparam int MAX_NEURONS  = (HIDDEN1 > HIDDEN2) ?
                                  (HIDDEN1 > OUTPUTS ? HIDDEN1 : OUTPUTS) :
                                  (HIDDEN2 > OUTPUTS ? HIDDEN2 : OUTPUTS);
    localparam int CFG_NEURON_W = (MAX_NEURONS > 1) ? $clog2(MAX_NEURONS) : 1;
    localparam int MAX_BEATS_L1 = (INPUTS + PARALLEL_INPUTS - 1) / PARALLEL_INPUTS;
    localparam int MAX_BEATS_L2 = (HIDDEN1 + PARALLEL_INPUTS - 1) / PARALLEL_INPUTS;
    localparam int MAX_BEATS_L3 = (HIDDEN2 + PARALLEL_INPUTS - 1) / PARALLEL_INPUTS;
    localparam int MAX_WEIGHT_BEATS = (MAX_BEATS_L1 > MAX_BEATS_L2) ?
                                      (MAX_BEATS_L1 > MAX_BEATS_L3 ? MAX_BEATS_L1 : MAX_BEATS_L3) :
                                      (MAX_BEATS_L2 > MAX_BEATS_L3 ? MAX_BEATS_L2 : MAX_BEATS_L3);
    localparam int CFG_WEIGHT_ADDR_W = (MAX_WEIGHT_BEATS > 1) ? $clog2(MAX_WEIGHT_BEATS) : 1;
    localparam int CLASS_W = (OUTPUTS > 1) ? $clog2(OUTPUTS) : 1;
    localparam int GLOBAL_NEURON_W = $clog2(HIDDEN1 + HIDDEN2 + OUTPUTS);

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_RUN,
        ST_OUT
    } run_state_t;

    run_state_t run_state_r;
    logic cfg_ready_i;
    logic cfg_write_en_i;
    logic [1:0] cfg_layer_sel_i;
    logic [CFG_NEURON_W-1:0] cfg_neuron_idx_i;
    logic [CFG_WEIGHT_ADDR_W-1:0] cfg_weight_addr_i;
    logic [PARALLEL_INPUTS-1:0] cfg_weight_data_i;
    logic [THRESHOLD_W-1:0] cfg_threshold_data_i;
    logic cfg_threshold_write_i;
    logic core_cfg_ready_i;
    logic [GLOBAL_NEURON_W-1:0] cfg_global_neuron_idx_i;
    logic config_done_r;
    logic config_last_seen_r;

    // Binarization signals
    logic [INPUTS-1:0] bin_inputs_i;
    logic binarized_valid_i;
    logic binarized_ready_i;

    logic core_image_valid_r;
    logic core_image_ready_i;
    logic core_done_i;
    logic core_busy_i;
    logic core_result_valid_i;
    logic [OUTPUTS*L3_COUNT_W-1:0] core_popcounts_i;
    logic argmax_valid_i;
    logic [CLASS_W-1:0] argmax_class_i;
    logic [L3_COUNT_W-1:0] argmax_score_i;
    logic infer_started_r;
    logic data_out_valid_r;
    logic [OUTPUT_BUS_WIDTH-1:0] data_out_data_r;

    assign config_ready = cfg_ready_i;
    assign core_busy_i = !core_image_ready_i;
    assign core_cfg_ready_i = core_image_ready_i;

    //Convert layer_sel + neuron_idx to global_neuron_idx
    always_comb begin
        case (cfg_layer_sel_i)
            2'd0: cfg_global_neuron_idx_i = GLOBAL_NEURON_W'(cfg_neuron_idx_i);
            2'd1: cfg_global_neuron_idx_i = GLOBAL_NEURON_W'(HIDDEN1 + cfg_neuron_idx_i);
            2'd2: cfg_global_neuron_idx_i = GLOBAL_NEURON_W'(HIDDEN1 + HIDDEN2 + cfg_neuron_idx_i);
            default: cfg_global_neuron_idx_i = '0;
        endcase
    end

    always_comb begin
        data_out_valid = data_out_valid_r;
        data_out_data  = data_out_data_r;
        data_out_keep  = '0;
        for (int i = 0; i < OUT_BYTES_USED; i++) begin
            if (i < OUTPUT_KEEP_W) data_out_keep[i] = data_out_valid_r;
        end
        data_out_last = data_out_valid_r;
    end

    config_manager_multi #(
        .CONFIG_BUS_WIDTH(CONFIG_BUS_WIDTH),
        .PW              (PARALLEL_INPUTS),
        .L0_INPUTS       (INPUTS),
        .L0_NEURONS      (HIDDEN1),
        .L0_COUNT_WIDTH  ($clog2(INPUTS + 1)),
        .L0_INPUT_BEATS  ((INPUTS + PARALLEL_INPUTS - 1) / PARALLEL_INPUTS),
        .L1_INPUTS       (HIDDEN1),
        .L1_NEURONS      (HIDDEN2),
        .L1_COUNT_WIDTH  ($clog2(HIDDEN1 + 1)),
        .L1_INPUT_BEATS  ((HIDDEN1 + PARALLEL_INPUTS - 1) / PARALLEL_INPUTS),
        .L2_INPUTS       (HIDDEN2),
        .L2_NEURONS      (OUTPUTS),
        .L2_COUNT_WIDTH  ($clog2(HIDDEN2 + 1)),
        .L2_INPUT_BEATS  ((HIDDEN2 + PARALLEL_INPUTS - 1) / PARALLEL_INPUTS)
    ) u_config_manager (
        .clk                    (clk),
        .rst                    (rst),
        .cfg_valid              (config_valid),
        .cfg_ready              (cfg_ready_i),
        .cfg_data               (config_data),
        .cfg_keep               (config_keep),
        .cfg_last               (config_last),
        .out_cfg_write_en       (cfg_write_en_i),
        .out_cfg_layer_sel      (cfg_layer_sel_i),
        .out_cfg_neuron_idx     (cfg_neuron_idx_i),
        .out_cfg_weight_addr    (cfg_weight_addr_i),
        .out_cfg_weight_data    (cfg_weight_data_i),
        .out_cfg_threshold_data (cfg_threshold_data_i),
        .out_cfg_threshold_write(cfg_threshold_write_i),
        .out_cfg_ready          (core_cfg_ready_i)
    );

    // Streaming input binarization
    input_binarize #(
        .PIXELS   (INPUTS),
        .PIXEL_W  (INPUT_DATA_WIDTH),
        .BUS_WIDTH(INPUT_BUS_WIDTH),
        .THRESHOLD(INPUT_DATA_WIDTH'(8'd128))
    ) u_input_binarize (
        .clk            (clk),
        .rst            (rst),
        .data_in_data   (data_in_data),
        .data_in_valid  (data_in_valid),
        .data_in_ready  (data_in_ready),
        .data_in_last   (data_in_last),
        .binarized_out  (bin_inputs_i),
        .binarized_valid(binarized_valid_i),
        .binarized_ready(binarized_ready_i)
    );

    bnn_core #(
        .INPUTS (INPUTS),
        .HIDDEN1(HIDDEN1),
        .HIDDEN2(HIDDEN2),
        .OUTPUTS(OUTPUTS),
        .PW     (PARALLEL_INPUTS),
        .PN     (PARALLEL_NEURONS[0])
    ) u_bnn_core (
        .clk                  (clk),
        .rst                  (rst),
        .image_input          (bin_inputs_i),
        .image_valid          (core_image_valid_r),
        .image_ready          (core_image_ready_i),
        .result_activations   (  /* unused */),
        .result_popcounts     (core_popcounts_i),
        .result_valid         (core_result_valid_i),
        .done                 (core_done_i),
        .cfg_write_en         (cfg_write_en_i),
        .cfg_global_neuron_idx(cfg_global_neuron_idx_i),
        .cfg_weight_addr      (cfg_weight_addr_i),
        .cfg_weight_data      (cfg_weight_data_i),
        .cfg_threshold_data   (cfg_threshold_data_i),
        .cfg_threshold_write  (cfg_threshold_write_i)
    );

    argmax #(
        .OUTPUTS(OUTPUTS),
        .COUNT_W(L3_COUNT_W)
    ) u_argmax (
        .valid_in    (core_result_valid_i),
        .popcounts_in(core_popcounts_i),
        .valid_out   (argmax_valid_i),
        .class_idx   (argmax_class_i),
        .max_value   (argmax_score_i)
    );

    always_ff @(posedge clk) begin
        if (rst) begin
            config_last_seen_r <= 1'b0;
            config_done_r      <= 1'b0;
        end else begin
            if (config_valid && cfg_ready_i && config_last) config_last_seen_r <= 1'b1;
            if (config_last_seen_r && cfg_ready_i) config_done_r <= 1'b1;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            infer_started_r <= 1'b0;
        end else begin
            case (run_state_r)
                ST_IDLE: begin
                    infer_started_r <= 1'b0;
                end
                ST_RUN: begin
                    if (core_busy_i) infer_started_r <= 1'b1;
                end
                ST_OUT: begin
                    if (data_out_valid_r && data_out_ready) infer_started_r <= 1'b0;
                end
                default: begin
                    infer_started_r <= 1'b0;
                end
            endcase
        end
    end

    // Simplified 3-state FSM
    always_ff @(posedge clk) begin
        if (rst) begin
            run_state_r        <= ST_IDLE;
            core_image_valid_r <= 1'b0;
            binarized_ready_i  <= 1'b0;
            data_out_valid_r   <= 1'b0;
            data_out_data_r    <= '0;
        end else begin
            core_image_valid_r <= 1'b0;

            case (run_state_r)
                ST_IDLE: begin
                    if (config_done_r && binarized_valid_i && !core_busy_i) begin
                        binarized_ready_i  <= 1'b1;
                        core_image_valid_r <= 1'b1;
                        run_state_r        <= ST_RUN;
                    end
                end

                ST_RUN: begin
                    binarized_ready_i <= 1'b0;
                    if (infer_started_r && core_done_i && argmax_valid_i) begin
                        data_out_valid_r <= 1'b1;
                        data_out_data_r <= '0;
                        data_out_data_r[CLASS_W-1:0] <= argmax_class_i;
                        run_state_r <= ST_OUT;
                    end
                end

                ST_OUT: begin
                    if (data_out_valid_r && data_out_ready) begin
                        data_out_valid_r <= 1'b0;
                        data_out_data_r  <= '0;
                        run_state_r      <= ST_IDLE;
                    end
                end

                default: begin
                    run_state_r <= ST_IDLE;
                end
            endcase
        end
    end
endmodule
