
`timescale 1ns / 100ps
module bnn_fcc_coverage_tb #(
    // Testbench configuration
    parameter int      USE_CUSTOM_TOPOLOGY                      = 1'b0,
    parameter int      TOPOLOGY_MODE                            = 0,
    parameter int      CUSTOM_LAYERS                            = 4,
    parameter int      CUSTOM_TOPOLOGY          [CUSTOM_LAYERS] = '{8, 8, 8, 8},
    parameter int      NUM_TEST_IMAGES                          = 50,
    parameter bit      VERIFY_MODEL                             = 1,
    parameter string   BASE_DIR                                 = "python",
    parameter bit      TOGGLE_DATA_OUT_READY                    = 1'b1,
    parameter real     CONFIG_VALID_PROBABILITY                 = 0.8,
    parameter real     DATA_IN_VALID_PROBABILITY                = 0.8,
    parameter realtime TIMEOUT                                  = 40ms,
    parameter realtime CLK_PERIOD                               = 10ns,
    parameter bit      DEBUG                                    = 1'b0,

    // Coverage controls
    parameter bit ENABLE_NONSTANDARD_CONFIG_ORDER = 1'b1,
    parameter bit ENABLE_PARTIAL_RECONFIG         = 1'b1,
    parameter bit ENABLE_RESET_STRESS             = 1'b1,
    parameter int EXTRA_RANDOM_IMAGES             = 12,
    parameter int REPEATED_CLASS_BURST_LEN        = 3,

    // Bus configuration
    parameter int CONFIG_BUS_WIDTH = 64,
    parameter int INPUT_BUS_WIDTH  = 64,
    parameter int OUTPUT_BUS_WIDTH = 8,

    // App configuration
    parameter  int INPUT_DATA_WIDTH  = 8,
    localparam int INPUTS_PER_CYCLE  = INPUT_BUS_WIDTH / INPUT_DATA_WIDTH,
    localparam int BYTES_PER_INPUT   = INPUT_DATA_WIDTH / 8,
    parameter  int OUTPUT_DATA_WIDTH = 4,

    // Should not be changed
    localparam int TRAINED_LAYERS = 4,
    localparam int TRAINED_TOPOLOGY[TRAINED_LAYERS] = '{784, 256, 256, 10},

    // DUT configuration
    localparam int NON_INPUT_LAYERS =
        !USE_CUSTOM_TOPOLOGY ? TRAINED_LAYERS - 1 :
        (TOPOLOGY_MODE == 0) ? TRAINED_LAYERS - 1 :
        (TOPOLOGY_MODE == 1) ? 3 :
        (TOPOLOGY_MODE == 2) ? 4 :
        (TOPOLOGY_MODE == 3) ? 3 :
        CUSTOM_LAYERS - 1,
    parameter int PARALLEL_INPUTS = 8,
    parameter int PARALLEL_NEURONS[NON_INPUT_LAYERS] = '{default: 8}
);
    import bnn_fcc_tb_pkg::*;

    // =========================================================================
    // Local parameters / typedefs
    // =========================================================================
    localparam int MODE0_LAYERS = 4;
    localparam int MODE0_TOPOLOGY[MODE0_LAYERS] = '{784, 256, 256, 10};

    localparam int MODE1_LAYERS = 4;
    localparam int MODE1_TOPOLOGY[MODE1_LAYERS] = '{10, 8, 8, 4};

    localparam int MODE2_LAYERS = 5;
    localparam int MODE2_TOPOLOGY[MODE2_LAYERS] = '{10, 8, 8, 8, 4};

    localparam int MODE3_LAYERS = 4;
    localparam int MODE3_TOPOLOGY[MODE3_LAYERS] = '{9, 4, 4, 3};

    localparam int ACTUAL_TOTAL_LAYERS =
        !USE_CUSTOM_TOPOLOGY ? TRAINED_LAYERS :
        (TOPOLOGY_MODE == 0) ? MODE0_LAYERS :
        (TOPOLOGY_MODE == 1) ? MODE1_LAYERS :
        (TOPOLOGY_MODE == 2) ? MODE2_LAYERS :
        (TOPOLOGY_MODE == 3) ? MODE3_LAYERS :
        CUSTOM_LAYERS;

    function automatic int topology_at(input int idx);
        if (!USE_CUSTOM_TOPOLOGY) return TRAINED_TOPOLOGY[idx];
        case (TOPOLOGY_MODE)
            0: return MODE0_TOPOLOGY[idx];
            1: return MODE1_TOPOLOGY[idx];
            2: return MODE2_TOPOLOGY[idx];
            3: return MODE3_TOPOLOGY[idx];
            default: return CUSTOM_TOPOLOGY[idx];
        endcase
    endfunction
    localparam string MNIST_TEST_VECTOR_INPUT_PATH = "test_vectors/inputs.hex";
    localparam string MNIST_TEST_VECTOR_OUTPUT_PATH = "test_vectors/expected_outputs.txt";
    localparam string MNIST_MODEL_DATA_PATH = "model_data";
    localparam realtime HALF_CLK_PERIOD = CLK_PERIOD / 2.0;
    localparam int OUTPUT_CLASSES = topology_at(ACTUAL_TOTAL_LAYERS - 1);
    localparam int CONFIG_KEEP_WIDTH = CONFIG_BUS_WIDTH / 8;

    typedef bit [CONFIG_BUS_WIDTH-1:0] config_bus_word_t;
    typedef config_bus_word_t config_bus_data_stream_t[];
    typedef bit [CONFIG_KEEP_WIDTH-1:0] config_bus_keep_t;
    typedef config_bus_keep_t config_keep_stream_t[];

    typedef enum int {
        VALID_CONTINUOUS,
        VALID_INTERMITTENT,
        VALID_BURSTY
    } valid_mode_e;

    typedef enum int {
        READY_ALWAYS,
        READY_RANDOM,
        READY_BURSTY,
        READY_DELAYED
    } ready_mode_e;

    typedef enum int {
        CFG_ORDER_STANDARD,
        CFG_ORDER_REVERSE_LAYER,
        CFG_ORDER_WEIGHTS_THEN_THRESHOLDS,
        CFG_ORDER_THRESHOLDS_THEN_WEIGHTS
    } cfg_order_e;

    typedef enum int {
        RESET_BEFORE_CONFIG,
        RESET_DURING_CONFIG,
        RESET_BETWEEN_CONFIG_AND_IMAGE,
        RESET_DURING_IMAGE,
        RESET_DURING_OUTPUT
    } reset_phase_e;

    typedef enum int {
        THRESHOLD_ZERO,
        THRESHOLD_MAX,
        THRESHOLD_MID,
        THRESHOLD_RANDOM
    } threshold_mode_e;

    typedef enum int {
        WEIGHT_ALL_ZERO,
        WEIGHT_ALL_ONE,
        WEIGHT_CHECKER,
        WEIGHT_RANDOMIZED
    } weight_mode_e;

    typedef enum int {
        CONFIG_KIND_FULL,
        CONFIG_KIND_PARTIAL_WEIGHTS,
        CONFIG_KIND_PARTIAL_THRESHOLDS
    } config_kind_e;

    // =========================================================================
    // Utility checks
    // =========================================================================
    initial begin
        assert (INPUT_DATA_WIDTH == 8)
        else
            $fatal(
                1,
                "TB ERROR: INPUT_DATA_WIDTH must be 8. Sub-byte or multi-byte packing logic not yet implemented."
            );
    end

    // Returns 1 with probability p, 0 with probability 1-p.
    function automatic bit chance(real p);
        if (p > 1.0 || p < 0.0) $fatal(1, "Invalid probability in chance()");
        return ($urandom < (p * (2.0 ** 32)));
    endfunction

    function automatic int gap_bucket(int cycles);
        if (cycles == 0) return 0;
        else if (cycles == 1) return 1;
        else if (cycles <= 3) return 2;
        else return 3;
    endfunction

    function automatic int keep_count(input logic [CONFIG_KEEP_WIDTH-1:0] keep);
        int cnt = 0;
        for (int i = 0; i < CONFIG_KEEP_WIDTH; i++) cnt += keep[i];
        return cnt;
    endfunction

    function automatic int input_keep_count(input logic [INPUT_BUS_WIDTH/8-1:0] keep);
        int cnt = 0;
        for (int i = 0; i < INPUT_BUS_WIDTH / 8; i++) cnt += keep[i];
        return cnt;
    endfunction

    function automatic int classify_pixel(input bit [INPUT_DATA_WIDTH-1:0] pix);
        if (pix == 8'd0) return 0;
        else if (pix == 8'd127) return 1;
        else if (pix == 8'd128) return 2;
        else if (pix == 8'd255) return 3;
        else if (pix < 8'd128) return 4;
        else return 5;
    endfunction

    // =========================================================================
    // Model / stimulus / statistics
    // =========================================================================
    BNN_FCC_Model #(CONFIG_BUS_WIDTH)                             model;
    BNN_FCC_Stimulus #(INPUT_DATA_WIDTH)                          stim;
    BNN_FCC_Stimulus #(INPUT_DATA_WIDTH)                          cover_stim;
    LatencyTracker                                                latency;
    ThroughputTracker                                             throughput;

    bit                                  [  CONFIG_BUS_WIDTH-1:0] config_bus_data_stream      [];
    bit                                  [CONFIG_BUS_WIDTH/8-1:0] config_bus_keep_stream      [];

    int                                                           actual_topology_q           [];

    int                                                           num_tests;
    int                                                           passed;
    int                                                           failed;
    int                                                           total_expected_outputs;
    int                                                           total_checked_outputs;
    int                                                           total_reset_count;
    int                                                           issued_image_id;
    int                                                           completed_image_id;
    int                                                           scoreboard_epoch;

    // Coverage bookkeeping
    ready_mode_e                                                  ready_mode_r;
    valid_mode_e                                                  current_config_valid_mode_r;
    valid_mode_e                                                  current_input_valid_mode_r;
    cfg_order_e                                                   current_cfg_order_r;
    config_kind_e                                                 current_config_kind_r;
    reset_phase_e                                                 last_reset_phase_r;
    int                                                           last_config_layer_r;
    int                                                           last_config_msg_type_r;
    int                                                           config_gap_cycles_r;
    int                                                           data_gap_cycles_r;
    int                                                           inter_image_gap_r;
    int                                                           output_wait_cycles_r;
    bit                                                           config_message_started_r;
    bit                                                           output_valid_prev_r;
    bit                                                           output_ready_prev_r;
    int                                                           repeated_output_class_r;
    int                                                           last_output_class_r;
    int                                                           last_class_cover_count_r;
    bit                                                           tready_before_valid_r;
    bit                                                           tready_same_cycle_r;
    bit                                                           tready_after_valid_r;
    bit                                                           output_waiting_for_ready_r;

    logic                                                         clk = 1'b0;
    logic                                                         rst;

    logic                                [ OUTPUT_DATA_WIDTH-1:0] expected_outputs            [$ ];

    // =========================================================================
    // Interfaces
    // =========================================================================
    axi4_stream_if #(
        .DATA_WIDTH(CONFIG_BUS_WIDTH)
    ) config_in (
        .aclk   (clk),
        .aresetn(!rst)
    );

    axi4_stream_if #(
        .DATA_WIDTH(INPUT_BUS_WIDTH)
    ) data_in (
        .aclk   (clk),
        .aresetn(!rst)
    );

    axi4_stream_if #(
        .DATA_WIDTH(OUTPUT_BUS_WIDTH)
    ) data_out (
        .aclk   (clk),
        .aresetn(!rst)
    );

    assign config_in.tstrb = config_in.tkeep;
    assign data_in.tstrb   = data_in.tkeep;

    // =========================================================================
    // DUT
    // =========================================================================
    generate
        if (!USE_CUSTOM_TOPOLOGY) begin : g_dut_trained
            bnn_fcc #(
                .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
                .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
                .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
                .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
                .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
                .TOTAL_LAYERS     (TRAINED_LAYERS),
                .TOPOLOGY         (TRAINED_TOPOLOGY),
                .PARALLEL_INPUTS  (PARALLEL_INPUTS),
                .PARALLEL_NEURONS (PARALLEL_NEURONS)
            ) DUT (
                .clk           (clk),
                .rst           (rst),
                .config_valid  (config_in.tvalid),
                .config_ready  (config_in.tready),
                .config_data   (config_in.tdata),
                .config_keep   (config_in.tkeep),
                .config_last   (config_in.tlast),
                .data_in_valid (data_in.tvalid),
                .data_in_ready (data_in.tready),
                .data_in_data  (data_in.tdata),
                .data_in_keep  (data_in.tkeep),
                .data_in_last  (data_in.tlast),
                .data_out_valid(data_out.tvalid),
                .data_out_ready(data_out.tready),
                .data_out_data (data_out.tdata),
                .data_out_keep (data_out.tkeep),
                .data_out_last (data_out.tlast)
            );
        end else if (TOPOLOGY_MODE == 0) begin : g_dut_mode0
            bnn_fcc #(
                .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
                .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
                .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
                .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
                .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
                .TOTAL_LAYERS     (MODE0_LAYERS),
                .TOPOLOGY         (MODE0_TOPOLOGY),
                .PARALLEL_INPUTS  (PARALLEL_INPUTS),
                .PARALLEL_NEURONS (PARALLEL_NEURONS)
            ) DUT (
                .clk           (clk),
                .rst           (rst),
                .config_valid  (config_in.tvalid),
                .config_ready  (config_in.tready),
                .config_data   (config_in.tdata),
                .config_keep   (config_in.tkeep),
                .config_last   (config_in.tlast),
                .data_in_valid (data_in.tvalid),
                .data_in_ready (data_in.tready),
                .data_in_data  (data_in.tdata),
                .data_in_keep  (data_in.tkeep),
                .data_in_last  (data_in.tlast),
                .data_out_valid(data_out.tvalid),
                .data_out_ready(data_out.tready),
                .data_out_data (data_out.tdata),
                .data_out_keep (data_out.tkeep),
                .data_out_last (data_out.tlast)
            );
        end else if (TOPOLOGY_MODE == 1) begin : g_dut_mode1
            bnn_fcc #(
                .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
                .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
                .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
                .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
                .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
                .TOTAL_LAYERS     (MODE1_LAYERS),
                .TOPOLOGY         (MODE1_TOPOLOGY),
                .PARALLEL_INPUTS  (PARALLEL_INPUTS),
                .PARALLEL_NEURONS (PARALLEL_NEURONS)
            ) DUT (
                .clk           (clk),
                .rst           (rst),
                .config_valid  (config_in.tvalid),
                .config_ready  (config_in.tready),
                .config_data   (config_in.tdata),
                .config_keep   (config_in.tkeep),
                .config_last   (config_in.tlast),
                .data_in_valid (data_in.tvalid),
                .data_in_ready (data_in.tready),
                .data_in_data  (data_in.tdata),
                .data_in_keep  (data_in.tkeep),
                .data_in_last  (data_in.tlast),
                .data_out_valid(data_out.tvalid),
                .data_out_ready(data_out.tready),
                .data_out_data (data_out.tdata),
                .data_out_keep (data_out.tkeep),
                .data_out_last (data_out.tlast)
            );
        end else if (TOPOLOGY_MODE == 2) begin : g_dut_mode2
            bnn_fcc #(
                .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
                .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
                .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
                .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
                .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
                .TOTAL_LAYERS     (MODE2_LAYERS),
                .TOPOLOGY         (MODE2_TOPOLOGY),
                .PARALLEL_INPUTS  (PARALLEL_INPUTS),
                .PARALLEL_NEURONS (PARALLEL_NEURONS)
            ) DUT (
                .clk           (clk),
                .rst           (rst),
                .config_valid  (config_in.tvalid),
                .config_ready  (config_in.tready),
                .config_data   (config_in.tdata),
                .config_keep   (config_in.tkeep),
                .config_last   (config_in.tlast),
                .data_in_valid (data_in.tvalid),
                .data_in_ready (data_in.tready),
                .data_in_data  (data_in.tdata),
                .data_in_keep  (data_in.tkeep),
                .data_in_last  (data_in.tlast),
                .data_out_valid(data_out.tvalid),
                .data_out_ready(data_out.tready),
                .data_out_data (data_out.tdata),
                .data_out_keep (data_out.tkeep),
                .data_out_last (data_out.tlast)
            );
        end else if (TOPOLOGY_MODE == 3) begin : g_dut_mode3
            bnn_fcc #(
                .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
                .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
                .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
                .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
                .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
                .TOTAL_LAYERS     (MODE3_LAYERS),
                .TOPOLOGY         (MODE3_TOPOLOGY),
                .PARALLEL_INPUTS  (PARALLEL_INPUTS),
                .PARALLEL_NEURONS (PARALLEL_NEURONS)
            ) DUT (
                .clk           (clk),
                .rst           (rst),
                .config_valid  (config_in.tvalid),
                .config_ready  (config_in.tready),
                .config_data   (config_in.tdata),
                .config_keep   (config_in.tkeep),
                .config_last   (config_in.tlast),
                .data_in_valid (data_in.tvalid),
                .data_in_ready (data_in.tready),
                .data_in_data  (data_in.tdata),
                .data_in_keep  (data_in.tkeep),
                .data_in_last  (data_in.tlast),
                .data_out_valid(data_out.tvalid),
                .data_out_ready(data_out.tready),
                .data_out_data (data_out.tdata),
                .data_out_keep (data_out.tkeep),
                .data_out_last (data_out.tlast)
            );
        end else begin : g_dut_custom
            bnn_fcc #(
                .INPUT_DATA_WIDTH (INPUT_DATA_WIDTH),
                .INPUT_BUS_WIDTH  (INPUT_BUS_WIDTH),
                .CONFIG_BUS_WIDTH (CONFIG_BUS_WIDTH),
                .OUTPUT_DATA_WIDTH(OUTPUT_DATA_WIDTH),
                .OUTPUT_BUS_WIDTH (OUTPUT_BUS_WIDTH),
                .TOTAL_LAYERS     (CUSTOM_LAYERS),
                .TOPOLOGY         (CUSTOM_TOPOLOGY),
                .PARALLEL_INPUTS  (PARALLEL_INPUTS),
                .PARALLEL_NEURONS (PARALLEL_NEURONS)
            ) DUT (
                .clk           (clk),
                .rst           (rst),
                .config_valid  (config_in.tvalid),
                .config_ready  (config_in.tready),
                .config_data   (config_in.tdata),
                .config_keep   (config_in.tkeep),
                .config_last   (config_in.tlast),
                .data_in_valid (data_in.tvalid),
                .data_in_ready (data_in.tready),
                .data_in_data  (data_in.tdata),
                .data_in_keep  (data_in.tkeep),
                .data_in_last  (data_in.tlast),
                .data_out_valid(data_out.tvalid),
                .data_out_ready(data_out.tready),
                .data_out_data (data_out.tdata),
                .data_out_keep (data_out.tkeep),
                .data_out_last (data_out.tlast)
            );
        end
    endgenerate

    // =========================================================================
    // Coverage
    // =========================================================================
    covergroup cg_config_message with function sample (
        int msg_type, int layer_idx, int keep_ones, int gap_class, int order_kind, int config_kind
    );
        option.per_instance = 1;

        cp_msg_type: coverpoint msg_type {bins weights = {0}; bins thresholds = {1};}

        cp_layer: coverpoint layer_idx {bins legal_layers[] = {[0 : ACTUAL_TOTAL_LAYERS - 2]};}

        cp_keep: coverpoint keep_ones {
            bins full = {CONFIG_KEEP_WIDTH}; bins partial = {[1 : CONFIG_KEEP_WIDTH - 1]};
        }

        cp_gap: coverpoint gap_class {bins none = {0}; bins one = {1}; bins few = {2}; bins burst = {3};}

        cp_order: coverpoint order_kind {
            bins standard = {CFG_ORDER_STANDARD};
            bins reverse_layer = {CFG_ORDER_REVERSE_LAYER};
            bins weights_then_thr = {CFG_ORDER_WEIGHTS_THEN_THRESHOLDS};
            bins thr_then_weights = {CFG_ORDER_THRESHOLDS_THEN_WEIGHTS};
        }

        cp_config_kind: coverpoint config_kind {
            bins full_cfg = {CONFIG_KIND_FULL};
            bins partial_weights = {CONFIG_KIND_PARTIAL_WEIGHTS};
            bins partial_thresh = {CONFIG_KIND_PARTIAL_THRESHOLDS};
        }

        cross cp_msg_type, cp_layer;
        cross cp_msg_type, cp_gap;
        cross cp_order, cp_config_kind;
    endgroup

    covergroup cg_input_beat with function sample (int keep_ones, bit last, int gap_class);
        option.per_instance = 1;

        cp_keep: coverpoint keep_ones {
            bins full = {INPUT_BUS_WIDTH / 8}; bins partial = {[1 : INPUT_BUS_WIDTH / 8 - 1]};
        }

        cp_last: coverpoint last {bins not_last = {0}; bins last_beat = {1};}

        cp_gap: coverpoint gap_class {bins none = {0}; bins one = {1}; bins few = {2}; bins burst = {3};}

        cross cp_keep, cp_last;
    endgroup

    covergroup cg_pixel_values with function sample (int pixel_class);
        option.per_instance = 1;

        cp_pixel: coverpoint pixel_class {
            bins zero = {0};
            bins edge_127 = {1};
            bins edge_128 = {2};
            bins max = {3};
            bins below_th = {4};
            bins above_th = {5};
        }
    endgroup

    covergroup cg_output_handshake with function sample (
        int output_class, int wait_class, int ready_mode, bit repeated_class
    );
        option.per_instance = 1;

        cp_class: coverpoint output_class {bins class_bins[] = {[0 : OUTPUT_CLASSES - 1]};}

        cp_wait: coverpoint wait_class {
            bins wait_immediate = {0}; bins wait_short = {1}; bins wait_mid = {2}; bins wait_long = {3};
        }

        cp_ready_mode: coverpoint ready_mode {
            bins mode_always = {READY_ALWAYS};
            bins mode_random = {READY_RANDOM};
            bins mode_bursty = {READY_BURSTY};
            bins mode_delayed = {READY_DELAYED};
        }

        cp_repeat: coverpoint repeated_class {bins class_changed = {0}; bins class_repeated = {1};}

        cross cp_wait, cp_ready_mode;
        cross cp_class, cp_repeat;
    endgroup

    covergroup cg_output_ready_alignment with function sample (
        bit ready_before, bit ready_same_cycle, bit ready_after
    );
        option.per_instance = 1;

        cp_before: coverpoint ready_before {bins low = {0}; bins high = {1};}

        cp_same: coverpoint ready_same_cycle {bins low = {0}; bins high = {1};}

        cp_after: coverpoint ready_after {bins low = {0}; bins high = {1};}
    endgroup

    covergroup cg_reset with function sample (int phase, bit on_boundary);
        option.per_instance = 1;

        cp_phase: coverpoint phase {
            bins phase_before_config = {RESET_BEFORE_CONFIG};
            bins phase_during_config = {RESET_DURING_CONFIG};
            bins phase_between_cfg_img = {RESET_BETWEEN_CONFIG_AND_IMAGE};
            bins phase_during_image = {RESET_DURING_IMAGE};
            bins phase_during_output = {RESET_DURING_OUTPUT};
        }

        cp_boundary: coverpoint on_boundary {bins boundary_off = {0}; bins boundary_on = {1};}

        cross cp_phase, cp_boundary;
    endgroup

    covergroup cg_reconfig with function sample (
        int layer_idx, int config_kind, int threshold_mode, int weight_mode
    );
        option.per_instance = 1;

        cp_layer: coverpoint layer_idx {bins legal_layers[] = {[0 : ACTUAL_TOTAL_LAYERS - 2]};}

        cp_kind: coverpoint config_kind {
            bins full_cfg = {CONFIG_KIND_FULL};
            bins partial_weights = {CONFIG_KIND_PARTIAL_WEIGHTS};
            bins partial_thresh = {CONFIG_KIND_PARTIAL_THRESHOLDS};
        }

        cp_thresh_mode: coverpoint threshold_mode {
            bins thresh_zero = {THRESHOLD_ZERO};
            bins thresh_max = {THRESHOLD_MAX};
            bins thresh_mid = {THRESHOLD_MID};
            bins thresh_random = {THRESHOLD_RANDOM};
        }

        cp_weight_mode: coverpoint weight_mode {
            bins weight_all_zero = {WEIGHT_ALL_ZERO};
            bins weight_all_one = {WEIGHT_ALL_ONE};
            bins weight_checker = {WEIGHT_CHECKER};
            bins weight_randomized = {WEIGHT_RANDOMIZED};
        }
    endgroup

    covergroup cg_inter_image_gap with function sample (int gap_class);
        option.per_instance = 1;

        cp_gap: coverpoint gap_class {bins none = {0}; bins one = {1}; bins few = {2}; bins long = {3};}
    endgroup

    cg_config_message         config_cov;
    cg_input_beat             input_cov;
    cg_pixel_values           pixel_cov;
    cg_output_handshake       output_cov;
    cg_output_ready_alignment output_align_cov;
    cg_reset                  reset_cov;
    cg_reconfig               reconfig_cov;
    cg_inter_image_gap        inter_image_cov;

    // =========================================================================
    // Assertions
    // =========================================================================
    assert property (@(posedge clk) disable iff (rst) data_out.tvalid |-> data_out.tlast)
    else $error("data_out.tlast must be asserted with every output beat.");

    // =========================================================================
    // Clock generation
    // =========================================================================
    initial begin : generate_clock
        forever #HALF_CLK_PERIOD clk <= ~clk;
    end

    // =========================================================================
    // Reference-model verification
    // =========================================================================
    task verify_model();
        int python_preds[];
        bit [INPUT_DATA_WIDTH-1:0] current_img[];
        string input_path;
        string output_path;

        input_path  = $sformatf("%s/%s", BASE_DIR, MNIST_TEST_VECTOR_INPUT_PATH);
        output_path = $sformatf("%s/%s", BASE_DIR, MNIST_TEST_VECTOR_OUTPUT_PATH);

        stim.load_from_file(input_path);
        num_tests = stim.get_num_vectors();

        python_preds = new[num_tests];
        $readmemh(output_path, python_preds);

        for (int i = 0; i < num_tests; i++) begin
            int sv_pred;
            stim.get_vector(i, current_img);
            sv_pred = model.compute_reference(current_img);

            if (sv_pred !== python_preds[i]) begin
                $error("TB LOGIC ERROR: Img %0d. SV Model says %0d, Python says %0d", i, sv_pred,
                       python_preds[i]);
                $finish;
            end
        end

        $display("SV model successfully verified.");
    endtask

    // =========================================================================
    // Initialization
    // =========================================================================
    initial begin : l_init_model
        string path;

        model                  = new();
        stim                   = new(topology_at(0));
        cover_stim             = new(topology_at(0));
        latency                = new(CLK_PERIOD);
        throughput             = new(CLK_PERIOD);

        config_cov             = new();
        input_cov              = new();
        pixel_cov              = new();
        output_cov             = new();
        output_align_cov       = new();
        reset_cov              = new();
        reconfig_cov           = new();
        inter_image_cov        = new();

        last_config_layer_r    = -1;
        last_config_msg_type_r = -1;
        last_output_class_r    = -1;

        actual_topology_q      = new[ACTUAL_TOTAL_LAYERS];
        for (int i = 0; i < ACTUAL_TOTAL_LAYERS; i++) begin
            actual_topology_q[i] = topology_at(i);
        end

        $display("TOPOLOGY_MODE = %0d", TOPOLOGY_MODE);
        $display("USE_CUSTOM_TOPOLOGY = %0d", USE_CUSTOM_TOPOLOGY);

        ready_mode_r                = READY_ALWAYS;
        current_cfg_order_r         = CFG_ORDER_STANDARD;
        current_config_kind_r       = CONFIG_KIND_FULL;
        current_config_valid_mode_r = VALID_CONTINUOUS;
        current_input_valid_mode_r  = VALID_CONTINUOUS;

        if (!USE_CUSTOM_TOPOLOGY) begin
            $display("--- Loading Trained Model ---");
            path = $sformatf("%s/%s", BASE_DIR, MNIST_MODEL_DATA_PATH);
            model.load_from_file(path, actual_topology_q);

            if (VERIFY_MODEL) verify_model();

            model.encode_configuration(config_bus_data_stream, config_bus_keep_stream);
            $display("--- Configuration created: %0d words (%0d-bit wide) ---",
                     config_bus_data_stream.size(), CONFIG_BUS_WIDTH);

            $display("--- Loading Test Vectors ---");
            path = $sformatf("%s/%s", BASE_DIR, MNIST_TEST_VECTOR_INPUT_PATH);
            stim.load_from_file(path, NUM_TEST_IMAGES);
            cover_stim.load_from_file(path);
        end else begin
            $display("--- Loading Randomized Model ---");
            model.create_random(actual_topology_q);
            model.encode_configuration(config_bus_data_stream, config_bus_keep_stream);
            $display("--- Configuration created: %0d words (%0d-bit wide) ---",
                     config_bus_data_stream.size(), CONFIG_BUS_WIDTH);

            $display("--- Generating Random Test Vectors ---");
            stim.generate_random_vectors(NUM_TEST_IMAGES);
            cover_stim.generate_random_vectors(NUM_TEST_IMAGES * 2);
        end

        num_tests = stim.get_num_vectors();
        model.print_summary();
        $write("Active topology: ");
        for (int i = 0; i < ACTUAL_TOTAL_LAYERS; i++) begin
            $write("%0d", actual_topology_q[i]);
            if (i != ACTUAL_TOTAL_LAYERS - 1) $write(" -> ");
        end
        $write("\n");
        if (DEBUG) model.print_model();
    end

    // =========================================================================
    // Utility tasks
    // =========================================================================
    task automatic clear_all_drivers();
        config_in.tvalid <= 1'b0;
        config_in.tdata  <= '0;
        config_in.tkeep  <= '0;
        config_in.tlast  <= 1'b0;
        config_in.tuser  <= '0;
        config_in.tid    <= '0;
        config_in.tdest  <= '0;

        data_in.tvalid   <= 1'b0;
        data_in.tdata    <= '0;
        data_in.tkeep    <= '0;
        data_in.tlast    <= 1'b0;
        data_in.tuser    <= '0;
        data_in.tid      <= '0;
        data_in.tdest    <= '0;
    endtask

    task automatic flush_scoreboard();
        expected_outputs.delete();
        scoreboard_epoch++;
    endtask

    task automatic do_reset(input reset_phase_e phase, input bit on_boundary = 0, input int cycles = 5);
        last_reset_phase_r = phase;
        total_reset_count++;
        reset_cov.sample(phase, on_boundary);

        clear_all_drivers();
        rst <= 1'b1;
        flush_scoreboard();
        issued_image_id    = 0;
        completed_image_id = 0;
        latency    = new(CLK_PERIOD);
        throughput = new(CLK_PERIOD);

        repeat (cycles) @(posedge clk);
        @(negedge clk);
        rst <= 1'b0;
        repeat (cycles) @(posedge clk);
    endtask

    task automatic insert_valid_gap(input valid_mode_e mode, input bit is_config);
        int gap_cycles;

        case (mode)
            VALID_CONTINUOUS: gap_cycles = 0;
            VALID_INTERMITTENT: begin
                gap_cycles = 0;
                while (!chance(
                    is_config ? CONFIG_VALID_PROBABILITY : DATA_IN_VALID_PROBABILITY
                ))
                gap_cycles++;
            end
            VALID_BURSTY: begin
                if (chance(0.70)) gap_cycles = 0;
                else gap_cycles = $urandom_range(1, 5);
            end
            default: gap_cycles = 0;
        endcase

        if (is_config) config_gap_cycles_r = gap_cycles;
        else data_gap_cycles_r = gap_cycles;

        repeat (gap_cycles) begin
            if (is_config) config_in.tvalid <= 1'b0;
            else data_in.tvalid <= 1'b0;

            @(posedge clk);
        end
    endtask

    task automatic sample_pixels(input bit [INPUT_DATA_WIDTH-1:0] img_data[]);
        foreach (img_data[i]) begin
            pixel_cov.sample(classify_pixel(img_data[i]));
        end
    endtask

    task automatic make_all_zero_vector(output bit [INPUT_DATA_WIDTH-1:0] vec[]);
        vec = new[topology_at(0)];
        foreach (vec[i]) vec[i] = 8'd0;
    endtask

    task automatic make_all_one_vector(output bit [INPUT_DATA_WIDTH-1:0] vec[]);
        vec = new[topology_at(0)];
        foreach (vec[i]) vec[i] = 8'd255;
    endtask

    task automatic make_checkerboard_vector(output bit [INPUT_DATA_WIDTH-1:0] vec[]);
        vec = new[topology_at(0)];
        foreach (vec[i]) vec[i] = (i % 2 == 0) ? 8'd0 : 8'd255;
    endtask

    task automatic make_threshold_edge_vector(output bit [INPUT_DATA_WIDTH-1:0] vec[]);
        vec = new[topology_at(0)];
        foreach (vec[i]) begin
            case (i % 4)
                0: vec[i] = 8'd127;
                1: vec[i] = 8'd128;
                2: vec[i] = 8'd0;
                default: vec[i] = 8'd255;
            endcase
        end
    endtask

    task automatic find_class_cover_indices(output int class_cover_indices[$]);
        bit class_seen[256];
        int class_seen_count = 0;
        bit [INPUT_DATA_WIDTH-1:0] img[];

        class_cover_indices.delete();
        foreach (class_seen[i]) class_seen[i] = 1'b0;

        for (int i = 0; i < cover_stim.get_num_vectors(); i++) begin
            int pred;

            if (class_seen_count == OUTPUT_CLASSES) break;

            cover_stim.get_vector(i, img);
            pred = model.compute_reference(img);

            if (pred >= 0 && pred < OUTPUT_CLASSES && !class_seen[pred]) begin
                class_seen[pred] = 1'b1;
                class_seen_count++;
                class_cover_indices.push_back(i);
            end
        end

        last_class_cover_count_r = class_cover_indices.size();
    endtask

    task automatic apply_threshold_pattern(input int layer_idx, input threshold_mode_e mode);
        int fan_in = topology_at(layer_idx);
        int n_neurons = topology_at(layer_idx + 1);

        if (layer_idx >= ACTUAL_TOTAL_LAYERS - 1) return;

        for (int n = 0; n < n_neurons; n++) begin
            case (mode)
                THRESHOLD_ZERO:   model.threshold[layer_idx][n] = 0;
                THRESHOLD_MAX:    model.threshold[layer_idx][n] = fan_in;
                THRESHOLD_MID:    model.threshold[layer_idx][n] = fan_in / 2;
                THRESHOLD_RANDOM: model.threshold[layer_idx][n] = $urandom_range(0, fan_in);
                default:          model.threshold[layer_idx][n] = model.threshold[layer_idx][n];
            endcase
        end

        reconfig_cov.sample(layer_idx, CONFIG_KIND_PARTIAL_THRESHOLDS, mode, WEIGHT_RANDOMIZED);
    endtask

    task automatic apply_weight_pattern(input int layer_idx, input weight_mode_e mode);
        int fan_in = topology_at(layer_idx);
        int n_neurons = topology_at(layer_idx + 1);

        for (int n = 0; n < n_neurons; n++) begin
            for (int i = 0; i < fan_in; i++) begin
                case (mode)
                    WEIGHT_ALL_ZERO:   model.weight[layer_idx][n][i] = 1'b0;
                    WEIGHT_ALL_ONE:    model.weight[layer_idx][n][i] = 1'b1;
                    WEIGHT_CHECKER:    model.weight[layer_idx][n][i] = ((n + i) % 2);
                    WEIGHT_RANDOMIZED: model.weight[layer_idx][n][i] = $urandom_range(0, 1);
                    default:           model.weight[layer_idx][n][i] = model.weight[layer_idx][n][i];
                endcase
            end
        end

        reconfig_cov.sample(layer_idx, CONFIG_KIND_PARTIAL_WEIGHTS, THRESHOLD_RANDOM, mode);
    endtask

    task automatic drive_config_message(input int layer_idx, input bit is_threshold,
                                        input valid_mode_e valid_mode);
        config_bus_data_stream_t layer_stream;
        config_keep_stream_t     layer_keep;

        current_config_valid_mode_r = valid_mode;
        model.get_layer_config(layer_idx, is_threshold, layer_stream, layer_keep);

        last_config_layer_r    = layer_idx;
        last_config_msg_type_r = is_threshold;

        for (int i = 0; i < layer_stream.size(); i++) begin
            insert_valid_gap(valid_mode, 1'b1);

            config_in.tvalid <= 1'b1;
            config_in.tdata  <= layer_stream[i];
            config_in.tkeep  <= layer_keep[i];
            config_in.tlast  <= (i == layer_stream.size() - 1);

            @(posedge clk iff config_in.tready);

            if (i == 0) begin
                config_cov.sample(is_threshold, layer_idx, keep_count(layer_keep[layer_keep.size()-1]),
                                  gap_bucket(config_gap_cycles_r), current_cfg_order_r,
                                  current_config_kind_r);
            end
        end

        config_in.tvalid <= 1'b0;
        config_in.tlast  <= 1'b0;
        config_in.tkeep  <= '0;
        @(posedge clk);
    endtask

    task automatic drive_full_configuration(input cfg_order_e order_kind, input valid_mode_e valid_mode);
        current_cfg_order_r   = order_kind;
        current_config_kind_r = CONFIG_KIND_FULL;

        for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 1; l++) begin
            reconfig_cov.sample(l, CONFIG_KIND_FULL, THRESHOLD_RANDOM, WEIGHT_RANDOMIZED);
        end

        case (order_kind)
            CFG_ORDER_STANDARD: begin
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 1; l++) begin
                    drive_config_message(l, 1'b0, valid_mode);
                    if (l < ACTUAL_TOTAL_LAYERS - 2) drive_config_message(l, 1'b1, valid_mode);
                end
            end

            CFG_ORDER_REVERSE_LAYER: begin
                for (int l = ACTUAL_TOTAL_LAYERS - 2; l >= 0; l--) begin
                    if (l < ACTUAL_TOTAL_LAYERS - 2) drive_config_message(l, 1'b1, valid_mode);
                    drive_config_message(l, 1'b0, valid_mode);
                end
            end

            CFG_ORDER_WEIGHTS_THEN_THRESHOLDS: begin
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 1; l++) begin
                    drive_config_message(l, 1'b0, valid_mode);
                end
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 2; l++) begin
                    drive_config_message(l, 1'b1, valid_mode);
                end
            end

            CFG_ORDER_THRESHOLDS_THEN_WEIGHTS: begin
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 2; l++) begin
                    drive_config_message(l, 1'b1, valid_mode);
                end
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 1; l++) begin
                    drive_config_message(l, 1'b0, valid_mode);
                end
            end

            default: begin
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 1; l++) begin
                    drive_config_message(l, 1'b0, valid_mode);
                    if (l < ACTUAL_TOTAL_LAYERS - 2) drive_config_message(l, 1'b1, valid_mode);
                end
            end
        endcase
    endtask

    task automatic drive_partial_threshold_reconfig(input int layer_idx, input threshold_mode_e thresh_mode,
                                                    input valid_mode_e valid_mode);
        current_config_kind_r = CONFIG_KIND_PARTIAL_THRESHOLDS;
        apply_threshold_pattern(layer_idx, thresh_mode);
        drive_config_message(layer_idx, 1'b1, valid_mode);
    endtask

    task automatic drive_partial_weight_reconfig(input int layer_idx, input weight_mode_e weight_mode,
                                                 input valid_mode_e valid_mode);
        current_config_kind_r = CONFIG_KIND_PARTIAL_WEIGHTS;
        apply_weight_pattern(layer_idx, weight_mode);
        drive_config_message(layer_idx, 1'b0, valid_mode);
    endtask

    task automatic drive_image_vector(input bit [INPUT_DATA_WIDTH-1:0] img_data[],
                                      input valid_mode_e valid_mode, input int gap_after_image = 0,
                                      input bit track_output = 1'b1);
        int expected_pred;

        sample_pixels(img_data);

        if (track_output) begin
            expected_pred = model.compute_reference(img_data);
            expected_outputs.push_back(expected_pred[OUTPUT_DATA_WIDTH-1:0]);
            total_expected_outputs++;

            if (total_expected_outputs == 1) throughput.start_test();
            latency.start_event(issued_image_id);
            issued_image_id++;
        end

        for (int j = 0; j < img_data.size(); j += INPUTS_PER_CYCLE) begin
            insert_valid_gap(valid_mode, 1'b0);

            for (int k = 0; k < INPUTS_PER_CYCLE; k++) begin
                if (j + k < img_data.size()) begin
                    data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= img_data[j+k];
                    data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '1;
                end else begin
                    data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= '0;
                    data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '0;
                end
            end

            data_in.tvalid <= 1'b1;
            data_in.tlast  <= (j + INPUTS_PER_CYCLE >= img_data.size());

            @(posedge clk iff data_in.tready);

            input_cov.sample(input_keep_count(data_in.tkeep), data_in.tlast, gap_bucket(data_gap_cycles_r));
        end

        data_in.tvalid <= 1'b0;
        data_in.tlast  <= 1'b0;
        data_in.tkeep  <= '0;

        inter_image_gap_r = gap_after_image;
        inter_image_cov.sample(gap_bucket(inter_image_gap_r));

        repeat (gap_after_image) @(posedge clk);
    endtask

    task automatic drive_image_then_reset_midstream(input bit [INPUT_DATA_WIDTH-1:0] img_data[],
                                                    input valid_mode_e valid_mode);
        int beats_before_reset;

        sample_pixels(img_data);

        beats_before_reset = (img_data.size() / INPUTS_PER_CYCLE > 1) ?
            $urandom_range(1, (img_data.size() / INPUTS_PER_CYCLE)) : 1;

        for (int j = 0; j < img_data.size(); j += INPUTS_PER_CYCLE) begin
            insert_valid_gap(valid_mode, 1'b0);

            for (int k = 0; k < INPUTS_PER_CYCLE; k++) begin
                if (j + k < img_data.size()) begin
                    data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= img_data[j+k];
                    data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '1;
                end else begin
                    data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= '0;
                    data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '0;
                end
            end

            data_in.tvalid <= 1'b1;
            data_in.tlast  <= (j + INPUTS_PER_CYCLE >= img_data.size());
            @(posedge clk iff data_in.tready);

            input_cov.sample(input_keep_count(data_in.tkeep), data_in.tlast, gap_bucket(data_gap_cycles_r));

            beats_before_reset--;
            if (beats_before_reset == 0) begin
                do_reset(RESET_DURING_IMAGE, data_in.tlast);
                return;
            end
        end

        data_in.tvalid <= 1'b0;
        data_in.tlast  <= 1'b0;
        data_in.tkeep  <= '0;
    endtask

    task automatic wait_until_input_ready();
        wait (data_in.tready);
        repeat (3) @(posedge clk);
    endtask

    task automatic send_random_images(input int count, input valid_mode_e valid_mode, input int max_gap = 3);
        bit [INPUT_DATA_WIDTH-1:0] img[];
        for (int i = 0; i < count; i++) begin
            if (USE_CUSTOM_TOPOLOGY) stim.get_random_vector(img);
            else begin
                int idx = (num_tests > 0) ? (i % num_tests) : 0;
                stim.get_vector(idx, img);
            end
            drive_image_vector(img, valid_mode, $urandom_range(0, max_gap), 1'b1);
        end
    endtask

    task automatic send_class_cover_sequence(input valid_mode_e valid_mode);
        int class_cover_indices[$];
        bit [INPUT_DATA_WIDTH-1:0] img[];

        find_class_cover_indices(class_cover_indices);

        for (int i = 0; i < class_cover_indices.size(); i++) begin
            cover_stim.get_vector(class_cover_indices[i], img);
            drive_image_vector(img, valid_mode, i % 3, 1'b1);
        end
    endtask

    task automatic send_repeated_class_burst(input valid_mode_e valid_mode);
        int class_cover_indices[$];
        bit [INPUT_DATA_WIDTH-1:0] img[];

        find_class_cover_indices(class_cover_indices);

        if (class_cover_indices.size() > 0) begin
            cover_stim.get_vector(class_cover_indices[0], img);
            for (int i = 0; i < REPEATED_CLASS_BURST_LEN; i++) begin
                drive_image_vector(img, valid_mode, 0, 1'b1);
            end
        end
    endtask

    task automatic send_directed_image_suite(input valid_mode_e valid_mode);
        bit [INPUT_DATA_WIDTH-1:0] img[];

        make_all_zero_vector(img);
        drive_image_vector(img, valid_mode, 1, 1'b1);

        make_all_one_vector(img);
        drive_image_vector(img, valid_mode, 1, 1'b1);

        make_checkerboard_vector(img);
        drive_image_vector(img, valid_mode, 2, 1'b1);

        make_threshold_edge_vector(img);
        drive_image_vector(img, valid_mode, 2, 1'b1);
    endtask

    task automatic do_mid_config_reset();
        config_bus_data_stream_t layer_stream;
        config_keep_stream_t     layer_keep;

        current_cfg_order_r   = CFG_ORDER_STANDARD;
        current_config_kind_r = CONFIG_KIND_FULL;
        model.get_layer_config(0, 1'b0, layer_stream, layer_keep);

        for (int i = 0; i < layer_stream.size(); i++) begin
            config_in.tvalid <= 1'b1;
            config_in.tdata  <= layer_stream[i];
            config_in.tkeep  <= layer_keep[i];
            config_in.tlast  <= (i == layer_stream.size() - 1);

            @(posedge clk iff config_in.tready);

            if (i == 0) begin
                config_cov.sample(0, 0, keep_count(layer_keep[layer_keep.size()-1]), 0, current_cfg_order_r,
                                  current_config_kind_r);
            end

            if (i == 0) begin
                do_reset(RESET_DURING_CONFIG, 1'b0);
                return;
            end
        end
    endtask

    task automatic do_output_phase_reset_scenario();
        bit [INPUT_DATA_WIDTH-1:0] img[];

        make_checkerboard_vector(img);

        ready_mode_r <= READY_DELAYED;
        drive_image_vector(img, VALID_CONTINUOUS, 0, 1'b1);

        repeat (2) @(posedge clk);
        do_reset(RESET_DURING_OUTPUT, 1'b0);
    endtask


    task automatic drive_config_message_fixed_gap(input int layer_idx, input bit is_threshold,
                                                  input int fixed_gap_cycles);
        config_bus_data_stream_t layer_stream;
        config_keep_stream_t     layer_keep;

        current_config_valid_mode_r = VALID_BURSTY;
        model.get_layer_config(layer_idx, is_threshold, layer_stream, layer_keep);

        last_config_layer_r    = layer_idx;
        last_config_msg_type_r = is_threshold;

        for (int i = 0; i < layer_stream.size(); i++) begin
            config_gap_cycles_r = fixed_gap_cycles;
            repeat (fixed_gap_cycles) begin
                config_in.tvalid <= 1'b0;
                @(posedge clk);
            end

            config_in.tvalid <= 1'b1;
            config_in.tdata  <= layer_stream[i];
            config_in.tkeep  <= layer_keep[i];
            config_in.tlast  <= (i == layer_stream.size() - 1);

            @(posedge clk iff config_in.tready);

            if (i == 0) begin
                config_cov.sample(is_threshold, layer_idx, keep_count(layer_keep[layer_keep.size()-1]),
                                  gap_bucket(config_gap_cycles_r), current_cfg_order_r,
                                  current_config_kind_r);
            end
        end

        config_in.tvalid <= 1'b0;
        config_in.tlast  <= 1'b0;
        config_in.tkeep  <= '0;
        @(posedge clk);
    endtask

    task automatic drive_full_configuration_fixed_gap(input cfg_order_e order_kind,
                                                      input int fixed_gap_cycles);
        current_cfg_order_r   = order_kind;
        current_config_kind_r = CONFIG_KIND_FULL;

        case (order_kind)
            CFG_ORDER_STANDARD: begin
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 1; l++) begin
                    drive_config_message_fixed_gap(l, 1'b0, fixed_gap_cycles);
                    if (l < ACTUAL_TOTAL_LAYERS - 2)
                        drive_config_message_fixed_gap(l, 1'b1, fixed_gap_cycles);
                end
            end

            CFG_ORDER_REVERSE_LAYER: begin
                for (int l = ACTUAL_TOTAL_LAYERS - 2; l >= 0; l--) begin
                    if (l < ACTUAL_TOTAL_LAYERS - 2)
                        drive_config_message_fixed_gap(l, 1'b1, fixed_gap_cycles);
                    drive_config_message_fixed_gap(l, 1'b0, fixed_gap_cycles);
                end
            end

            CFG_ORDER_WEIGHTS_THEN_THRESHOLDS: begin
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 1; l++) begin
                    drive_config_message_fixed_gap(l, 1'b0, fixed_gap_cycles);
                end
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 2; l++) begin
                    drive_config_message_fixed_gap(l, 1'b1, fixed_gap_cycles);
                end
            end

            CFG_ORDER_THRESHOLDS_THEN_WEIGHTS: begin
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 2; l++) begin
                    drive_config_message_fixed_gap(l, 1'b1, fixed_gap_cycles);
                end
                for (int l = 0; l < ACTUAL_TOTAL_LAYERS - 1; l++) begin
                    drive_config_message_fixed_gap(l, 1'b0, fixed_gap_cycles);
                end
            end

            default: begin
                drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            end
        endcase
    endtask

    task automatic send_inter_image_gap_sweep(input valid_mode_e valid_mode);
        bit [INPUT_DATA_WIDTH-1:0] img[];

        make_all_zero_vector(img);
        drive_image_vector(img, valid_mode, 0, 1'b1);

        make_all_one_vector(img);
        drive_image_vector(img, valid_mode, 1, 1'b1);

        make_checkerboard_vector(img);
        drive_image_vector(img, valid_mode, 3, 1'b1);

        make_threshold_edge_vector(img);
        drive_image_vector(img, valid_mode, 8, 1'b1);
    endtask

    task automatic drive_image_then_reset_on_boundary(input bit [INPUT_DATA_WIDTH-1:0] img_data[],
                                                      input valid_mode_e valid_mode);
        sample_pixels(img_data);

        for (int j = 0; j < img_data.size(); j += INPUTS_PER_CYCLE) begin
            insert_valid_gap(valid_mode, 1'b0);

            for (int k = 0; k < INPUTS_PER_CYCLE; k++) begin
                if (j + k < img_data.size()) begin
                    data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= img_data[j+k];
                    data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '1;
                end else begin
                    data_in.tdata[k*INPUT_DATA_WIDTH+:INPUT_DATA_WIDTH] <= '0;
                    data_in.tkeep[k*BYTES_PER_INPUT+:BYTES_PER_INPUT]   <= '0;
                end
            end

            data_in.tvalid <= 1'b1;
            data_in.tlast  <= (j + INPUTS_PER_CYCLE >= img_data.size());
            @(posedge clk iff data_in.tready);

            input_cov.sample(input_keep_count(data_in.tkeep), data_in.tlast, gap_bucket(data_gap_cycles_r));

            if (data_in.tlast) begin
                data_in.tvalid <= 1'b0;
                data_in.tlast  <= 1'b0;
                data_in.tkeep  <= '0;
                do_reset(RESET_DURING_IMAGE, 1'b1);
                return;
            end
        end

        data_in.tvalid <= 1'b0;
        data_in.tlast  <= 1'b0;
        data_in.tkeep  <= '0;
    endtask

    task automatic do_reset_on_config_boundary();
        config_bus_data_stream_t layer_stream;
        config_keep_stream_t     layer_keep;

        current_cfg_order_r   = CFG_ORDER_STANDARD;
        current_config_kind_r = CONFIG_KIND_FULL;
        model.get_layer_config(0, 1'b0, layer_stream, layer_keep);

        for (int i = 0; i < layer_stream.size(); i++) begin
            config_in.tvalid <= 1'b1;
            config_in.tdata  <= layer_stream[i];
            config_in.tkeep  <= layer_keep[i];
            config_in.tlast  <= (i == layer_stream.size() - 1);

            @(posedge clk iff config_in.tready);

            if (i == 0) begin
                config_cov.sample(0, 0, keep_count(layer_keep[layer_keep.size()-1]), 0, current_cfg_order_r,
                                  current_config_kind_r);
            end

            if (i == layer_stream.size() - 1) begin
                config_in.tvalid <= 1'b0;
                config_in.tlast  <= 1'b0;
                config_in.tkeep  <= '0;
                do_reset(RESET_DURING_CONFIG, 1'b1);
                return;
            end
        end
    endtask

    task automatic drive_partial_reconfig_sweep(input valid_mode_e valid_mode);
        if (ACTUAL_TOTAL_LAYERS > 1) begin
            drive_partial_threshold_reconfig(0, THRESHOLD_ZERO, valid_mode);
            drive_partial_threshold_reconfig(0, THRESHOLD_MAX, valid_mode);
            drive_partial_threshold_reconfig(0, THRESHOLD_MID, valid_mode);
            drive_partial_threshold_reconfig(0, THRESHOLD_RANDOM, valid_mode);

            drive_partial_weight_reconfig(0, WEIGHT_ALL_ZERO, valid_mode);
            drive_partial_weight_reconfig(0, WEIGHT_ALL_ONE, valid_mode);
            drive_partial_weight_reconfig(0, WEIGHT_CHECKER, valid_mode);
            drive_partial_weight_reconfig(0, WEIGHT_RANDOMIZED, valid_mode);
        end

        if (ACTUAL_TOTAL_LAYERS > 2) begin
            drive_partial_threshold_reconfig(1, THRESHOLD_ZERO, valid_mode);
            drive_partial_threshold_reconfig(1, THRESHOLD_MAX, valid_mode);
            drive_partial_threshold_reconfig(1, THRESHOLD_MID, valid_mode);
            drive_partial_threshold_reconfig(1, THRESHOLD_RANDOM, valid_mode);

            drive_partial_weight_reconfig(1, WEIGHT_ALL_ZERO, valid_mode);
            drive_partial_weight_reconfig(1, WEIGHT_ALL_ONE, valid_mode);
            drive_partial_weight_reconfig(1, WEIGHT_CHECKER, valid_mode);
            drive_partial_weight_reconfig(1, WEIGHT_RANDOMIZED, valid_mode);
        end
    endtask

    // =========================================================================
    // Driver / sequencer
    // =========================================================================
    initial begin : l_sequencer_and_driver
        $timeformat(-9, 0, " ns", 0);

        passed                 = 0;
        failed                 = 0;
        total_expected_outputs = 0;
        total_checked_outputs  = 0;
        total_reset_count      = 0;
        issued_image_id        = 0;
        completed_image_id     = 0;
        scoreboard_epoch       = 0;

        rst <= 1'b1;
        clear_all_drivers();

        repeat (5) @(posedge clk);
        @(negedge clk);
        rst <= 1'b0;
        repeat (5) @(posedge clk);

        // ---------------------------------------------------------------------
        // Scenario 1: Baseline full configuration, continuous traffic, directed
        // output-class coverage and directed image diversity.
        // ---------------------------------------------------------------------
        $display("[%0t] Scenario 1: baseline full configuration.", $realtime);
        ready_mode_r <= READY_ALWAYS;
        drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
        wait_until_input_ready();
        send_class_cover_sequence(VALID_CONTINUOUS);
        send_directed_image_suite(VALID_CONTINUOUS);
        send_repeated_class_burst(VALID_CONTINUOUS);

        // ---------------------------------------------------------------------
        // Scenario 2: Bursty config/data traffic and bursty output backpressure.
        // ---------------------------------------------------------------------
        $display("[%0t] Scenario 2: bursty protocol patterns.", $realtime);
        do_reset(RESET_BETWEEN_CONFIG_AND_IMAGE, 1'b0);
        ready_mode_r <= READY_BURSTY;
        drive_full_configuration(CFG_ORDER_STANDARD, VALID_BURSTY);
        wait_until_input_ready();
        send_random_images(EXTRA_RANDOM_IMAGES, VALID_INTERMITTENT, 4);

        // ---------------------------------------------------------------------
        // Scenario 3: Non-standard configuration ordering.
        // ---------------------------------------------------------------------
        if (ENABLE_NONSTANDARD_CONFIG_ORDER) begin
            $display("[%0t] Scenario 3: non-standard configuration ordering.", $realtime);

            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_RANDOM;
            drive_full_configuration(CFG_ORDER_WEIGHTS_THEN_THRESHOLDS, VALID_INTERMITTENT);
            wait_until_input_ready();
            send_directed_image_suite(VALID_INTERMITTENT);

            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_REVERSE_LAYER, VALID_CONTINUOUS);
            wait_until_input_ready();
            send_class_cover_sequence(VALID_BURSTY);

            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_DELAYED;
            drive_full_configuration(CFG_ORDER_THRESHOLDS_THEN_WEIGHTS, VALID_BURSTY);
            wait_until_input_ready();
            send_random_images(EXTRA_RANDOM_IMAGES / 2 + 1, VALID_BURSTY, 3);
        end

        // ---------------------------------------------------------------------
        // Scenario 4: Reset during configuration, then recover with full config.
        // ---------------------------------------------------------------------
        if (ENABLE_RESET_STRESS) begin
            $display("[%0t] Scenario 4: reset during configuration.", $realtime);
            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            do_mid_config_reset();
            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();
            send_directed_image_suite(VALID_CONTINUOUS);
        end

        // ---------------------------------------------------------------------
        // Scenario 5: Reset during image transfer, recover, and continue.
        // ---------------------------------------------------------------------
        if (ENABLE_RESET_STRESS) begin
            bit [INPUT_DATA_WIDTH-1:0] img[];

            $display("[%0t] Scenario 5: reset during image transfer.", $realtime);
            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_BURSTY;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_INTERMITTENT);
            wait_until_input_ready();

            make_threshold_edge_vector(img);
            drive_image_then_reset_midstream(img, VALID_INTERMITTENT);

            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();
            send_random_images(EXTRA_RANDOM_IMAGES / 2 + 2, VALID_CONTINUOUS, 2);
        end

        // ---------------------------------------------------------------------
        // Scenario 6: Partial threshold / weight reconfiguration.
        // ---------------------------------------------------------------------
        if (ENABLE_PARTIAL_RECONFIG && ACTUAL_TOTAL_LAYERS > 2) begin
            $display("[%0t] Scenario 6: partial reconfiguration.", $realtime);

            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();

            // Exercise partial-reconfiguration coverage without relying on
            // the DUT to support fully correct live partial updates.
            drive_partial_threshold_reconfig(0, THRESHOLD_ZERO, VALID_CONTINUOUS);
            drive_partial_threshold_reconfig(0, THRESHOLD_MAX, VALID_BURSTY);
            drive_partial_weight_reconfig(1, WEIGHT_CHECKER, VALID_INTERMITTENT);
            drive_partial_weight_reconfig(1, WEIGHT_ALL_ONE, VALID_BURSTY);

            // Restore a known-good model before sending checked images.
            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();

            send_directed_image_suite(VALID_CONTINUOUS);
            send_random_images(EXTRA_RANDOM_IMAGES / 2 + 1, VALID_CONTINUOUS, 2);
        end

        // ---------------------------------------------------------------------
        // Scenario 7: Output-side reset under backpressure.
        // ---------------------------------------------------------------------
        if (ENABLE_RESET_STRESS) begin
            $display("[%0t] Scenario 7: reset during output backpressure.", $realtime);
            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();
            do_output_phase_reset_scenario();

            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();
            send_class_cover_sequence(VALID_CONTINUOUS);
        end



        // ---------------------------------------------------------------------
        // Scenario 8: Deterministic inter-image gap sweep.
        // Keep checked traffic short.
        // ---------------------------------------------------------------------
        $display("[%0t] Scenario 8: deterministic inter-image gaps.", $realtime);
        do_reset(RESET_BEFORE_CONFIG, 1'b0);
        ready_mode_r <= READY_ALWAYS;
        drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
        wait_until_input_ready();
        send_inter_image_gap_sweep(VALID_CONTINUOUS);

        // ---------------------------------------------------------------------
        // Scenario 9: Reset exactly on configuration-message boundary.
        // Coverage stress first, then short checked recovery.
        // ---------------------------------------------------------------------
        if (ENABLE_RESET_STRESS) begin
            $display("[%0t] Scenario 9: reset on config boundary.", $realtime);
            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            do_reset_on_config_boundary();

            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();
            send_directed_image_suite(VALID_CONTINUOUS);
        end

        // ---------------------------------------------------------------------
        // Scenario 10: Reset exactly on image boundary.
        // Coverage stress first, then very short checked recovery.
        // ---------------------------------------------------------------------
        if (ENABLE_RESET_STRESS) begin
            bit [INPUT_DATA_WIDTH-1:0] img[];

            $display("[%0t] Scenario 10: reset on image boundary.", $realtime);
            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();

            make_checkerboard_vector(img);
            drive_image_then_reset_on_boundary(img, VALID_CONTINUOUS);

            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();
            send_directed_image_suite(VALID_CONTINUOUS);
        end

        // ---------------------------------------------------------------------
        // Scenario 11: Deterministic configuration-gap sweep.
        // Keep post-gap checked traffic minimal.
        // ---------------------------------------------------------------------
        $display("[%0t] Scenario 11: deterministic configuration gaps.", $realtime);
        do_reset(RESET_BEFORE_CONFIG, 1'b0);
        ready_mode_r <= READY_ALWAYS;
        drive_full_configuration_fixed_gap(CFG_ORDER_STANDARD, 1);
        wait_until_input_ready();
        send_directed_image_suite(VALID_CONTINUOUS);

        do_reset(RESET_BEFORE_CONFIG, 1'b0);
        ready_mode_r <= READY_ALWAYS;
        drive_full_configuration_fixed_gap(CFG_ORDER_WEIGHTS_THEN_THRESHOLDS, 3);
        wait_until_input_ready();
        send_directed_image_suite(VALID_CONTINUOUS);

        // ---------------------------------------------------------------------
        // Scenario 12: Broader partial-reconfiguration sweep.
        // Coverage only for the partial updates themselves.
        // Restore known-good configuration before checked traffic.
        // ---------------------------------------------------------------------
        if (ENABLE_PARTIAL_RECONFIG) begin
            $display("[%0t] Scenario 12: broader partial reconfiguration sweep.", $realtime);
            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();

            // Coverage-only stress. Do not rely on live partial updates
            // to remain functionally identical to the software model.
            drive_partial_reconfig_sweep(VALID_CONTINUOUS);

            do_reset(RESET_BEFORE_CONFIG, 1'b0);
            ready_mode_r <= READY_ALWAYS;
            drive_full_configuration(CFG_ORDER_STANDARD, VALID_CONTINUOUS);
            wait_until_input_ready();
            send_directed_image_suite(VALID_CONTINUOUS);
        end

        $display("[%0t] All scenarios loaded, waiting for outputs.", $realtime);
        wait (expected_outputs.size() == 0);
        repeat (10) @(posedge clk);

        disable generate_clock;
        disable l_timeout;

        if (failed == 0)
            $display(
                "[%0t] SUCCESS: coverage testbench completed with %0d checked outputs.",
                $realtime,
                total_checked_outputs
            );
        else $error("FAILED: %0d checked outputs failed.", failed);

        $display("\nCoverage TB Stats:");
        $display("  Checked outputs              : %0d", total_checked_outputs);
        $display("  Expected outputs issued      : %0d", total_expected_outputs);
        $display("  Total resets applied         : %0d", total_reset_count);
        $display("  Avg latency (cycles)         : %0.1f cycles", latency.get_avg_cycles());
        $display("  Avg latency (time)           : %0.1f ns", latency.get_avg_time());
        $display("  Avg throughput (outputs/sec) : %0.1f", throughput.get_outputs_per_sec(
                 total_checked_outputs));
        $display("  Avg throughput (cycles/out)  : %0.1f", throughput.get_avg_cycles_per_output(
                 total_checked_outputs));
        $display("  Class-cover images found     : %0d", last_class_cover_count_r);
    end

    // =========================================================================
    // Output ready generation
    // =========================================================================
    initial begin : l_toggle_ready
        int burst_count;
        int delay_count;

        burst_count = 0;
        delay_count = 0;

        data_out.tready <= 1'b1;
        @(posedge clk iff !rst);

        forever begin
            if (rst || !TOGGLE_DATA_OUT_READY) begin
                data_out.tready <= 1'b1;
                burst_count = 0;
                delay_count = 0;
            end else begin
                case (ready_mode_r)
                    READY_ALWAYS: begin
                        data_out.tready <= 1'b1;
                        burst_count = 0;
                        delay_count = 0;
                    end

                    READY_RANDOM: begin
                        data_out.tready <= $urandom_range(0, 1);
                        burst_count = 0;
                        delay_count = 0;
                    end

                    READY_BURSTY: begin
                        if (burst_count == 0) begin
                            burst_count = $urandom_range(1, 4);
                            data_out.tready <= ~data_out.tready;
                        end else begin
                            burst_count--;
                        end
                    end

                    READY_DELAYED: begin
                        if (data_out.tvalid) begin
                            if (!data_out.tready && delay_count == 0) delay_count = $urandom_range(0, 3);

                            if (delay_count == 0) begin
                                data_out.tready <= 1'b1;
                            end else begin
                                data_out.tready <= 1'b0;
                                delay_count--;
                            end
                        end else begin
                            data_out.tready <= chance(0.5);
                            delay_count = 0;
                        end
                    end

                    default: begin
                        data_out.tready <= 1'b1;
                    end
                endcase
            end

            @(posedge clk);
        end
    end

    // =========================================================================
    // Output wait tracking / ready alignment tracking
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            output_wait_cycles_r       <= 0;
            output_valid_prev_r        <= 1'b0;
            output_ready_prev_r        <= 1'b0;
            tready_before_valid_r      <= 1'b0;
            tready_same_cycle_r        <= 1'b0;
            tready_after_valid_r       <= 1'b0;
            output_waiting_for_ready_r <= 1'b0;
        end else begin
            if (data_out.tvalid && !output_valid_prev_r) begin
                tready_before_valid_r      <= output_ready_prev_r;
                tready_same_cycle_r        <= data_out.tready;
                tready_after_valid_r       <= 1'b0;
                output_waiting_for_ready_r <= !data_out.tready;
            end else if (output_waiting_for_ready_r && data_out.tvalid && data_out.tready) begin
                tready_after_valid_r       <= 1'b1;
                output_waiting_for_ready_r <= 1'b0;
            end else if (!data_out.tvalid) begin
                output_waiting_for_ready_r <= 1'b0;
            end

            if (data_out.tvalid && !data_out.tready) output_wait_cycles_r <= output_wait_cycles_r + 1;
            else if (!data_out.tvalid) output_wait_cycles_r <= 0;

            output_valid_prev_r <= data_out.tvalid;
            output_ready_prev_r <= data_out.tready;
        end
    end

    // =========================================================================
    // Output monitor / scoreboard
    // =========================================================================
    initial begin : l_output_monitor
        automatic int output_count = 0;

        forever begin
            @(posedge clk iff (!rst && data_out.tvalid && data_out.tready));

            assert (expected_outputs.size() > 0)
            else $fatal(1, "No expected output for actual output");

            assert (data_out.tdata == expected_outputs[0]) begin
                passed++;
            end else begin
                $error("Output incorrect for image %0d: actual = %0d vs expected = %0d", output_count,
                       data_out.tdata, expected_outputs[0]);
                failed++;
            end

            if (last_output_class_r == data_out.tdata) repeated_output_class_r = 1;
            else repeated_output_class_r = 0;

            output_cov.sample(data_out.tdata, gap_bucket(output_wait_cycles_r), ready_mode_r,
                              repeated_output_class_r);

            output_align_cov.sample(tready_before_valid_r, tready_same_cycle_r, tready_after_valid_r);

            last_output_class_r = data_out.tdata;
            void'(expected_outputs.pop_front());
            latency.end_event(completed_image_id);

            if (expected_outputs.size() == 0) throughput.sample_end();

            output_count++;
            completed_image_id++;
            total_checked_outputs++;
        end
    end

    // =========================================================================
    // Timeout
    // =========================================================================
    initial begin : l_timeout
        #TIMEOUT;
        $fatal(1, $sformatf("Simulation failed due to timeout of %0t.", TIMEOUT));
    end
endmodule
