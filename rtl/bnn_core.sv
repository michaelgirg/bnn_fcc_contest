`default_nettype none
// =============================================================================
// Module: bnn_core
// =============================================================================
module bnn_core #(
    parameter int INPUTS  = 784,
    parameter int HIDDEN1 = 256,
    parameter int HIDDEN2 = 256,
    parameter int OUTPUTS = 10,
    parameter int PW      = 8,
    parameter int PN      = 8
) (
    input  wire logic                                       clk,
    input  wire logic                                       rst,
    input  wire logic [                         INPUTS-1:0] image_input,
    input  wire logic                                       image_valid,
    output logic                                            image_ready,
    output logic      [                        OUTPUTS-1:0] result_activations,
    output logic      [      OUTPUTS*$clog2(HIDDEN2+1)-1:0] result_popcounts,
    output logic                                            result_valid,
    output logic                                            done,
    input  wire logic                                       cfg_write_en,
    input  wire logic [$clog2(HIDDEN1+HIDDEN2+OUTPUTS)-1:0] cfg_global_neuron_idx,
    input  wire logic [       $clog2((INPUTS+PW-1)/PW)-1:0] cfg_weight_addr,
    input  wire logic [                             PW-1:0] cfg_weight_data,
    input  wire logic [               $clog2(INPUTS+1)-1:0] cfg_threshold_data,
    input  wire logic                                       cfg_threshold_write
);
    // =========================================================================
    // Local parameters
    // =========================================================================
    localparam int L1_COUNT_WIDTH = $clog2(INPUTS + 1);
    localparam int L2_COUNT_WIDTH = $clog2(HIDDEN1 + 1);
    localparam int L3_COUNT_WIDTH = $clog2(HIDDEN2 + 1);
    localparam int L1_ADDR_W = $clog2((INPUTS + PW - 1) / PW);
    localparam int L2_ADDR_W = $clog2((HIDDEN1 + PW - 1) / PW);
    localparam int L3_ADDR_W = $clog2((HIDDEN2 + PW - 1) / PW);
    localparam int L1_NEURON_W = (HIDDEN1 > 1) ? $clog2(HIDDEN1) : 1;
    localparam int L2_NEURON_W = (HIDDEN2 > 1) ? $clog2(HIDDEN2) : 1;
    localparam int L3_NEURON_W = (OUTPUTS > 1) ? $clog2(OUTPUTS) : 1;

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [2:0] {
        IDLE    = 3'b000,
        RUN_L1  = 3'b001,
        LOAD_L2 = 3'b010,
        RUN_L2  = 3'b011,
        LOAD_L3 = 3'b100,
        RUN_L3  = 3'b101
    } state_t;
    state_t state_r, state_prev_r;

    // =========================================================================
    // Layer signals
    // =========================================================================
    logic l1_input_load, l1_start, l1_done, l1_busy, l1_output_valid;
    logic [HIDDEN1-1:0] l1_activations_out, l1_activations_r;
    logic l2_input_load, l2_start, l2_done, l2_busy, l2_output_valid;
    logic [HIDDEN2-1:0] l2_activations_out, l2_activations_r;
    logic l3_input_load, l3_start, l3_done, l3_busy, l3_output_valid;
    logic [OUTPUTS-1:0] l3_activations_out;
    logic [OUTPUTS*L3_COUNT_WIDTH-1:0] l3_popcounts_out, l3_popcounts_r;

    logic done_r, result_valid_r;
    assign done         = done_r;
    assign result_valid = result_valid_r;
    assign image_ready  = (state_r == IDLE);

    // =========================================================================
    // State tracking
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) state_prev_r <= IDLE;
        else state_prev_r <= state_r;
    end

    // =========================================================================
    // Start pulse generation - ONE-SHOT on state entry
    // =========================================================================
    assign l1_start = (state_r == RUN_L1) && (state_prev_r != RUN_L1);
    assign l2_start = (state_r == RUN_L2) && (state_prev_r != RUN_L2);
    assign l3_start = (state_r == RUN_L3) && (state_prev_r != RUN_L3);

    // =========================================================================
    // Load pulse generation
    // =========================================================================
    assign l1_input_load = (state_r == IDLE) && image_valid;
    assign l2_input_load = (state_r == LOAD_L2);
    assign l3_input_load = (state_r == LOAD_L3);

    // =========================================================================
    // Inter-layer capture - WAIT for output_valid, not just done
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            l1_activations_r <= '0;
            l2_activations_r <= '0;
            l3_popcounts_r   <= '0;
        end else begin
            if (l1_output_valid) l1_activations_r <= l1_activations_out;
            if (l2_output_valid) l2_activations_r <= l2_activations_out;
            if (l3_output_valid) l3_popcounts_r <= l3_popcounts_out;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state_r        <= IDLE;
            done_r         <= 1'b0;
            result_valid_r <= 1'b0;
        end else begin
            done_r         <= 1'b0;
            result_valid_r <= 1'b0;

            case (state_r)
                IDLE: begin
                    if (image_valid) begin
                        state_r <= RUN_L1;
                    end
                end

                RUN_L1: begin
                    if (l1_output_valid) begin
                        state_r <= LOAD_L2;
                    end
                end

                LOAD_L2: begin
                    state_r <= RUN_L2;
                end

                RUN_L2: begin
                    if (l2_output_valid) begin
                        state_r <= LOAD_L3;
                    end
                end

                LOAD_L3: begin
                    state_r <= RUN_L3;
                end

                RUN_L3: begin
                    if (l3_output_valid) begin
                        result_valid_r <= 1'b1;
                        done_r         <= 1'b1;
                        state_r        <= IDLE;
                    end
                end

                default: begin
                    state_r <= IDLE;
                end
            endcase
        end
    end

    assign result_activations = l3_activations_out;
    assign result_popcounts   = l3_popcounts_r;

    // =========================================================================
    // Configuration routing
    // =========================================================================
    logic l1_cfg_write_en, l2_cfg_write_en, l3_cfg_write_en;
    logic l1_cfg_threshold_write, l2_cfg_threshold_write, l3_cfg_threshold_write;
    logic [L1_NEURON_W-1:0] l1_local_neuron_idx;
    logic [L2_NEURON_W-1:0] l2_local_neuron_idx;
    logic [L3_NEURON_W-1:0] l3_local_neuron_idx;
    logic cfg_enable;
    assign cfg_enable = (state_r == IDLE);

    always_comb begin
        l1_cfg_write_en = 1'b0;
        l2_cfg_write_en = 1'b0;
        l3_cfg_write_en = 1'b0;
        l1_cfg_threshold_write = 1'b0;
        l2_cfg_threshold_write = 1'b0;
        l3_cfg_threshold_write = 1'b0;
        l1_local_neuron_idx = '0;
        l2_local_neuron_idx = '0;
        l3_local_neuron_idx = '0;

        if (cfg_enable) begin
            if (cfg_global_neuron_idx < HIDDEN1) begin
                l1_cfg_write_en        = cfg_write_en;
                l1_cfg_threshold_write = cfg_threshold_write;
                l1_local_neuron_idx    = L1_NEURON_W'(cfg_global_neuron_idx);
            end else if (cfg_global_neuron_idx < HIDDEN1 + HIDDEN2) begin
                l2_cfg_write_en        = cfg_write_en;
                l2_cfg_threshold_write = cfg_threshold_write;
                l2_local_neuron_idx    = L2_NEURON_W'(cfg_global_neuron_idx - HIDDEN1);
            end else begin
                l3_cfg_write_en        = cfg_write_en;
                l3_cfg_threshold_write = cfg_threshold_write;
                l3_local_neuron_idx    = L3_NEURON_W'(cfg_global_neuron_idx - (HIDDEN1 + HIDDEN2));
            end
        end
    end

    // =========================================================================
    // Layer instantiations
    // =========================================================================
    bnn_layer #(
        .INPUTS      (INPUTS),
        .NEURONS     (HIDDEN1),
        .PW          (PW),
        .PN          (PN),
        .OUTPUT_LAYER(1'b0)
    ) layer1 (
        .clk                (clk),
        .rst                (rst),
        .input_load         (l1_input_load),
        .input_vector       (image_input),
        .start              (l1_start),
        .done               (l1_done),
        .busy               (l1_busy),
        .output_valid       (l1_output_valid),
        .activations_out    (l1_activations_out),
        .popcounts_out      (  /* unused */),
        .cfg_write_en       (l1_cfg_write_en),
        .cfg_neuron_idx     (l1_local_neuron_idx),
        .cfg_weight_addr    (cfg_weight_addr[L1_ADDR_W-1:0]),
        .cfg_weight_data    (cfg_weight_data),
        .cfg_threshold_data (cfg_threshold_data[L1_COUNT_WIDTH-1:0]),
        .cfg_threshold_write(l1_cfg_threshold_write),
        .cfg_ready          (  /* unused */)
    );

    bnn_layer #(
        .INPUTS      (HIDDEN1),
        .NEURONS     (HIDDEN2),
        .PW          (PW),
        .PN          (PN),
        .OUTPUT_LAYER(1'b0)
    ) layer2 (
        .clk                (clk),
        .rst                (rst),
        .input_load         (l2_input_load),
        .input_vector       (l1_activations_r),
        .start              (l2_start),
        .done               (l2_done),
        .busy               (l2_busy),
        .output_valid       (l2_output_valid),
        .activations_out    (l2_activations_out),
        .popcounts_out      (  /* unused */),
        .cfg_write_en       (l2_cfg_write_en),
        .cfg_neuron_idx     (l2_local_neuron_idx),
        .cfg_weight_addr    (cfg_weight_addr[L2_ADDR_W-1:0]),
        .cfg_weight_data    (cfg_weight_data),
        .cfg_threshold_data (cfg_threshold_data[L2_COUNT_WIDTH-1:0]),
        .cfg_threshold_write(l2_cfg_threshold_write),
        .cfg_ready          (  /* unused */)
    );

    bnn_layer #(
        .INPUTS      (HIDDEN2),
        .NEURONS     (OUTPUTS),
        .PW          (PW),
        .PN          (PN),
        .OUTPUT_LAYER(1'b1)
    ) layer3 (
        .clk                (clk),
        .rst                (rst),
        .input_load         (l3_input_load),
        .input_vector       (l2_activations_r),
        .start              (l3_start),
        .done               (l3_done),
        .busy               (l3_busy),
        .output_valid       (l3_output_valid),
        .activations_out    (l3_activations_out),
        .popcounts_out      (l3_popcounts_out),
        .cfg_write_en       (l3_cfg_write_en),
        .cfg_neuron_idx     (l3_local_neuron_idx),
        .cfg_weight_addr    (cfg_weight_addr[L3_ADDR_W-1:0]),
        .cfg_weight_data    (cfg_weight_data),
        .cfg_threshold_data (cfg_threshold_data[L3_COUNT_WIDTH-1:0]),
        .cfg_threshold_write(l3_cfg_threshold_write),
        .cfg_ready          (  /* unused */)
    );
endmodule
`default_nettype wire
