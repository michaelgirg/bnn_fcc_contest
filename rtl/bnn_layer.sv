//==============================================================================
// Weight Bank Module - Separate BRAM instance per processor
//==============================================================================
module weight_bank #(
    parameter int PW    = 8,
    parameter int DEPTH = 3136
) (
    input  wire logic                     clk,
    input  wire logic                     write_en,
    input  wire logic [$clog2(DEPTH)-1:0] write_addr,
    input  wire logic [           PW-1:0] write_data,
    input  wire logic [$clog2(DEPTH)-1:0] read_addr,
    output logic      [           PW-1:0] read_data
);
    localparam int ADDR_WIDTH = $clog2(DEPTH);

    (* ram_style = "block" *) logic [PW-1:0] mem[0:DEPTH-1];

    // Write port
    always_ff @(posedge clk) begin
        if (write_en) begin
            mem[write_addr] <= write_data;
        end
    end

    // Read port with output register for BRAM inference
    always_ff @(posedge clk) begin
        read_data <= mem[read_addr];
    end
endmodule

//==============================================================================
// Threshold Bank Module - Separate distributed RAM per processor
//==============================================================================
module threshold_bank #(
    parameter int COUNT_WIDTH = 10,
    parameter int DEPTH       = 32
) (
    input  wire logic                     clk,
    input  wire logic                     write_en,
    input  wire logic [$clog2(DEPTH)-1:0] write_addr,
    input  wire logic [  COUNT_WIDTH-1:0] write_data,
    input  wire logic [$clog2(DEPTH)-1:0] read_addr,
    output logic      [  COUNT_WIDTH-1:0] read_data
);
    localparam int ADDR_WIDTH = $clog2(DEPTH);

    (* ram_style = "distributed" *) logic [COUNT_WIDTH-1:0] mem[0:DEPTH-1];

    // Write port
    always_ff @(posedge clk) begin
        if (write_en) begin
            mem[write_addr] <= write_data;
        end
    end

    // Read port with output register
    always_ff @(posedge clk) begin
        read_data <= mem[read_addr];
    end
endmodule

//==============================================================================
// BNN Layer - Main Module
//==============================================================================
module bnn_layer #(
    parameter int INPUTS                = 784,
    parameter int NEURONS               = 256,
    parameter int PW                    = 8,
    parameter int PN                    = 8,
    parameter bit OUTPUT_LAYER          = 0,
    parameter int COUNT_WIDTH           = $clog2(INPUTS + 1),
    parameter int INPUT_BEATS           = (INPUTS + PW - 1) / PW,
    parameter int CFG_NEURON_WIDTH      = (NEURONS > 1) ? $clog2(NEURONS) : 1,
    parameter int CFG_WEIGHT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1
) (
    input  wire logic                             clk,
    input  wire logic                             rst,
    input  wire logic                             input_load,
    input  wire logic [               INPUTS-1:0] input_vector,
    input  wire logic                             start,
    output logic                                  done,
    output logic                                  busy,
    output logic                                  output_valid,
    output logic      [              NEURONS-1:0] activations_out,
    output logic      [  NEURONS*COUNT_WIDTH-1:0] popcounts_out,
    input  wire logic                             cfg_write_en,
    input  wire logic [     CFG_NEURON_WIDTH-1:0] cfg_neuron_idx,
    input  wire logic [CFG_WEIGHT_ADDR_WIDTH-1:0] cfg_weight_addr,
    input  wire logic [                   PW-1:0] cfg_weight_data,
    input  wire logic [          COUNT_WIDTH-1:0] cfg_threshold_data,
    input  wire logic                             cfg_threshold_write,
    output logic                                  cfg_ready
);
    //==========================================================================
    // Local Parameters
    //==========================================================================
    localparam int NEURON_GROUPS = (NEURONS + PN - 1) / PN;
    localparam int INPUT_ADDR_WIDTH = (INPUT_BEATS > 1) ? $clog2(INPUT_BEATS) : 1;
    localparam int WEIGHT_DEPTH = INPUT_BEATS * NEURON_GROUPS;
    localparam int WEIGHT_ADDR_WIDTH = (WEIGHT_DEPTH > 1) ? $clog2(WEIGHT_DEPTH) : 1;
    localparam int GROUP_WIDTH = (NEURON_GROUPS > 1) ? $clog2(NEURON_GROUPS) : 1;
    localparam int PADDED_INPUTS = INPUT_BEATS * PW;
    localparam int LAST_GROUP_ACTIVE = (NEURONS % PN == 0) ? PN : (NEURONS % PN);
    localparam logic [PN-1:0] LAST_GROUP_MASK = (LAST_GROUP_ACTIVE >= PN) ? {PN{1'b1}} : PN'((PN'(1) << LAST_GROUP_ACTIVE) - PN'(1));
    localparam logic [GROUP_WIDTH-1:0] LAST_GROUP_IDX = GROUP_WIDTH'(NEURON_GROUPS - 1);
    localparam logic [INPUT_ADDR_WIDTH-1:0] LAST_BEAT_IDX = INPUT_ADDR_WIDTH'(INPUT_BEATS - 1);

    //==========================================================================
    // Type Definitions
    //==========================================================================
    typedef enum logic [1:0] {
        IDLE,
        BEAT,
        WAIT_OUT
    } state_t;

    //==========================================================================
    // Helper Functions
    //==========================================================================
    function automatic int get_np_idx(input int neuron_idx);
        return neuron_idx % PN;
    endfunction

    function automatic int get_group_idx(input int neuron_idx);
        return neuron_idx / PN;
    endfunction

    //==========================================================================
    // State and Control Signals
    //==========================================================================
    state_t                        state_r;
    logic   [     GROUP_WIDTH-1:0] group_idx_r;
    logic   [INPUT_ADDR_WIDTH-1:0] beat_idx_r;
    logic   [              PN-1:0] np_active_r;
    logic                          done_r;
    logic                          output_valid_r;
    logic                          collect_done;
    logic                          all_groups_done;
    logic                          all_np_done;

    assign all_groups_done = (group_idx_r == LAST_GROUP_IDX);

    //==========================================================================
    // Input Buffer - Parallel-load register bank
    // Note: This cannot be inferred as RAM due to simultaneous multi-write
    //==========================================================================
    logic [PW-1:0] input_buffer[INPUT_BEATS];
    logic [PADDED_INPUTS-1:0] input_vector_padded;

    assign input_vector_padded = {{(PADDED_INPUTS - INPUTS) {1'b0}}, input_vector};

    always_ff @(posedge clk) begin
        if (input_load && !busy) begin
            for (int i = 0; i < INPUT_BEATS; i++) begin
                input_buffer[i] <= input_vector_padded[i*PW+:PW];
            end
        end
    end

    //==========================================================================
    // Configuration Address Decode
    //==========================================================================
    logic [ CFG_NEURON_WIDTH-1:0] cfg_np_idx;
    logic [      GROUP_WIDTH-1:0] cfg_grp_idx;
    logic [WEIGHT_ADDR_WIDTH-1:0] cfg_weight_local_addr;

    assign cfg_np_idx = CFG_NEURON_WIDTH'(get_np_idx(int'(cfg_neuron_idx)));
    assign cfg_grp_idx = GROUP_WIDTH'(get_group_idx(int'(cfg_neuron_idx)));
    assign cfg_weight_local_addr = WEIGHT_ADDR_WIDTH'(int'(cfg_grp_idx) * INPUT_BEATS + int'(cfg_weight_addr));

    //==========================================================================
    // Memory Bank Signals
    //==========================================================================
    logic [WEIGHT_ADDR_WIDTH-1:0] weight_rd_addr   [PN];
    logic [      GROUP_WIDTH-1:0] threshold_rd_addr[PN];
    logic [               PW-1:0] weight_rd_data   [PN];
    logic [      COUNT_WIDTH-1:0] threshold_rd_data[PN];

    logic [               PN-1:0] weight_wr_en;
    logic [               PN-1:0] threshold_wr_en;

    // Generate write enable signals for each bank
    always_comb begin
        for (int i = 0; i < PN; i++) begin
            weight_wr_en[i]    = cfg_write_en && (cfg_np_idx == i) && !busy;
            threshold_wr_en[i] = cfg_threshold_write && (cfg_np_idx == i) && !busy;
        end
    end

    //==========================================================================
    // Instantiate Weight and Threshold Banks
    //==========================================================================
    generate
        for (genvar i = 0; i < PN; i++) begin : gen_memory_banks
            weight_bank #(
                .PW   (PW),
                .DEPTH(WEIGHT_DEPTH)
            ) u_weight_bank (
                .clk       (clk),
                .write_en  (weight_wr_en[i]),
                .write_addr(cfg_weight_local_addr),
                .write_data(cfg_weight_data),
                .read_addr (weight_rd_addr[i]),
                .read_data (weight_rd_data[i])
            );

            threshold_bank #(
                .COUNT_WIDTH(COUNT_WIDTH),
                .DEPTH      (NEURON_GROUPS)
            ) u_threshold_bank (
                .clk       (clk),
                .write_en  (threshold_wr_en[i]),
                .write_addr(cfg_grp_idx),
                .write_data(cfg_threshold_data),
                .read_addr (threshold_rd_addr[i]),
                .read_data (threshold_rd_data[i])
            );
        end
    endgenerate

    //==========================================================================
    // Control Signals
    //==========================================================================
    logic issue_beat;
    logic issue_last;
    logic start_group;

    assign issue_beat = (state_r == BEAT);
    assign issue_last = (state_r == BEAT) && (beat_idx_r == LAST_BEAT_IDX);

    // Initiate new group processing:
    // - From IDLE on start: begin first group (group 0)
    // - From WAIT_OUT after group completion: begin next group
    //   Note: addresses are precomputed for group_idx_r+1 since group_idx_r 
    //   increments on the same cycle as start_group assertion
    assign start_group = ((state_r == IDLE) && start) ||
                         ((state_r == WAIT_OUT) && all_np_done && !all_groups_done);

    //==========================================================================
    // Valid and Last Pipeline
    // Pipeline depth = 2 cycles from issue_beat to np_valid_in:
    //   Cycle 1: RAM output registers (weight_rd_data, input_stage_r)
    //   Cycle 2: NP input registers (np_weight_in, np_input_in)
    //==========================================================================
    logic [1:0] np_valid_pipe_r;
    logic [1:0] np_last_pipe_r;
    logic       np_valid_in;
    logic       np_last;

    always_ff @(posedge clk) begin
        if (rst || state_r == IDLE) begin
            np_valid_pipe_r <= '0;
            np_last_pipe_r  <= '0;
        end else begin
            np_valid_pipe_r <= {np_valid_pipe_r[0], issue_beat};
            np_last_pipe_r  <= {np_last_pipe_r[0], issue_last};
        end
    end

    assign np_valid_in = np_valid_pipe_r[1];
    assign np_last     = np_last_pipe_r[1];

    //==========================================================================
    // Data Pipeline Stages
    //==========================================================================
    logic [              PW-1:0] input_stage_r;
    logic [              PW-1:0] np_input_in;
    logic [              PW-1:0] np_weight_in    [PN];
    logic [     COUNT_WIDTH-1:0] np_threshold_in [PN];
    logic [INPUT_ADDR_WIDTH-1:0] input_rd_addr_r;

    // Stage 1: Read from input buffer and weight banks (registered outputs from banks)
    // Stage 2: Register inputs to neuron processors
    always_ff @(posedge clk) begin
        if (rst) begin
            input_stage_r <= '0;
            np_input_in   <= '0;
            for (int i = 0; i < PN; i++) begin
                np_weight_in[i]    <= '0;
                np_threshold_in[i] <= '0;
            end
        end else begin
            // Stage 1: Latch input buffer read
            input_stage_r <= input_buffer[input_rd_addr_r];

            // Stage 2: Register to NP inputs
            np_input_in   <= input_stage_r;
            for (int i = 0; i < PN; i++) begin
                np_weight_in[i]    <= np_active_r[i] ? weight_rd_data[i] : '0;
                np_threshold_in[i] <= threshold_rd_data[i];
            end
        end
    end

    //==========================================================================
    // Address Generation - Combinational next-address calculation
    //==========================================================================
    logic [GROUP_WIDTH-1:0] grp_sel_next;

    always_comb begin
        if ((state_r == IDLE) && start) begin
            grp_sel_next = '0;
        end else begin
            grp_sel_next = group_idx_r + GROUP_WIDTH'(1);
        end
    end

    // Weight and threshold address control
    generate
        for (genvar i = 0; i < PN; i++) begin : gen_addr_ctrl
            always_ff @(posedge clk) begin
                if (rst) begin
                    weight_rd_addr[i]    <= '0;
                    threshold_rd_addr[i] <= '0;
                end else begin
                    if (start_group) begin
                        weight_rd_addr[i]    <= WEIGHT_ADDR_WIDTH'(int'(grp_sel_next) * INPUT_BEATS);
                        threshold_rd_addr[i] <= grp_sel_next;
                    end else if (issue_beat) begin
                        weight_rd_addr[i] <= weight_rd_addr[i] + WEIGHT_ADDR_WIDTH'(1);
                    end
                end
            end
        end
    endgenerate

    // Input address control
    always_ff @(posedge clk) begin
        if (rst) begin
            input_rd_addr_r <= '0;
        end else begin
            if (start_group) begin
                input_rd_addr_r <= '0;
            end else if (issue_beat) begin
                if (beat_idx_r == LAST_BEAT_IDX) begin
                    input_rd_addr_r <= '0;
                end else begin
                    input_rd_addr_r <= input_rd_addr_r + INPUT_ADDR_WIDTH'(1);
                end
            end
        end
    end

    //==========================================================================
    // Active Processor Mask - Combinational next-mask calculation
    //==========================================================================
    logic [PN-1:0] np_active_next;
    logic [GROUP_WIDTH-1:0] next_grp_for_mask;

    always_comb begin
        next_grp_for_mask = group_idx_r + GROUP_WIDTH'(1);

        if (next_grp_for_mask == LAST_GROUP_IDX) begin
            np_active_next = LAST_GROUP_MASK;
        end else begin
            np_active_next = {PN{1'b1}};
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            np_active_r <= '0;
        end else begin
            if ((state_r == IDLE) && start) begin
                np_active_r <= (NEURONS >= PN) ? {PN{1'b1}} : LAST_GROUP_MASK;
            end else if ((state_r == WAIT_OUT) && all_np_done && !all_groups_done) begin
                np_active_r <= np_active_next;
            end else if (state_r == IDLE && !start) begin
                np_active_r <= '0;
            end
        end
    end

    //==========================================================================
    // Neuron Processor Instantiation
    //==========================================================================
    logic [         PN-1:0] np_valid_out;
    logic [         PN-1:0] np_activation;
    logic [COUNT_WIDTH-1:0] np_popcount   [PN];

    assign all_np_done = &(np_valid_out | ~np_active_r);

    generate
        for (genvar i = 0; i < PN; i++) begin : gen_nps
            neuron_processor #(
                .PW               (PW),
                .MAX_NEURON_INPUTS(INPUTS),
                .OUTPUT_LAYER     (OUTPUT_LAYER)
            ) u_np (
                .clk         (clk),
                .rst         (rst),
                .valid_in    (np_valid_in && np_active_r[i]),
                .last        (np_last),
                .x           (np_input_in),
                .w           (np_weight_in[i]),
                .threshold   (np_threshold_in[i]),
                .valid_out   (np_valid_out[i]),
                .activation  (np_activation[i]),
                .popcount_out(np_popcount[i])
            );
        end
    endgenerate

    //==========================================================================
    // Output Collection
    //==========================================================================
    logic [    NEURONS-1:0] activations_buffer;
    logic [COUNT_WIDTH-1:0] popcounts_buffer   [NEURONS];

    // Combinational neuron index calculation
    int                     neuron_idx_comb    [     PN];
    always_comb begin
        for (int i = 0; i < PN; i++) begin
            neuron_idx_comb[i] = int'(group_idx_r) * PN + i;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            activations_buffer <= '0;
            popcounts_buffer   <= '{default: '0};
            collect_done       <= 1'b0;
        end else begin
            collect_done <= 1'b0;

            if ((state_r == IDLE) && start) begin
                activations_buffer <= '0;
                popcounts_buffer   <= '{default: '0};
            end else if ((state_r == WAIT_OUT) && all_np_done) begin
                for (int i = 0; i < PN; i++) begin
                    if (np_valid_out[i] && np_active_r[i]) begin
                        if (neuron_idx_comb[i] < NEURONS) begin
                            popcounts_buffer[neuron_idx_comb[i]] <= np_popcount[i];
                            if (OUTPUT_LAYER) begin
                                activations_buffer[neuron_idx_comb[i]] <= 1'b0;
                            end else begin
                                activations_buffer[neuron_idx_comb[i]] <= np_activation[i];
                            end
                        end
                    end
                end

                if (all_groups_done) begin
                    collect_done <= 1'b1;
                end
            end
        end
    end

    //==========================================================================
    // Main FSM
    //==========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            state_r        <= IDLE;
            group_idx_r    <= '0;
            beat_idx_r     <= '0;
            done_r         <= 1'b0;
            output_valid_r <= 1'b0;
        end else begin
            done_r <= 1'b0;

            // FSM logic FIRST
            case (state_r)
                IDLE: begin
                    output_valid_r <= 1'b0;
                    if (start) begin
                        state_r     <= BEAT;
                        group_idx_r <= '0;
                        beat_idx_r  <= '0;
                    end
                end
                BEAT: begin
                    if (beat_idx_r == LAST_BEAT_IDX) begin
                        state_r    <= WAIT_OUT;
                        beat_idx_r <= '0;
                    end else begin
                        beat_idx_r <= beat_idx_r + INPUT_ADDR_WIDTH'(1);
                    end
                end
                WAIT_OUT: begin
                    if (all_np_done) begin
                        if (all_groups_done) begin
                            state_r <= IDLE;
                        end else begin
                            group_idx_r <= group_idx_r + GROUP_WIDTH'(1);
                            beat_idx_r  <= '0;
                            state_r     <= BEAT;
                        end
                    end
                end
                default: state_r <= IDLE;
            endcase

            // Set outputs AFTER FSM so it overrides any clearing
            if (collect_done) begin
                output_valid_r <= 1'b1;
                done_r         <= 1'b1;
            end
        end
    end


    //==========================================================================
    // Output Assignments
    //==========================================================================
    assign busy            = (state_r != IDLE);
    assign done            = done_r;
    assign cfg_ready       = !busy;
    assign output_valid    = output_valid_r;
    assign activations_out = activations_buffer;

    always_comb begin
        for (int i = 0; i < NEURONS; i++) begin
            popcounts_out[i*COUNT_WIDTH+:COUNT_WIDTH] = popcounts_buffer[i];
        end
    end

endmodule

