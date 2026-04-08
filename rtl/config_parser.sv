// rtl/config_parser.sv
//
// Phase 1: Decodes the 128-bit config message header arriving on
// a CONFIG_BUS_WIDTH AXI4-Stream bus. The header spans two beats
// on the default 64-bit bus:
//
//   Beat 0 (Word 0):
//     [7:0]   msg_type         (0=weights, 1=thresholds)
//     [15:8]  layer_id
//     [31:16] layer_inputs     (fan-in of this layer)
//     [47:32] num_neurons
//     [63:48] bytes_per_neuron
//
//   Beat 1 (Word 1):
//     [31:0]  total_bytes      (payload size in bytes)
//     [63:32] reserved
//
// After Word 1 is captured, header_valid pulses for one cycle and
// the FSM moves to S_PAYLOAD, counting bytes until config_last.
//
// Phase 2 will add: asymmetric FIFO + per-byte payload output.
// Phase 3 will add: RAM write logic for weights and thresholds.

module config_parser #(
    parameter int CONFIG_BUS_WIDTH = 64
)(
    input  logic clk,
    input  logic rst,  // synchronous, active-high

    // AXI4-Stream config input
    input  logic                            config_valid,
    output logic                            config_ready,
    input  logic [CONFIG_BUS_WIDTH-1:0]     config_data,
    input  logic [CONFIG_BUS_WIDTH/8-1:0]   config_keep,
    input  logic                            config_last,

    // Decoded header (held stable; header_valid pulses for 1 cycle when new)
    output logic        header_valid,
    output logic [7:0]  o_msg_type,
    output logic [7:0]  o_layer_id,
    output logic [15:0] o_layer_inputs,
    output logic [15:0] o_num_neurons,
    output logic [15:0] o_bytes_per_neuron,
    output logic [31:0] o_total_bytes,

    // Payload status (Phase 2 will replace these with a real byte stream)
    output logic [31:0] payload_byte_count, // bytes consumed in current message
    output logic        config_done         // held high after all messages received
                                            // (Phase 3 will drive this properly)
);

    // -----------------------------------------------------------------------
    // FSM state encoding
    // -----------------------------------------------------------------------
    typedef enum logic [1:0] {
        S_HDR_W0  = 2'b00,  // waiting for header beat 0
        S_HDR_W1  = 2'b01,  // waiting for header beat 1
        S_PAYLOAD = 2'b10   // consuming payload beats
    } state_t;

    state_t state;

    // In Phase 1 we never apply backpressure. Phase 2 may deassert
    // config_ready briefly while the FIFO is full.
    assign config_ready = 1'b1;

    logic handshake;
    assign handshake = config_valid & config_ready;

    // -----------------------------------------------------------------------
    // Header registers
    // -----------------------------------------------------------------------
    logic [7:0]  r_msg_type;
    logic [7:0]  r_layer_id;
    logic [15:0] r_layer_inputs;
    logic [15:0] r_num_neurons;
    logic [15:0] r_bytes_per_neuron;
    logic [31:0] r_total_bytes;

    // -----------------------------------------------------------------------
    // FSM + datapath — single always_ff for clean synthesis
    // -----------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            state              <= S_HDR_W0;
            header_valid       <= 1'b0;
            payload_byte_count <= '0;
            r_msg_type         <= '0;
            r_layer_id         <= '0;
            r_layer_inputs     <= '0;
            r_num_neurons      <= '0;
            r_bytes_per_neuron <= '0;
            r_total_bytes      <= '0;
        end else begin

            header_valid <= 1'b0; // default: not pulsing this cycle

            case (state)

                // --------------------------------------------------------
                // S_HDR_W0: latch lower 64 bits of header
                // --------------------------------------------------------
                S_HDR_W0: begin
                    if (handshake) begin
                        r_msg_type         <= config_data[7:0];
                        r_layer_id         <= config_data[15:8];
                        r_layer_inputs     <= config_data[31:16];
                        r_num_neurons      <= config_data[47:32];
                        r_bytes_per_neuron <= config_data[63:48];
                        state              <= S_HDR_W1;
                    end
                end

                // --------------------------------------------------------
                // S_HDR_W1: latch upper 64 bits of header
                //   config_data[31:0]  = total_bytes
                //   config_data[63:32] = reserved (ignored)
                // --------------------------------------------------------
                S_HDR_W1: begin
                    if (handshake) begin
                        r_total_bytes      <= config_data[31:0];
                        header_valid       <= 1'b1;
                        payload_byte_count <= '0;
                        state              <= S_PAYLOAD;
                    end
                end

                // --------------------------------------------------------
                // S_PAYLOAD: count valid bytes via TKEEP until config_last
                // --------------------------------------------------------
                S_PAYLOAD: begin
                    if (handshake) begin
                        payload_byte_count <= payload_byte_count
                                             + $countones(config_keep);
                        if (config_last)
                            state <= S_HDR_W0;
                    end
                end

                default: state <= S_HDR_W0;
            endcase
        end
    end

    // -----------------------------------------------------------------------
    // Output assignments
    // -----------------------------------------------------------------------
    assign o_msg_type         = r_msg_type;
    assign o_layer_id         = r_layer_id;
    assign o_layer_inputs     = r_layer_inputs;
    assign o_num_neurons      = r_num_neurons;
    assign o_bytes_per_neuron = r_bytes_per_neuron;
    assign o_total_bytes      = r_total_bytes;

    // Phase 3 will properly track when all layers are loaded
    assign config_done = 1'b0;

endmodule