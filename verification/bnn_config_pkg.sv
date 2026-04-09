package bnn_config_pkg;

    typedef byte unsigned byte_t;
    typedef byte_t byte_q_t[$];

    typedef struct packed {
        logic [63:0] data;
        logic [7:0]  keep;
        logic        last;
    } axi64_beat_t;

    typedef struct packed {
        logic [7:0]  msgtype;
        logic [7:0]  layerid;
        logic [15:0] layerinputs;
        logic [15:0] numneurons;
        logic [15:0] bytesperneuron;
        logic [31:0] totalbytes;
    } cfg_header_t;

    typedef struct {
        int unsigned neuron;
        int unsigned beat;
        int unsigned word;
    } weight_write_t;

    typedef struct {
        int unsigned neuron;
        int unsigned thresh;
    } thresh_write_t;

    function automatic int unsigned ceil_div(input int unsigned a, input int unsigned b);
        return (a + b - 1) / b;
    endfunction

    function automatic int unsigned countw(input int unsigned n_inputs);
        int unsigned i;
        begin
            if (n_inputs == 0) return 1;
            countw = 1;
            for (i = 1; i < 32; i++) begin
                if ((1 << i) >= (n_inputs + 1))
                    countw = i;
            end
        end
    endfunction

    task automatic append_u16_le(ref byte_q_t q, input int unsigned v);
        q.push_back(byte_t'(v[7:0]));
        q.push_back(byte_t'(v[15:8]));
    endtask

    task automatic append_u32_le(ref byte_q_t q, input int unsigned v);
        q.push_back(byte_t'(v[7:0]));
        q.push_back(byte_t'(v[15:8]));
        q.push_back(byte_t'(v[23:16]));
        q.push_back(byte_t'(v[31:24]));
    endtask

    task automatic append_header(
        ref byte_q_t q,
        input int unsigned msgtype,
        input int unsigned layerid,
        input int unsigned layerinputs,
        input int unsigned numneurons,
        input int unsigned bytesperneuron,
        input int unsigned totalbytes
    );
        q.push_back(byte_t'(msgtype));
        q.push_back(byte_t'(layerid));
        append_u16_le(q, layerinputs);
        append_u16_le(q, numneurons);
        append_u16_le(q, bytesperneuron);
        append_u32_le(q, totalbytes);
        append_u32_le(q, 32'd0); // reserved
    endtask

    task automatic build_weight_message(
        ref byte_q_t stream,
        input int unsigned layerid,
        input int unsigned layerinputs,
        input bit weights[][]
    );
        int unsigned numneurons;
        int unsigned bytesperneuron;
        byte_q_t payload;
        int unsigned n, i, bidx;
        byte_t neu_byte;

        numneurons      = $size(weights, 1);
        bytesperneuron  = ceil_div(layerinputs, 8);

        payload.delete();
        for (n = 0; n < numneurons; n++) begin
            for (bidx = 0; bidx < bytesperneuron; bidx++) begin
                neu_byte = 8'h00;
                for (i = 0; i < 8; i++) begin
                    int unsigned inp_idx;
                    inp_idx = bidx * 8 + i;
                    if (inp_idx < layerinputs) begin
                        if (weights[n][inp_idx])
                            neu_byte[i] = 1'b1;
                    end else begin
                        neu_byte[i] = 1'b1; // pad with ones
                    end
                end
                payload.push_back(neu_byte);
            end
        end

        append_header(stream, 0, layerid, layerinputs, numneurons, bytesperneuron, payload.size());
        foreach (payload[i]) stream.push_back(payload[i]);
    endtask

    task automatic build_threshold_message(
        ref byte_q_t stream,
        input int unsigned layerid,
        input int unsigned thresholds[]
    );
        int unsigned numneurons;
        byte_q_t payload;

        numneurons = $size(thresholds);
        payload.delete();

        foreach (thresholds[i]) begin
            append_u32_le(payload, thresholds[i]);
        end

        append_header(stream, 1, layerid, 0, numneurons, 4, payload.size());
        foreach (payload[i]) stream.push_back(payload[i]);
    endtask

    task automatic build_layer_stream(
        ref byte_q_t stream,
        input int unsigned layerid,
        input int unsigned layerinputs,
        input bit weights[][],
        input int unsigned thresholds[]
    );
        build_weight_message(stream, layerid, layerinputs, weights);
        build_threshold_message(stream, layerid, thresholds);
    endtask

    task automatic pack_bytes_to_axi64(
        input  byte_q_t bytes,
        output axi64_beat_t beats[$]
    );
        axi64_beat_t beat;
        int unsigned i, b;

        beats.delete();

        for (i = 0; i < bytes.size(); i += 8) begin
            beat = '0;
            for (b = 0; b < 8; b++) begin
                if ((i + b) < bytes.size()) begin
                    beat.data[b*8 +: 8] = bytes[i+b];
                    beat.keep[b]        = 1'b1;
                end
            end
            beat.last = ((i + 8) >= bytes.size());
            beats.push_back(beat);
        end
    endtask

    function automatic cfg_header_t parse_header(input byte_q_t stream, input int unsigned off);
        cfg_header_t h;
        begin
            h.msgtype        = stream[off + 0];
            h.layerid        = stream[off + 1];
            h.layerinputs    = {stream[off + 3], stream[off + 2]};
            h.numneurons     = {stream[off + 5], stream[off + 4]};
            h.bytesperneuron = {stream[off + 7], stream[off + 6]};
            h.totalbytes     = {stream[off + 11], stream[off + 10], stream[off + 9], stream[off + 8]};
            return h;
        end
    endfunction

    task automatic process_stream_for_layer(
        input  byte_q_t stream,
        input  int unsigned target_layer_id,
        input  int unsigned pw,
        output cfg_header_t weight_meta,
        output bit          weight_meta_valid,
        output weight_write_t exp_w[$],
        output thresh_write_t exp_t[$]
    );
        int unsigned off;
        cfg_header_t h;

        exp_w.delete();
        exp_t.delete();
        weight_meta       = '0;
        weight_meta_valid = 1'b0;

        off = 0;
        while ((off + 16) <= stream.size()) begin
            int unsigned payload_off;
            h = parse_header(stream, off);
            off = off + 16;
            payload_off = off;

            if (h.layerid == target_layer_id[7:0]) begin
                if (h.msgtype == 0) begin
                    int unsigned beats;
                    beats = ceil_div(h.layerinputs, pw);
                    for (int unsigned n = 0; n < h.numneurons; n++) begin
                        for (int unsigned beat = 0; beat < beats; beat++) begin
                            int unsigned word;
                            word = 0;
                            for (int unsigned p = 0; p < pw; p++) begin
                                int unsigned idx;
                                int unsigned byte_index;
                                int unsigned bit_index;
                                bit wbit;
                                idx = beat * pw + p;
                                if (idx < h.layerinputs) begin
                                    byte_index = payload_off + n*h.bytesperneuron + (idx / 8);
                                    bit_index  = idx % 8;
                                    wbit = stream[byte_index][bit_index];
                                end else begin
                                    wbit = 1'b1;
                                end
                                word[p] = wbit;
                            end
                            exp_w.push_back('{neuron:n, beat:beat, word:word});
                        end
                    end
                    weight_meta       = h;
                    weight_meta_valid = 1'b1;
                end else begin
                    int unsigned cw;
                    cw = weight_meta_valid ? countw(weight_meta.layerinputs) : 1;
                    for (int unsigned n = 0; n < h.numneurons; n++) begin
                        int unsigned raw;
                        int unsigned mask;
                        raw  = {stream[payload_off + n*4 + 3],
                                stream[payload_off + n*4 + 2],
                                stream[payload_off + n*4 + 1],
                                stream[payload_off + n*4 + 0]};
                        mask = (cw >= 32) ? 32'hFFFF_FFFF : ((1 << cw) - 1);
                        exp_t.push_back('{neuron:n, thresh:(raw & mask)});
                    end
                end
            end

            off = payload_off + h.totalbytes;
        end
    endtask

endpackage