//============================================================================
// Crystal Neural Network - FPGA Implementation
// Auto-generated from PyTorch crystal model
//
// Architecture:
//   Input dimension:  784
//   Output dimension: 10
//   Total neurons:    32
//   Frozen neurons:   26 (81.2%)
//
// Fixed-point format: Q8.8 (signed)
// Total bits per value: 16
//
// Note: Weights are embedded as parameters (ROM)
// Synthesis will optimize frozen weights as constants
//============================================================================
`timescale 1ns / 1ps


module crystal_net #(
    parameter INPUT_DIM = 784,
    parameter OUTPUT_DIM = 10,
    parameter NUM_NEURONS = 32,
    parameter DATA_WIDTH = 16
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         valid_in,
    input  wire signed [DATA_WIDTH-1:0] pixel_in,      // Serial input (one pixel per cycle)
    input  wire [$clog2(INPUT_DIM)-1:0] pixel_idx,     // Which pixel
    input  wire                         last_pixel,    // Last pixel flag

    output reg                          valid_out,
    output reg  signed [DATA_WIDTH-1:0] class_scores [0:OUTPUT_DIM-1],
    output reg  [3:0]                   predicted_class
);

// Internal registers
reg signed [DATA_WIDTH-1:0] input_buffer [0:INPUT_DIM-1];
reg signed [31:0] neuron_accum [0:NUM_NEURONS-1];  // Wide accumulator
reg signed [DATA_WIDTH-1:0] neuron_out [0:NUM_NEURONS-1];
reg [2:0] state;

localparam IDLE = 3'd0;
localparam LOADING = 3'd1;
localparam COMPUTE_NEURONS = 3'd2;
localparam COMPUTE_OUTPUT = 3'd3;
localparam DONE = 3'd4;

// Computation counters
reg [$clog2(INPUT_DIM)-1:0] comp_idx;
reg [$clog2(NUM_NEURONS)-1:0] neuron_idx;


// Tanh approximation function (piecewise linear)
function signed [15:0] tanh_approx;
    input signed [15:0] x;
    reg signed [15:0] abs_x;
    reg sign_bit;
    begin
        sign_bit = x[15];
        abs_x = sign_bit ? -x : x;

        // Piecewise linear: tanh(x) ≈ x for small x, saturates to ±1
        if (abs_x < 128)
            tanh_approx = x;  // Linear region
        else if (abs_x < 384)
            tanh_approx = sign_bit ? -218
                                   : 218;
        else
            tanh_approx = sign_bit ? -253
                                   : 253;
    end
endfunction


// Weight ROM (frozen weights - synthesized as LUTs/BRAM)
// In full implementation, these would be the actual trained weights
// For now, showing the structure with placeholder initialization

reg signed [15:0] input_weights [0:NUM_NEURONS-1][0:INPUT_DIM-1];
reg signed [15:0] biases [0:NUM_NEURONS-1];
reg signed [15:0] output_weights [0:OUTPUT_DIM-1][0:NUM_NEURONS-1];

// Weight initialization would be done via $readmemh or parameters
// initial begin
//     $readmemh("weights_input.hex", input_weights);
//     $readmemh("weights_bias.hex", biases);
//     $readmemh("weights_output.hex", output_weights);
// end

// Main state machine
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        valid_out <= 1'b0;
        comp_idx <= 0;
        neuron_idx <= 0;
    end else begin
        case (state)
            IDLE: begin
                valid_out <= 1'b0;
                if (valid_in) begin
                    state <= LOADING;
                    // Initialize accumulators with biases
                    for (int i = 0; i < NUM_NEURONS; i++) begin
                        neuron_accum[i] <= {{16{biases[i][15]}}, biases[i]};
                    end
                end
            end

            LOADING: begin
                // Store input pixel
                input_buffer[pixel_idx] <= pixel_in;

                // MAC for all neurons (parallel)
                for (int n = 0; n < NUM_NEURONS; n++) begin
                    neuron_accum[n] <= neuron_accum[n] +
                        (pixel_in * input_weights[n][pixel_idx]);
                end

                if (last_pixel) begin
                    state <= COMPUTE_NEURONS;
                    neuron_idx <= 0;
                end
            end

            COMPUTE_NEURONS: begin
                // Apply tanh activation to all neurons
                for (int n = 0; n < NUM_NEURONS; n++) begin
                    // Scale accumulator back and apply tanh
                    neuron_out[n] <= tanh_approx(neuron_accum[n][23:8]);
                end
                state <= COMPUTE_OUTPUT;
                comp_idx <= 0;
            end

            COMPUTE_OUTPUT: begin
                // Compute output scores (could be pipelined)
                for (int o = 0; o < OUTPUT_DIM; o++) begin
                    reg signed [31:0] sum;
                    sum = 0;
                    for (int n = 0; n < NUM_NEURONS; n++) begin
                        sum = sum + (neuron_out[n] * output_weights[o][n]);
                    end
                    class_scores[o] <= sum[23:8];
                end
                state <= DONE;
            end

            DONE: begin
                valid_out <= 1'b1;
                state <= IDLE;
            end
        endcase
    end
end


// Argmax for predicted class
always @(*) begin
    reg signed [15:0] max_score;
    max_score = class_scores[0];
    predicted_class = 4'd0;

    for (int i = 1; i < OUTPUT_DIM; i++) begin
        if (class_scores[i] > max_score) begin
            max_score = class_scores[i];
            predicted_class = i[3:0];
        end
    end
end

endmodule
