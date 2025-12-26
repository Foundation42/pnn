//============================================================================
// Crystal Neural Network - Synthesizable Verilog for FPGA
// Auto-generated from PyTorch crystal model
//
// Simplified version for Yosys synthesis:
// - 4 neurons (demo), 4 inputs, 4 outputs
// - Fixed-point Q4.4 arithmetic (8-bit)
// - Fully combinational (no state machine)
//============================================================================

module crystal_neuron #(
    parameter signed [7:0] W0 = 8'sd16,   // Weight for input 0 (1.0 in Q4.4)
    parameter signed [7:0] W1 = 8'sd8,    // Weight for input 1 (0.5)
    parameter signed [7:0] W2 = -8'sd8,   // Weight for input 2 (-0.5)
    parameter signed [7:0] W3 = 8'sd4,    // Weight for input 3 (0.25)
    parameter signed [7:0] BIAS = 8'sd0   // Bias
) (
    input wire signed [7:0] in0,
    input wire signed [7:0] in1,
    input wire signed [7:0] in2,
    input wire signed [7:0] in3,
    output wire signed [7:0] out
);
    // Multiply-accumulate (16-bit intermediate for precision)
    wire signed [15:0] sum;
    wire signed [15:0] prod0, prod1, prod2, prod3;

    assign prod0 = in0 * W0;
    assign prod1 = in1 * W1;
    assign prod2 = in2 * W2;
    assign prod3 = in3 * W3;

    // Sum with bias (shift by 4 to account for Q4.4 * Q4.4 = Q8.8)
    assign sum = (prod0 + prod1 + prod2 + prod3) >>> 4 + {BIAS, 8'b0} >>> 4;

    // Tanh approximation: clamp to [-1, 1] range (-16 to +16 in Q4.4)
    wire signed [15:0] clamped;
    assign clamped = (sum > 16'sd127) ? 16'sd127 :
                     (sum < -16'sd128) ? -16'sd128 : sum;

    // Piecewise linear tanh: output = x for |x| < 0.5, else saturate
    assign out = clamped[7:0];

endmodule

//============================================================================
// Top-level crystal network (4 inputs -> 4 neurons -> 4 outputs)
//============================================================================
module crystal_net_synth (
    input wire signed [7:0] in0,
    input wire signed [7:0] in1,
    input wire signed [7:0] in2,
    input wire signed [7:0] in3,

    output wire signed [7:0] out0,
    output wire signed [7:0] out1,
    output wire signed [7:0] out2,
    output wire signed [7:0] out3,

    output wire [1:0] predicted_class  // argmax of outputs
);

    //========================================
    // Hidden layer neurons (with frozen weights)
    //========================================
    wire signed [7:0] h0, h1, h2, h3;

    // Neuron 0: weights from training (example values)
    crystal_neuron #(
        .W0(8'sd12),   // 0.75
        .W1(-8'sd8),   // -0.5
        .W2(8'sd4),    // 0.25
        .W3(8'sd16),   // 1.0
        .BIAS(8'sd2)   // 0.125
    ) neuron0 (
        .in0(in0), .in1(in1), .in2(in2), .in3(in3),
        .out(h0)
    );

    // Neuron 1
    crystal_neuron #(
        .W0(-8'sd4),   // -0.25
        .W1(8'sd12),   // 0.75
        .W2(8'sd8),    // 0.5
        .W3(-8'sd6),   // -0.375
        .BIAS(-8'sd1)  // -0.0625
    ) neuron1 (
        .in0(in0), .in1(in1), .in2(in2), .in3(in3),
        .out(h1)
    );

    // Neuron 2
    crystal_neuron #(
        .W0(8'sd8),    // 0.5
        .W1(8'sd8),    // 0.5
        .W2(-8'sd12),  // -0.75
        .W3(8'sd4),    // 0.25
        .BIAS(8'sd0)
    ) neuron2 (
        .in0(in0), .in1(in1), .in2(in2), .in3(in3),
        .out(h2)
    );

    // Neuron 3
    crystal_neuron #(
        .W0(-8'sd16),  // -1.0
        .W1(8'sd4),    // 0.25
        .W2(8'sd6),    // 0.375
        .W3(8'sd10),   // 0.625
        .BIAS(8'sd3)
    ) neuron3 (
        .in0(in0), .in1(in1), .in2(in2), .in3(in3),
        .out(h3)
    );

    //========================================
    // Output layer neurons
    //========================================

    // Output 0
    crystal_neuron #(
        .W0(8'sd10), .W1(-8'sd6), .W2(8'sd8), .W3(-8'sd4),
        .BIAS(8'sd1)
    ) output0 (
        .in0(h0), .in1(h1), .in2(h2), .in3(h3),
        .out(out0)
    );

    // Output 1
    crystal_neuron #(
        .W0(-8'sd8), .W1(8'sd12), .W2(-8'sd4), .W3(8'sd6),
        .BIAS(-8'sd2)
    ) output1 (
        .in0(h0), .in1(h1), .in2(h2), .in3(h3),
        .out(out1)
    );

    // Output 2
    crystal_neuron #(
        .W0(8'sd4), .W1(8'sd4), .W2(8'sd10), .W3(-8'sd8),
        .BIAS(8'sd0)
    ) output2 (
        .in0(h0), .in1(h1), .in2(h2), .in3(h3),
        .out(out2)
    );

    // Output 3
    crystal_neuron #(
        .W0(-8'sd6), .W1(-8'sd4), .W2(8'sd6), .W3(8'sd14),
        .BIAS(8'sd2)
    ) output3 (
        .in0(h0), .in1(h1), .in2(h2), .in3(h3),
        .out(out3)
    );

    //========================================
    // Argmax: find class with highest output
    //========================================
    wire signed [7:0] max01, max23, max_all;
    wire [1:0] idx01, idx23;

    // Compare 0 vs 1
    assign max01 = (out0 > out1) ? out0 : out1;
    assign idx01 = (out0 > out1) ? 2'd0 : 2'd1;

    // Compare 2 vs 3
    assign max23 = (out2 > out3) ? out2 : out3;
    assign idx23 = (out2 > out3) ? 2'd2 : 2'd3;

    // Compare winners
    assign max_all = (max01 > max23) ? max01 : max23;
    assign predicted_class = (max01 > max23) ? idx01 : idx23;

endmodule
