//============================================================================
// Crystal Neural Network - Testbench
//============================================================================
`timescale 1ns / 1ps

module crystal_net_tb;

parameter INPUT_DIM = 784;
parameter OUTPUT_DIM = 10;
parameter DATA_WIDTH = 16;

reg clk;
reg rst_n;
reg valid_in;
reg signed [DATA_WIDTH-1:0] pixel_in;
reg [$clog2(INPUT_DIM)-1:0] pixel_idx;
reg last_pixel;

wire valid_out;
wire signed [DATA_WIDTH-1:0] class_scores [0:OUTPUT_DIM-1];
wire [3:0] predicted_class;

// Instantiate DUT
crystal_net #(
    .INPUT_DIM(INPUT_DIM),
    .OUTPUT_DIM(OUTPUT_DIM),
    .DATA_WIDTH(DATA_WIDTH)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .pixel_in(pixel_in),
    .pixel_idx(pixel_idx),
    .last_pixel(last_pixel),
    .valid_out(valid_out),
    .class_scores(class_scores),
    .predicted_class(predicted_class)
);

// Clock generation
initial begin
    clk = 0;
    forever #5 clk = ~clk;  // 100 MHz
end

// Test stimulus
initial begin
    $display("Crystal Neural Network FPGA Testbench");
    $display("=====================================");

    // Reset
    rst_n = 0;
    valid_in = 0;
    pixel_in = 0;
    pixel_idx = 0;
    last_pixel = 0;

    #100;
    rst_n = 1;
    #20;

    // Feed a test image (random values)
    $display("Feeding test image...");
    valid_in = 1;

    for (int i = 0; i < INPUT_DIM; i++) begin
        pixel_idx = i;
        pixel_in = $random % 256 - 128;  // Random pixel value
        last_pixel = (i == INPUT_DIM - 1);
        #10;
    end

    valid_in = 0;

    // Wait for result
    wait(valid_out);
    #10;

    $display("Prediction complete!");
    $display("Predicted class: %d", predicted_class);
    $display("Class scores:");
    for (int i = 0; i < OUTPUT_DIM; i++) begin
        $display("  Class %d: %d", i, class_scores[i]);
    end

    #100;
    $finish;
end

endmodule
