/*
 * Crystal Neural Network - CUDA Kernel
 * Compiled from PyTorch crystal model
 *
 * Frozen: 44 neurons
 * Active: 20 neurons
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INPUT_DIM 784
#define OUTPUT_DIM 10
#define NUM_NEURONS 64

// Weights stored in constant memory for fast access
__constant__ float d_input_weights[NUM_NEURONS * INPUT_DIM];
__constant__ float d_biases[NUM_NEURONS];
__constant__ float d_output_weights[OUTPUT_DIM * NUM_NEURONS];

// CUDA kernel for batch inference
__global__ void crystal_forward_kernel(
    const float* __restrict__ inputs,  // [batch_size, INPUT_DIM]
    float* __restrict__ outputs,        // [batch_size, OUTPUT_DIM]
    int batch_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const float* input = inputs + batch_idx * INPUT_DIM;
    float* output = outputs + batch_idx * OUTPUT_DIM;

    // Compute neuron activations
    float activations[NUM_NEURONS];
    for (int n = 0; n < NUM_NEURONS; n++) {
        float sum = d_biases[n];
        for (int i = 0; i < INPUT_DIM; i++) {
            sum += d_input_weights[n * INPUT_DIM + i] * input[i];
        }
        activations[n] = tanhf(sum);
    }

    // Compute outputs
    for (int o = 0; o < OUTPUT_DIM; o++) {
        float sum = 0.0f;
        for (int n = 0; n < NUM_NEURONS; n++) {
            sum += activations[n] * d_output_weights[o * NUM_NEURONS + n];
        }
        output[o] = sum;
    }
}

// Host function to run inference
void crystal_inference_cuda(
    const float* h_inputs,
    float* h_outputs,
    int batch_size
) {
    float *d_inputs, *d_outputs;

    cudaMalloc(&d_inputs, batch_size * INPUT_DIM * sizeof(float));
    cudaMalloc(&d_outputs, batch_size * OUTPUT_DIM * sizeof(float));

    cudaMemcpy(d_inputs, h_inputs, batch_size * INPUT_DIM * sizeof(float),
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    crystal_forward_kernel<<<blocks, threads>>>(d_inputs, d_outputs, batch_size);

    cudaMemcpy(h_outputs, d_outputs, batch_size * OUTPUT_DIM * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_inputs);
    cudaFree(d_outputs);
}
