/*
 * Crystal Neural Network - CUDA Benchmark
 *
 * Tests GPU inference speed of compiled crystal network.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Include the crystal network constants
#include "crystal_net.c"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Device arrays for weights (in global memory for this benchmark)
__device__ float d_frozen_input_weights[NUM_FROZEN * INPUT_DIM];
__device__ float d_frozen_biases[NUM_FROZEN];
__device__ float d_frozen_output_weights[OUTPUT_DIM * NUM_FROZEN];
__device__ int d_frozen_indices[NUM_FROZEN];

__device__ float d_active_input_weights[NUM_ACTIVE * INPUT_DIM];
__device__ float d_active_biases[NUM_ACTIVE];
__device__ float d_active_output_weights[OUTPUT_DIM * NUM_ACTIVE];
__device__ int d_active_indices[NUM_ACTIVE];

__device__ float d_interactions[NUM_NEURONS * NUM_NEURONS];

// CUDA kernel for batch inference
__global__ void crystal_forward_kernel(
    const float* __restrict__ inputs,   // [batch_size, INPUT_DIM]
    float* __restrict__ outputs,         // [batch_size, OUTPUT_DIM]
    int batch_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const float* input = inputs + batch_idx * INPUT_DIM;
    float* output = outputs + batch_idx * OUTPUT_DIM;

    // Local arrays for this sample
    float activations[NUM_NEURONS];
    float modulated[NUM_NEURONS];

    // Compute frozen neuron activations
    for (int i = 0; i < NUM_FROZEN; i++) {
        float sum = d_frozen_biases[i];
        for (int j = 0; j < INPUT_DIM; j++) {
            sum += d_frozen_input_weights[i * INPUT_DIM + j] * input[j];
        }
        activations[d_frozen_indices[i]] = tanhf(sum);
    }

    // Compute active neuron activations
    for (int i = 0; i < NUM_ACTIVE; i++) {
        float sum = d_active_biases[i];
        for (int j = 0; j < INPUT_DIM; j++) {
            sum += d_active_input_weights[i * INPUT_DIM + j] * input[j];
        }
        activations[d_active_indices[i]] = tanhf(sum);
    }

    // Apply neuron interactions (modulation)
    for (int i = 0; i < NUM_NEURONS; i++) {
        float interaction_sum = 0.0f;
        for (int j = 0; j < NUM_NEURONS; j++) {
            interaction_sum += activations[j] * d_interactions[i * NUM_NEURONS + j];
        }
        modulated[i] = tanhf(activations[i] + 0.1f * interaction_sum);
    }

    // Compute output from all neurons
    for (int o = 0; o < OUTPUT_DIM; o++) {
        float sum = 0.0f;
        for (int i = 0; i < NUM_FROZEN; i++) {
            sum += modulated[d_frozen_indices[i]] * d_frozen_output_weights[o * NUM_FROZEN + i];
        }
        for (int i = 0; i < NUM_ACTIVE; i++) {
            sum += modulated[d_active_indices[i]] * d_active_output_weights[o * NUM_ACTIVE + i];
        }
        output[o] = sum;
    }
}

// Copy weights to device
void init_device_weights() {
    // Flatten frozen input weights
    float* frozen_input_flat = (float*)malloc(NUM_FROZEN * INPUT_DIM * sizeof(float));
    for (int i = 0; i < NUM_FROZEN; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            frozen_input_flat[i * INPUT_DIM + j] = FROZEN_INPUT_WEIGHTS[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_frozen_input_weights, frozen_input_flat,
                                   NUM_FROZEN * INPUT_DIM * sizeof(float)));
    free(frozen_input_flat);

    CUDA_CHECK(cudaMemcpyToSymbol(d_frozen_biases, FROZEN_BIASES,
                                   NUM_FROZEN * sizeof(float)));

    // Flatten frozen output weights
    float* frozen_output_flat = (float*)malloc(OUTPUT_DIM * NUM_FROZEN * sizeof(float));
    for (int o = 0; o < OUTPUT_DIM; o++) {
        for (int i = 0; i < NUM_FROZEN; i++) {
            frozen_output_flat[o * NUM_FROZEN + i] = FROZEN_OUTPUT_WEIGHTS[o][i];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_frozen_output_weights, frozen_output_flat,
                                   OUTPUT_DIM * NUM_FROZEN * sizeof(float)));
    free(frozen_output_flat);

    CUDA_CHECK(cudaMemcpyToSymbol(d_frozen_indices, FROZEN_INDICES,
                                   NUM_FROZEN * sizeof(int)));

    // Active weights
    float* active_input_flat = (float*)malloc(NUM_ACTIVE * INPUT_DIM * sizeof(float));
    for (int i = 0; i < NUM_ACTIVE; i++) {
        for (int j = 0; j < INPUT_DIM; j++) {
            active_input_flat[i * INPUT_DIM + j] = ACTIVE_INPUT_WEIGHTS[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_active_input_weights, active_input_flat,
                                   NUM_ACTIVE * INPUT_DIM * sizeof(float)));
    free(active_input_flat);

    CUDA_CHECK(cudaMemcpyToSymbol(d_active_biases, ACTIVE_BIASES,
                                   NUM_ACTIVE * sizeof(float)));

    float* active_output_flat = (float*)malloc(OUTPUT_DIM * NUM_ACTIVE * sizeof(float));
    for (int o = 0; o < OUTPUT_DIM; o++) {
        for (int i = 0; i < NUM_ACTIVE; i++) {
            active_output_flat[o * NUM_ACTIVE + i] = ACTIVE_OUTPUT_WEIGHTS[o][i];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_active_output_weights, active_output_flat,
                                   OUTPUT_DIM * NUM_ACTIVE * sizeof(float)));
    free(active_output_flat);

    CUDA_CHECK(cudaMemcpyToSymbol(d_active_indices, ACTIVE_INDICES,
                                   NUM_ACTIVE * sizeof(int)));

    // Flatten interactions
    float* interactions_flat = (float*)malloc(NUM_NEURONS * NUM_NEURONS * sizeof(float));
    for (int i = 0; i < NUM_NEURONS; i++) {
        for (int j = 0; j < NUM_NEURONS; j++) {
            interactions_flat[i * NUM_NEURONS + j] = INTERACTIONS[i][j];
        }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_interactions, interactions_flat,
                                   NUM_NEURONS * NUM_NEURONS * sizeof(float)));
    free(interactions_flat);
}

// Generate random input
void generate_random_inputs(float* inputs, int batch_size, int seed) {
    srand(seed);
    for (int i = 0; i < batch_size * INPUT_DIM; i++) {
        inputs[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

// Benchmark function
void benchmark_cuda(int batch_size, int num_iterations) {
    printf("\n=== CUDA Batch Inference (batch=%d) ===\n", batch_size);

    // Allocate host memory
    float* h_inputs = (float*)malloc(batch_size * INPUT_DIM * sizeof(float));
    float* h_outputs = (float*)malloc(batch_size * OUTPUT_DIM * sizeof(float));

    // Allocate device memory
    float *d_inputs, *d_outputs;
    CUDA_CHECK(cudaMalloc(&d_inputs, batch_size * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputs, batch_size * OUTPUT_DIM * sizeof(float)));

    // Generate random inputs
    generate_random_inputs(h_inputs, batch_size, 42);

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_inputs, h_inputs, batch_size * INPUT_DIM * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Configure kernel
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    // Warmup
    for (int i = 0; i < 10; i++) {
        crystal_forward_kernel<<<blocks, threads>>>(d_inputs, d_outputs, batch_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < num_iterations; i++) {
        crystal_forward_kernel<<<blocks, threads>>>(d_inputs, d_outputs, batch_size);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    int total_inferences = batch_size * num_iterations;
    float per_inference_us = milliseconds * 1000.0f / total_inferences;
    float throughput = total_inferences / (milliseconds / 1000.0f);

    printf("Total inferences: %d\n", total_inferences);
    printf("Total time: %.2f ms\n", milliseconds);
    printf("Per inference: %.3f µs\n", per_inference_us);
    printf("Throughput: %.0f inferences/sec\n", throughput);

    // Copy back results (just to verify)
    CUDA_CHECK(cudaMemcpy(h_outputs, d_outputs, batch_size * OUTPUT_DIM * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Print first few predictions
    printf("Sample predictions: ");
    for (int b = 0; b < 5 && b < batch_size; b++) {
        int pred = 0;
        float max_val = h_outputs[b * OUTPUT_DIM];
        for (int o = 1; o < OUTPUT_DIM; o++) {
            if (h_outputs[b * OUTPUT_DIM + o] > max_val) {
                max_val = h_outputs[b * OUTPUT_DIM + o];
                pred = o;
            }
        }
        printf("%d ", pred);
    }
    printf("...\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_inputs));
    CUDA_CHECK(cudaFree(d_outputs));
    free(h_inputs);
    free(h_outputs);
}

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║       Crystal Neural Network - CUDA Benchmark                    ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  GPU: %-56s  ║\n", prop.name);
    printf("║  Neurons: %3d  (Frozen: %3d, Active: %3d)                        ║\n",
           NUM_NEURONS, NUM_FROZEN, NUM_ACTIVE);
    printf("║  Frozen: %.1f%%                                                    ║\n",
           100.0 * NUM_FROZEN / NUM_NEURONS);
    printf("╚══════════════════════════════════════════════════════════════════╝\n");

    // Initialize weights on device
    printf("\nLoading weights to GPU...\n");
    init_device_weights();
    printf("Done!\n");

    // Run benchmarks at different batch sizes
    benchmark_cuda(64, 1000);
    benchmark_cuda(256, 1000);
    benchmark_cuda(1024, 500);
    benchmark_cuda(4096, 200);
    benchmark_cuda(16384, 100);
    benchmark_cuda(65536, 50);

    printf("\n✅ CUDA benchmark complete!\n\n");

    return 0;
}
