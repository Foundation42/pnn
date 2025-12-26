/*
 * Crystal Neural Network Benchmark
 *
 * Tests inference speed and correctness of compiled crystal network.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "crystal_net.c"  // Include the generated code directly

// Simple MNIST-like random input generator
void generate_random_input(float* input, int seed) {
    srand(seed);
    for (int i = 0; i < INPUT_DIM; i++) {
        // Normalized random values similar to MNIST
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

// Get current time in nanoseconds
long long get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// Benchmark single inference
void benchmark_single(int num_iterations) {
    float input[INPUT_DIM];
    float output[OUTPUT_DIM];

    printf("\n=== Single Inference Benchmark ===\n");
    printf("Iterations: %d\n", num_iterations);

    // Warmup
    for (int i = 0; i < 100; i++) {
        generate_random_input(input, i);
        crystal_forward(input, output);
    }

    // Timed run
    long long start = get_time_ns();

    for (int i = 0; i < num_iterations; i++) {
        generate_random_input(input, i);
        crystal_forward(input, output);
    }

    long long end = get_time_ns();
    double total_ms = (end - start) / 1e6;
    double per_inference_us = (end - start) / 1e3 / num_iterations;
    double throughput = num_iterations / (total_ms / 1000.0);

    printf("Total time: %.2f ms\n", total_ms);
    printf("Per inference: %.2f µs\n", per_inference_us);
    printf("Throughput: %.0f inferences/sec\n", throughput);
}

// Benchmark batch inference (simulated)
void benchmark_batch(int batch_size, int num_batches) {
    float* inputs = (float*)malloc(batch_size * INPUT_DIM * sizeof(float));
    float* outputs = (float*)malloc(batch_size * OUTPUT_DIM * sizeof(float));

    printf("\n=== Batch Inference Benchmark ===\n");
    printf("Batch size: %d\n", batch_size);
    printf("Num batches: %d\n", num_batches);

    // Generate all inputs
    for (int b = 0; b < batch_size; b++) {
        generate_random_input(inputs + b * INPUT_DIM, b);
    }

    // Warmup
    for (int b = 0; b < batch_size; b++) {
        crystal_forward(inputs + b * INPUT_DIM, outputs + b * OUTPUT_DIM);
    }

    // Timed run
    long long start = get_time_ns();

    for (int iter = 0; iter < num_batches; iter++) {
        for (int b = 0; b < batch_size; b++) {
            crystal_forward(inputs + b * INPUT_DIM, outputs + b * OUTPUT_DIM);
        }
    }

    long long end = get_time_ns();
    int total_inferences = batch_size * num_batches;
    double total_ms = (end - start) / 1e6;
    double per_inference_us = (end - start) / 1e3 / total_inferences;
    double throughput = total_inferences / (total_ms / 1000.0);

    printf("Total inferences: %d\n", total_inferences);
    printf("Total time: %.2f ms\n", total_ms);
    printf("Per inference: %.2f µs\n", per_inference_us);
    printf("Throughput: %.0f inferences/sec\n", throughput);

    free(inputs);
    free(outputs);
}

// Test prediction consistency
void test_predictions() {
    float input[INPUT_DIM];

    printf("\n=== Prediction Test ===\n");
    printf("Running 10 predictions with fixed seeds...\n\n");

    for (int seed = 0; seed < 10; seed++) {
        generate_random_input(input, seed * 12345);
        int prediction = crystal_predict(input);
        printf("  Seed %d: predicted class %d\n", seed, prediction);
    }
}

// Memory usage estimate
void print_memory_usage() {
    printf("\n=== Memory Usage ===\n");

    size_t frozen_input = NUM_FROZEN * INPUT_DIM * sizeof(float);
    size_t frozen_bias = NUM_FROZEN * sizeof(float);
    size_t frozen_output = OUTPUT_DIM * NUM_FROZEN * sizeof(float);
    size_t frozen_total = frozen_input + frozen_bias + frozen_output;

    size_t active_input = NUM_ACTIVE * INPUT_DIM * sizeof(float);
    size_t active_bias = NUM_ACTIVE * sizeof(float);
    size_t active_output = OUTPUT_DIM * NUM_ACTIVE * sizeof(float);
    size_t active_total = active_input + active_bias + active_output;

    size_t interactions = NUM_NEURONS * NUM_NEURONS * sizeof(float);

    size_t total = frozen_total + active_total + interactions;

    printf("Frozen weights:  %6.2f KB (%d neurons)\n", frozen_total / 1024.0, NUM_FROZEN);
    printf("Active weights:  %6.2f KB (%d neurons)\n", active_total / 1024.0, NUM_ACTIVE);
    printf("Interactions:    %6.2f KB (%dx%d matrix)\n", interactions / 1024.0, NUM_NEURONS, NUM_NEURONS);
    printf("─────────────────────────────\n");
    printf("Total:           %6.2f KB\n", total / 1024.0);
    printf("\nNote: Frozen weights are compile-time constants (in .rodata section)\n");
}

int main(int argc, char** argv) {
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     Crystal Neural Network - Compiled C Benchmark        ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Neurons: %3d  (Frozen: %3d, Active: %3d)                ║\n",
           NUM_NEURONS, NUM_FROZEN, NUM_ACTIVE);
    printf("║  Input:   %3d  Output: %3d                               ║\n",
           INPUT_DIM, OUTPUT_DIM);
    printf("║  Frozen:  %.1f%%                                          ║\n",
           100.0 * NUM_FROZEN / NUM_NEURONS);
    printf("╚══════════════════════════════════════════════════════════╝\n");

    print_memory_usage();
    test_predictions();

    // Run benchmarks
    benchmark_single(10000);
    benchmark_batch(64, 100);
    benchmark_batch(256, 50);
    benchmark_batch(1000, 20);

    printf("\n✅ Benchmark complete!\n\n");

    return 0;
}
