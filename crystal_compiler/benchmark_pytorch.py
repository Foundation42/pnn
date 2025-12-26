"""
Benchmark PyTorch crystal network for comparison with compiled C.
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bvh_growing_crystal import GrowingCrystalField


def benchmark_pytorch(model_path: str, num_iterations: int = 10000):
    """Benchmark PyTorch inference speed."""

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    num_neurons = checkpoint['num_neurons']

    model = GrowingCrystalField(
        input_dim=784,
        output_dim=10,
        seed_neurons=num_neurons,
        max_neurons=num_neurons
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.num_neurons = num_neurons
    model.frozen_mask = checkpoint['frozen_mask']
    model.eval()

    frozen_count = model.frozen_mask.sum().item()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Crystal Neural Network - PyTorch Benchmark           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Neurons: {num_neurons:3d}  (Frozen: {frozen_count:3.0f}, Active: {num_neurons - frozen_count:3.0f})                ║")
    print("║  Input:   784  Output:  10                               ║")
    print(f"║  Frozen:  {100*frozen_count/num_neurons:.1f}%                                          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Generate random inputs
    np.random.seed(42)

    # Warmup
    print("\nWarming up...")
    for _ in range(100):
        x = torch.randn(1, 784)
        with torch.no_grad():
            _ = model(x)

    # Single inference benchmark
    print(f"\n=== Single Inference Benchmark ===")
    print(f"Iterations: {num_iterations}")

    start = time.perf_counter()
    for i in range(num_iterations):
        x = torch.randn(1, 784)
        with torch.no_grad():
            _ = model(x)
    end = time.perf_counter()

    total_ms = (end - start) * 1000
    per_inference_us = total_ms * 1000 / num_iterations
    throughput = num_iterations / (end - start)

    print(f"Total time: {total_ms:.2f} ms")
    print(f"Per inference: {per_inference_us:.2f} µs")
    print(f"Throughput: {throughput:.0f} inferences/sec")

    # Batch benchmarks
    for batch_size, num_batches in [(64, 100), (256, 50), (1000, 20)]:
        print(f"\n=== Batch Inference Benchmark ===")
        print(f"Batch size: {batch_size}")
        print(f"Num batches: {num_batches}")

        total_inferences = batch_size * num_batches

        start = time.perf_counter()
        for _ in range(num_batches):
            x = torch.randn(batch_size, 784)
            with torch.no_grad():
                _ = model(x)
        end = time.perf_counter()

        total_ms = (end - start) * 1000
        per_inference_us = total_ms * 1000 / total_inferences
        throughput = total_inferences / (end - start)

        print(f"Total inferences: {total_inferences}")
        print(f"Total time: {total_ms:.2f} ms")
        print(f"Per inference: {per_inference_us:.2f} µs")
        print(f"Throughput: {throughput:.0f} inferences/sec")

    print("\n✅ PyTorch benchmark complete!")

    return per_inference_us


if __name__ == "__main__":
    model_path = "runs/crystal_compile_test/model_final.pt"
    benchmark_pytorch(model_path)
