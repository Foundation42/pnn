"""
Benchmark PyTorch CUDA for comparison with compiled CUDA.
"""

import torch
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from bvh_growing_crystal import GrowingCrystalField


def benchmark_pytorch_cuda(model_path: str):
    """Benchmark PyTorch CUDA inference speed."""

    device = torch.device('cuda')

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
    model = model.to(device)

    frozen_count = model.frozen_mask.sum().item()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       Crystal Neural Network - PyTorch CUDA Benchmark            ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  GPU: {torch.cuda.get_device_name(0):56s}  ║")
    print(f"║  Neurons: {num_neurons:3d}  (Frozen: {frozen_count:3.0f}, Active: {num_neurons - frozen_count:3.0f})                        ║")
    print(f"║  Frozen: {100*frozen_count/num_neurons:.1f}%                                                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Batch benchmarks
    batch_configs = [
        (64, 1000),
        (256, 1000),
        (1024, 500),
        (4096, 200),
        (16384, 100),
        (65536, 50),
    ]

    for batch_size, num_iterations in batch_configs:
        print(f"\n=== PyTorch CUDA Batch Inference (batch={batch_size}) ===")

        # Generate inputs on GPU
        x = torch.randn(batch_size, 784, device=device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        # Timed run
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)
        end.record()

        torch.cuda.synchronize()
        milliseconds = start.elapsed_time(end)

        total_inferences = batch_size * num_iterations
        per_inference_us = milliseconds * 1000.0 / total_inferences
        throughput = total_inferences / (milliseconds / 1000.0)

        print(f"Total inferences: {total_inferences}")
        print(f"Total time: {milliseconds:.2f} ms")
        print(f"Per inference: {per_inference_us:.3f} µs")
        print(f"Throughput: {throughput:.0f} inferences/sec")

    print("\n✅ PyTorch CUDA benchmark complete!")


if __name__ == "__main__":
    model_path = "runs/crystal_compile_test/model_final.pt"
    benchmark_pytorch_cuda(model_path)
