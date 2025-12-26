"""
Test compiled crystal on actual MNIST test set.
Generates test vectors and compares C predictions with ground truth.
"""

import torch
import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from torchvision import datasets, transforms


def generate_c_test_program(compiled_dir: Path, test_inputs: np.ndarray,
                            test_labels: np.ndarray) -> str:
    """Generate a C program that tests MNIST samples."""

    num_samples = len(test_labels)

    code = f'''
#include <stdio.h>
#include "crystal_net.c"

// Test data (normalized MNIST samples)
static const float TEST_INPUTS[{num_samples}][{784}] = {{
'''

    for i, inp in enumerate(test_inputs):
        weights_str = ", ".join(f"{x:.6f}f" for x in inp)
        code += f"    {{ {weights_str} }},\n"

    code += f'''
}};

static const int TEST_LABELS[{num_samples}] = {{
    {', '.join(str(l) for l in test_labels)}
}};

int main() {{
    int correct = 0;

    for (int i = 0; i < {num_samples}; i++) {{
        int pred = crystal_predict(TEST_INPUTS[i]);
        if (pred == TEST_LABELS[i]) {{
            correct++;
        }}
    }}

    printf("Accuracy: %d/%d (%.1f%%)\\n", correct, {num_samples},
           100.0 * correct / {num_samples});

    return 0;
}}
'''
    return code


def test_mnist_accuracy(compiled_dir: Path, num_samples: int = 1000):
    """Test compiled model on MNIST."""

    print(f"Loading MNIST test set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Get samples
    indices = np.random.permutation(len(test_dataset))[:num_samples]
    test_inputs = []
    test_labels = []

    for idx in indices:
        img, label = test_dataset[idx]
        test_inputs.append(img.numpy().flatten())
        test_labels.append(label)

    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)

    print(f"Generating C test program with {num_samples} samples...")

    # Generate test program
    test_code = generate_c_test_program(compiled_dir, test_inputs, test_labels)

    # Write to temp file (use absolute paths)
    compiled_dir = compiled_dir.resolve()
    test_file = compiled_dir / "test_mnist.c"
    test_file.write_text(test_code)

    # Compile
    print("Compiling test program...")

    exe_file = compiled_dir / "test_mnist"
    result = subprocess.run(
        ["gcc", "-O3", "-march=native", "-ffast-math", "-o", str(exe_file),
         "test_mnist.c", "-lm"],
        cwd=str(compiled_dir),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return

    # Run
    print("Running MNIST accuracy test...")
    result = subprocess.run([str(exe_file)], capture_output=True, text=True)
    print(result.stdout)

    # Cleanup
    test_file.unlink()
    exe_file.unlink()


if __name__ == "__main__":
    # Test 32-neuron model
    print("=" * 60)
    print("Testing 32-neuron model (81% frozen)")
    print("=" * 60)
    test_mnist_accuracy(Path("crystal_compiler/compiled_test"), num_samples=1000)

    print()

    # Test 64-neuron model
    print("=" * 60)
    print("Testing 64-neuron model (69% frozen)")
    print("=" * 60)
    test_mnist_accuracy(Path("crystal_compiler/compiled_64"), num_samples=1000)
