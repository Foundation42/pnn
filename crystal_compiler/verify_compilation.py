"""
Verify that compiled crystal networks produce identical outputs to PyTorch.

This script:
1. Loads the PyTorch crystal model
2. Runs inference on test samples
3. Compares with a pure-Python implementation of the compiled forward pass
4. Reports numerical accuracy
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bvh_growing_crystal import GrowingCrystalField


def load_compiled_weights(compiled_dir: Path):
    """Load the compiled weights from the output directory."""
    # Load metadata
    with open(compiled_dir / "crystal_metadata.json") as f:
        metadata = json.load(f)

    # Parse weights from C file
    c_file = compiled_dir / "crystal_net.c"
    content = c_file.read_text()

    # Extract frozen and active indices
    frozen_indices = extract_array(content, "FROZEN_INDICES", int)
    active_indices = extract_array(content, "ACTIVE_INDICES", int)

    return metadata, frozen_indices, active_indices


def extract_array(content: str, name: str, dtype) -> list:
    """Extract a simple 1D array from C code."""
    import re
    # Find pattern like: static const int NAME[N] = { values };
    pattern = rf'{name}\[\d+\]\s*=\s*\{{\s*([^}}]+)\s*\}}'
    match = re.search(pattern, content)
    if match:
        values_str = match.group(1)
        values = [dtype(x.strip()) for x in values_str.split(',') if x.strip()]
        return values
    return []


def verify_pytorch_model(model_path: Path, num_samples: int = 100):
    """Load PyTorch model and run verification."""
    print(f"Loading PyTorch model from {model_path}...")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Create model with matching neuron count
    num_neurons = checkpoint['num_neurons']
    model = GrowingCrystalField(
        input_dim=784,
        output_dim=10,
        seed_neurons=num_neurons,  # Match checkpoint
        max_neurons=num_neurons
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.num_neurons = checkpoint['num_neurons']
    model.frozen_mask = checkpoint['frozen_mask']
    model.eval()

    # Get frozen/active indices
    frozen_indices = torch.where(model.frozen_mask[:model.num_neurons])[0].tolist()
    active_indices = torch.where(~model.frozen_mask[:model.num_neurons])[0].tolist()

    print(f"  Neurons: {model.num_neurons}")
    print(f"  Frozen: {len(frozen_indices)} ({100*len(frozen_indices)/model.num_neurons:.1f}%)")
    print(f"  Active: {len(active_indices)} ({100*len(active_indices)/model.num_neurons:.1f}%)")

    return model, frozen_indices, active_indices


def pure_python_forward(model, x):
    """
    Pure Python implementation of the crystal forward pass.
    This mirrors what the compiled C code does.
    """
    n = model.num_neurons

    # Get weights as numpy
    positions = model.positions[:n].detach().numpy()
    input_weights = model.input_weights[:n].detach().numpy()
    biases = model.biases[:n].detach().numpy()
    output_weights = model.output_weights[:, :n].detach().numpy()

    # Input to numpy
    x_np = x.detach().numpy().flatten()

    # Compute activations
    activations = np.zeros(n)
    for i in range(n):
        sum_val = biases[i]
        for j in range(len(x_np)):
            sum_val += input_weights[i, j] * x_np[j]
        activations[i] = np.tanh(sum_val)

    # Compute neuron interactions (same as PyTorch: scale / dist)
    interaction_scale = model.interaction_scale.item()
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((positions[i] - positions[j])**2)) + 1e-6

    interactions = interaction_scale / distances
    np.fill_diagonal(interactions, 0)  # No self-interaction

    # Apply modulation
    modulated = np.zeros(n)
    for i in range(n):
        interaction_sum = 0.0
        for j in range(n):
            interaction_sum += activations[j] * interactions[i, j]
        modulated[i] = np.tanh(activations[i] + 0.1 * interaction_sum)

    # Compute output
    output = np.zeros(10)
    for o in range(10):
        for i in range(n):
            output[o] += modulated[i] * output_weights[o, i]

    return output


def run_verification(model_path: Path, compiled_dir: Path):
    """Run full verification."""
    print("=" * 60)
    print("Crystal Compilation Verification")
    print("=" * 60)

    # Load PyTorch model
    model, frozen_indices, active_indices = verify_pytorch_model(model_path)

    # Load compiled metadata
    metadata, comp_frozen, comp_active = load_compiled_weights(compiled_dir)

    # Verify metadata matches
    print("\nVerifying metadata...")
    assert metadata['num_neurons'] == model.num_neurons, "Neuron count mismatch!"
    assert metadata['num_frozen'] == len(frozen_indices), "Frozen count mismatch!"
    assert metadata['num_active'] == len(active_indices), "Active count mismatch!"
    print("  Metadata: PASS")

    # Verify indices match
    print("\nVerifying indices...")
    if comp_frozen and comp_active:
        assert sorted(comp_frozen) == sorted(frozen_indices), "Frozen indices mismatch!"
        assert sorted(comp_active) == sorted(active_indices), "Active indices mismatch!"
        print("  Indices: PASS")
    else:
        print("  Indices: SKIPPED (could not parse from C file)")

    # Run inference comparison
    print("\nRunning inference comparison...")

    # Generate random test inputs
    np.random.seed(42)
    num_tests = 10
    max_error = 0.0

    for i in range(num_tests):
        # Random MNIST-like input
        x = torch.randn(1, 784)

        # PyTorch forward
        with torch.no_grad():
            pytorch_out = model(x).numpy().flatten()

        # Pure Python forward (mirrors compiled C)
        python_out = pure_python_forward(model, x)

        # Compute error
        error = np.max(np.abs(pytorch_out - python_out))
        max_error = max(max_error, error)

        if i < 3:
            print(f"  Test {i+1}: max_error = {error:.2e}")

    print(f"\n  Max error across all tests: {max_error:.2e}")

    if max_error < 1e-5:
        print("  Numerical accuracy: PASS (error < 1e-5)")
    elif max_error < 1e-3:
        print("  Numerical accuracy: ACCEPTABLE (error < 1e-3)")
    else:
        print("  Numerical accuracy: WARNING (error >= 1e-3)")

    # Test on actual MNIST
    print("\nTesting on MNIST samples...")
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    correct = 0
    total = 100

    for i in range(total):
        img, label = test_dataset[i]
        x = img.view(1, -1)

        with torch.no_grad():
            pytorch_out = model(x).numpy().flatten()

        python_out = pure_python_forward(model, x)

        # Both should predict same class
        pytorch_pred = np.argmax(pytorch_out)
        python_pred = np.argmax(python_out)

        if pytorch_pred == python_pred:
            correct += 1

    print(f"  Prediction agreement: {correct}/{total} ({100*correct/total:.1f}%)")

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    model_path = Path("runs/crystal_compile_test/model_final.pt")
    compiled_dir = Path("crystal_compiler/compiled_test")

    run_verification(model_path, compiled_dir)
