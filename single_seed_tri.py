"""
Triangular Neural Network

Start from THREE seeds in a triangle, each splits into THREE.
Triangles all the way down - fractal exploration of the loss landscape.

3 → 5 → 7 → 9... fast parallel coverage!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SingleSeedNetwork(nn.Module):
    """
    A neural network that grows from a SINGLE seed neuron.

    The purest expression: one point → crystalline structure
    """

    def __init__(self,
                 input_size: int = 784,
                 output_size: int = 10,
                 max_neurons: int = 100,
                 split_threshold: float = 0.7):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.max_neurons = max_neurons
        self.split_threshold = split_threshold

        # Pre-allocate for growth
        # Start with THREE seeds in a triangle
        self.register_buffer('positions', torch.zeros(max_neurons, 3))
        # Triangle formation in the center of the volume
        self.positions[0] = torch.tensor([50.0, 60.0, 50.0])  # Top of triangle
        self.positions[1] = torch.tensor([40.0, 40.0, 50.0])  # Bottom left
        self.positions[2] = torch.tensor([60.0, 40.0, 50.0])  # Bottom right

        # Learnable weights
        self.input_weights = nn.Parameter(torch.randn(max_neurons, input_size) * 0.01)
        self.output_weights = nn.Parameter(torch.randn(max_neurons, output_size) * 0.01)

        # Alive mask - first THREE neurons alive initially
        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.alive_mask[0:3] = True

        # Load tracking
        self.register_buffer('activation_load', torch.zeros(max_neurons))

        self.total_splits = 0

    @property
    def n_alive(self):
        return self.alive_mask.sum().item()

    def forward(self, x):
        """Forward pass through alive neurons only."""
        alive_idx = torch.where(self.alive_mask)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            return torch.zeros(x.shape[0], self.output_size, device=x.device)

        # Input projection to each alive neuron
        # Each neuron has its own input weights
        activations = F.relu(x @ self.input_weights[alive_idx].T)

        # Track load (activation magnitude per neuron)
        with torch.no_grad():
            load = activations.abs().mean(dim=0)
            # Exponential moving average
            self.activation_load[alive_idx] = (
                0.9 * self.activation_load[alive_idx] + 0.1 * load
            )

        # Output: weighted sum
        output = activations @ self.output_weights[alive_idx]

        return output

    def should_split(self, neuron_idx):
        """Check if neuron is overloaded."""
        return self.activation_load[neuron_idx] > self.split_threshold

    def split_neuron(self, parent_idx):
        """Split an overloaded neuron into THREE - triangular exploration!"""
        # Need 2 empty slots for 2 children (parent stays)
        empty_slots = torch.where(~self.alive_mask)[0]
        if len(empty_slots) < 2:
            return False

        child1_idx = empty_slots[0].item()
        child2_idx = empty_slots[1].item()
        parent_idx = parent_idx.item() if torch.is_tensor(parent_idx) else parent_idx

        with torch.no_grad():
            parent_input_w = self.input_weights.data[parent_idx]
            parent_output_w = self.output_weights.data[parent_idx]
            noise_scale = 0.01

            # Child 1: offset in one direction
            offset1 = torch.randn(3, device=self.positions.device) * 5.0
            self.positions[child1_idx] = self.positions[parent_idx] + offset1
            self.input_weights.data[child1_idx] = parent_input_w * (1 + torch.randn_like(parent_input_w) * noise_scale)
            self.output_weights.data[child1_idx] = parent_output_w * (1 + torch.randn_like(parent_output_w) * noise_scale)
            self.alive_mask[child1_idx] = True
            self.activation_load[child1_idx] = 0

            # Child 2: offset in another direction
            offset2 = torch.randn(3, device=self.positions.device) * 5.0
            self.positions[child2_idx] = self.positions[parent_idx] + offset2
            self.input_weights.data[child2_idx] = parent_input_w * (1 + torch.randn_like(parent_input_w) * noise_scale)
            self.output_weights.data[child2_idx] = parent_output_w * (1 + torch.randn_like(parent_output_w) * noise_scale)
            self.alive_mask[child2_idx] = True
            self.activation_load[child2_idx] = 0

            # Reset parent load too
            self.activation_load[parent_idx] = 0

        self.total_splits += 1
        print(f"✂️  Tri-split! Neuron {parent_idx} → {parent_idx}, {child1_idx}, {child2_idx}. Total: {self.n_alive} neurons")
        return True

    def maybe_split(self, force=False):
        """Check all neurons for splitting."""
        alive_idx = torch.where(self.alive_mask)[0]

        if len(alive_idx) == 0:
            return 0

        n_splits = 0

        # Find most overloaded neuron
        loads = self.activation_load[alive_idx]
        max_load_local_idx = loads.argmax()
        max_load_idx = alive_idx[max_load_local_idx]
        max_load = loads[max_load_local_idx]

        if max_load > self.split_threshold or (force and self.n_alive < self.max_neurons):
            if self.split_neuron(max_load_idx):
                n_splits += 1

        return n_splits


def train_single_seed(epochs=30, split_every=3):
    """Train the single seed network on MNIST."""

    print("=" * 60)
    print("  TRIANGULAR NEURAL NETWORK")
    print("  3 seeds → tri-split → crystalline structure")
    print("  Triangles all the way down!")
    print("=" * 60)

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Create network with ONE seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SingleSeedNetwork(
        input_size=784,
        output_size=10,
        max_neurons=50,
        split_threshold=0.5
    ).to(device)

    print(f"\nStarting with {net.n_alive} seed neurons (triangle)")
    print(f"Max neurons: {net.max_neurons}")
    print(f"Device: {device}")
    print()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        n_batches = 0

        for data, target in train_loader:
            data = data.view(-1, 784).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Check for splits
        n_splits = 0
        if (epoch + 1) % split_every == 0:
            # Force split in early epochs to get growth started
            force = (epoch < epochs // 2 and net.n_alive < 20)
            n_splits = net.maybe_split(force=force)

        # Test accuracy
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)
                output = net(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        acc = 100 * correct / total

        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Loss: {total_loss/n_batches:.4f} | "
              f"Acc: {acc:.1f}% | "
              f"Neurons: {net.n_alive} | "
              f"Splits: {n_splits}")

    print()
    print("=" * 60)
    print(f"  Final: {net.n_alive} neurons grown from 3 seeds!")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  Total splits: {net.total_splits}")
    print("=" * 60)

    return net


if __name__ == "__main__":
    train_single_seed(epochs=50, split_every=3)  # More epochs, less frequent splits
