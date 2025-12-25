"""
Single Seed with Gap-Targeting Splits

The key insight from knowledge-attractors: NEW KNOWLEDGE LIES IN THE GAPS.

When we split, don't clone the parent. Instead:
1. Find where the network is BLIND (high-loss samples)
2. Initialize the child with FRESH weights targeting the gap
3. The child sees what the parent can't

This should match or beat 10-seed performance from a single starting point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class GapTargetingNetwork(nn.Module):
    """
    A neural network that grows by targeting gaps in its understanding.

    Each new neuron is born with fresh eyes, trained to see what
    the existing network misses.
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
        self.register_buffer('positions', torch.zeros(max_neurons, 3))
        self.positions[0] = torch.tensor([50.0, 50.0, 50.0])

        # Learnable weights
        self.input_weights = nn.Parameter(torch.randn(max_neurons, input_size) * 0.01)
        self.output_weights = nn.Parameter(torch.randn(max_neurons, output_size) * 0.01)

        # Alive mask
        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.alive_mask[0] = True

        # Load tracking
        self.register_buffer('activation_load', torch.zeros(max_neurons))

        self.total_splits = 0

        # Gap tracking: store indices and losses of hard samples
        self.hard_samples = []  # (index, loss) tuples from last epoch

    @property
    def n_alive(self):
        return self.alive_mask.sum().item()

    def forward(self, x):
        """Forward pass through alive neurons only."""
        alive_idx = torch.where(self.alive_mask)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            return torch.zeros(x.shape[0], self.output_size, device=x.device)

        activations = F.relu(x @ self.input_weights[alive_idx].T)

        with torch.no_grad():
            load = activations.abs().mean(dim=0)
            self.activation_load[alive_idx] = (
                0.9 * self.activation_load[alive_idx] + 0.1 * load
            )

        output = activations @ self.output_weights[alive_idx]
        return output

    def track_gaps(self, data_loader, device, top_k=1000):
        """Find the samples where we're most blind (highest loss)."""
        self.eval()
        sample_losses = []

        with torch.no_grad():
            sample_idx = 0
            for data, target in data_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)

                output = self(data)
                # Per-sample loss
                losses = F.cross_entropy(output, target, reduction='none')

                for i, loss in enumerate(losses):
                    sample_losses.append((sample_idx + i, loss.item()))

                sample_idx += len(data)

        # Sort by loss (highest first) - these are the gaps
        sample_losses.sort(key=lambda x: -x[1])
        self.hard_samples = sample_losses[:top_k]

        self.train()
        return self.hard_samples

    def spawn_gap_neuron(self, train_dataset, device, warmup_steps=20):
        """
        Spawn a new neuron specifically trained on the gaps.

        Instead of cloning parent + noise, we:
        1. Create fresh random weights
        2. Briefly train ONLY on the hard samples
        3. This neuron learns to see what others miss
        """
        empty_slots = torch.where(~self.alive_mask)[0]
        if len(empty_slots) == 0:
            return False

        child_idx = empty_slots[0].item()

        # Fresh random initialization - NOT a clone!
        with torch.no_grad():
            self.input_weights.data[child_idx] = torch.randn(self.input_size, device=device) * 0.1
            self.output_weights.data[child_idx] = torch.randn(self.output_size, device=device) * 0.1

            # Position: spread across the volume based on which child this is
            t = self.total_splits / self.max_neurons
            self.positions[child_idx] = torch.tensor([
                20 + t * 60,  # Spread along x
                30 + (self.total_splits % 5) * 10,  # Vary y
                30 + ((self.total_splits // 5) % 5) * 10  # Vary z
            ], device=device)

        # Get hard sample indices
        hard_indices = [idx for idx, _ in self.hard_samples[:500]]
        if len(hard_indices) < 10:
            print(f"  Not enough hard samples ({len(hard_indices)}), using random init")
            self.alive_mask[child_idx] = True
            self.total_splits += 1
            return True

        # Create a mini-dataset of just the hard samples
        hard_subset = Subset(train_dataset, hard_indices)
        hard_loader = DataLoader(hard_subset, batch_size=64, shuffle=True)

        # Temporarily freeze all other neurons
        # We'll train ONLY the new neuron on hard samples
        old_alive = self.alive_mask.clone()
        self.alive_mask[:] = False
        self.alive_mask[child_idx] = True

        # Quick optimizer just for the new neuron's weights
        child_params = [
            self.input_weights,  # Will only update child_idx row
            self.output_weights
        ]
        child_optimizer = torch.optim.Adam(child_params, lr=0.05)

        # Warmup: train new neuron on gaps
        self.train()
        for step in range(warmup_steps):
            for data, target in hard_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)

                child_optimizer.zero_grad()

                # Forward through just the child
                activation = F.relu(data @ self.input_weights[child_idx])
                output = activation.unsqueeze(1) @ self.output_weights[child_idx].unsqueeze(0)
                output = output.squeeze(1)

                loss = F.cross_entropy(output, target)
                loss.backward()

                # Only update the child's weights
                with torch.no_grad():
                    if self.input_weights.grad is not None:
                        grad_mask = torch.zeros_like(self.input_weights.grad)
                        grad_mask[child_idx] = 1.0
                        self.input_weights.grad *= grad_mask
                    if self.output_weights.grad is not None:
                        grad_mask = torch.zeros_like(self.output_weights.grad)
                        grad_mask[child_idx] = 1.0
                        self.output_weights.grad *= grad_mask

                child_optimizer.step()

                if step >= warmup_steps - 1:
                    break
            if step >= warmup_steps - 1:
                break

        # Restore all neurons (including new one)
        self.alive_mask = old_alive
        self.alive_mask[child_idx] = True
        self.activation_load[child_idx] = 0

        self.total_splits += 1

        # Verify the child learned something
        self.eval()
        with torch.no_grad():
            test_data = torch.stack([train_dataset[i][0] for i in hard_indices[:100]])
            test_data = test_data.view(-1, 784).to(device)
            test_target = torch.tensor([train_dataset[i][1] for i in hard_indices[:100]], device=device)

            # Child-only prediction
            activation = F.relu(test_data @ self.input_weights[child_idx])
            output = activation.unsqueeze(1) @ self.output_weights[child_idx].unsqueeze(0)
            output = output.squeeze(1)
            pred = output.argmax(dim=1)
            child_acc = pred.eq(test_target).float().mean().item()

        self.train()

        print(f"  Spawned gap-targeting neuron {child_idx}")
        print(f"    Trained on {len(hard_indices)} hard samples")
        print(f"    Child accuracy on gaps: {child_acc*100:.1f}%")

        return True

    def maybe_split(self, train_dataset, device):
        """Check if we should spawn a new gap-targeting neuron."""
        if self.n_alive >= self.max_neurons:
            return 0

        # Find most overloaded neuron
        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) == 0:
            return 0

        loads = self.activation_load[alive_idx]
        max_load = loads.max().item()

        if max_load > self.split_threshold or self.n_alive < 5:
            if self.spawn_gap_neuron(train_dataset, device):
                return 1

        return 0


def train_gap_network(epochs=50, split_every=5):
    """Train the gap-targeting network on MNIST."""

    print("=" * 60)
    print("  GAP-TARGETING NEURAL NETWORK")
    print("  New knowledge lies in the gaps!")
    print("  Children are born with fresh eyes, trained on what we miss")
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
    gap_track_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GapTargetingNetwork(
        input_size=784,
        output_size=10,
        max_neurons=50,
        split_threshold=0.5
    ).to(device)

    print(f"\nStarting with {net.n_alive} seed neuron")
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
        if (epoch + 1) % split_every == 0 and net.n_alive < 20:
            print(f"\n--- Epoch {epoch+1}: Searching for gaps ---")
            # Find the gaps first
            hard_samples = net.track_gaps(gap_track_loader, device)
            avg_hard_loss = sum(l for _, l in hard_samples[:100]) / 100
            print(f"  Average loss on top 100 hard samples: {avg_hard_loss:.3f}")

            # Spawn a neuron targeting the gaps
            n_splits = net.maybe_split(train_dataset, device)

            # Recreate optimizer to include new parameters
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

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
    print(f"  Final: {net.n_alive} neurons grown from 1 seed!")
    print(f"  Accuracy: {acc:.1f}%")
    print(f"  Total splits: {net.total_splits}")
    print("=" * 60)

    return net


if __name__ == "__main__":
    train_gap_network(epochs=30, split_every=3)
