"""
Single Seed Hybrid: Clone + Gap Focus

The best of both worlds:
1. Clone parent weights (inherit the foundation)
2. Fine-tune on hard samples (specialize on gaps)

Children inherit what we know AND learn to see what we miss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class HybridGapNetwork(nn.Module):
    """
    Clone the parent, then specialize on the gaps.
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

        self.register_buffer('positions', torch.zeros(max_neurons, 3))
        self.positions[0] = torch.tensor([50.0, 50.0, 50.0])

        self.input_weights = nn.Parameter(torch.randn(max_neurons, input_size) * 0.01)
        self.output_weights = nn.Parameter(torch.randn(max_neurons, output_size) * 0.01)

        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.alive_mask[0] = True

        self.register_buffer('activation_load', torch.zeros(max_neurons))

        self.total_splits = 0
        self.hard_samples = []

    @property
    def n_alive(self):
        return self.alive_mask.sum().item()

    def forward(self, x):
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
        """Find samples where we're most blind."""
        self.eval()
        sample_losses = []

        with torch.no_grad():
            sample_idx = 0
            for data, target in data_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)

                output = self(data)
                losses = F.cross_entropy(output, target, reduction='none')

                for i, loss in enumerate(losses):
                    sample_losses.append((sample_idx + i, loss.item()))

                sample_idx += len(data)

        sample_losses.sort(key=lambda x: -x[1])
        self.hard_samples = sample_losses[:top_k]

        self.train()
        return self.hard_samples

    def spawn_hybrid_neuron(self, train_dataset, device, finetune_steps=30):
        """
        Spawn a neuron that:
        1. CLONES the most overloaded parent (inherits foundation)
        2. Fine-tunes on hard samples (specializes on gaps)
        """
        empty_slots = torch.where(~self.alive_mask)[0]
        if len(empty_slots) == 0:
            return False

        child_idx = empty_slots[0].item()

        # Find the most overloaded parent
        alive_idx = torch.where(self.alive_mask)[0]
        loads = self.activation_load[alive_idx]
        parent_local_idx = loads.argmax()
        parent_idx = alive_idx[parent_local_idx].item()

        # Step 1: CLONE parent weights (inherit foundation)
        with torch.no_grad():
            self.input_weights.data[child_idx] = self.input_weights.data[parent_idx].clone()
            self.output_weights.data[child_idx] = self.output_weights.data[parent_idx].clone()

            # Small initial perturbation to break symmetry
            self.input_weights.data[child_idx] *= (1 + torch.randn_like(self.input_weights.data[child_idx]) * 0.01)
            self.output_weights.data[child_idx] *= (1 + torch.randn_like(self.output_weights.data[child_idx]) * 0.01)

            # Position near parent
            offset = torch.randn(3, device=device) * 5.0
            self.positions[child_idx] = self.positions[parent_idx] + offset

        # Get hard sample indices
        hard_indices = [idx for idx, _ in self.hard_samples[:500]]
        if len(hard_indices) < 10:
            print(f"  Not enough hard samples, using clone only")
            self.alive_mask[child_idx] = True
            self.activation_load[child_idx] = 0
            self.activation_load[parent_idx] *= 0.5
            self.total_splits += 1
            return True

        # Step 2: Fine-tune child on hard samples
        hard_subset = Subset(train_dataset, hard_indices)
        hard_loader = DataLoader(hard_subset, batch_size=64, shuffle=True)

        # Activate child for fine-tuning
        self.alive_mask[child_idx] = True

        # Create optimizer just for fine-tuning
        # We'll manually zero gradients for non-child neurons
        finetune_optimizer = torch.optim.Adam([
            {'params': [self.input_weights], 'lr': 0.02},
            {'params': [self.output_weights], 'lr': 0.02}
        ])

        self.train()

        # Track child's improvement on gaps
        initial_gap_acc = None

        for step in range(finetune_steps):
            for data, target in hard_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)

                finetune_optimizer.zero_grad()

                # Forward through FULL network (all alive neurons)
                output = self(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                # BUT only update the child's weights
                with torch.no_grad():
                    # Mask gradients: only keep child's
                    if self.input_weights.grad is not None:
                        mask = torch.zeros(self.max_neurons, 1, device=device)
                        mask[child_idx] = 1.0
                        self.input_weights.grad *= mask
                    if self.output_weights.grad is not None:
                        mask = torch.zeros(self.max_neurons, 1, device=device)
                        mask[child_idx] = 1.0
                        self.output_weights.grad *= mask

                finetune_optimizer.step()

            if step == 0:
                # Measure initial accuracy on gaps
                with torch.no_grad():
                    test_data = torch.stack([train_dataset[i][0] for i in hard_indices[:100]])
                    test_data = test_data.view(-1, 784).to(device)
                    test_target = torch.tensor([train_dataset[i][1] for i in hard_indices[:100]], device=device)
                    output = self(test_data)
                    pred = output.argmax(dim=1)
                    initial_gap_acc = pred.eq(test_target).float().mean().item()

        # Measure final accuracy on gaps
        self.eval()
        with torch.no_grad():
            test_data = torch.stack([train_dataset[i][0] for i in hard_indices[:100]])
            test_data = test_data.view(-1, 784).to(device)
            test_target = torch.tensor([train_dataset[i][1] for i in hard_indices[:100]], device=device)
            output = self(test_data)
            pred = output.argmax(dim=1)
            final_gap_acc = pred.eq(test_target).float().mean().item()
        self.train()

        # Reset loads
        self.activation_load[child_idx] = 0
        self.activation_load[parent_idx] *= 0.5

        self.total_splits += 1

        print(f"  Spawned hybrid neuron {child_idx} from parent {parent_idx}")
        print(f"    Gap accuracy: {initial_gap_acc*100:.1f}% → {final_gap_acc*100:.1f}%")

        return True

    def maybe_split(self, train_dataset, device):
        if self.n_alive >= self.max_neurons:
            return 0

        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) == 0:
            return 0

        loads = self.activation_load[alive_idx]
        max_load = loads.max().item()

        if max_load > self.split_threshold or self.n_alive < 5:
            if self.spawn_hybrid_neuron(train_dataset, device):
                return 1

        return 0


def train_hybrid_network(epochs=50, split_every=3):
    """Train the hybrid gap-targeting network on MNIST."""

    print("=" * 60)
    print("  HYBRID: CLONE + GAP FOCUS")
    print("  Inherit foundation, specialize on gaps")
    print("  Best of both worlds!")
    print("=" * 60)

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
    net = HybridGapNetwork(
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
            hard_samples = net.track_gaps(gap_track_loader, device)
            avg_hard_loss = sum(l for _, l in hard_samples[:100]) / 100
            print(f"  Average loss on 100 hardest samples: {avg_hard_loss:.3f}")

            n_splits = net.maybe_split(train_dataset, device)

            # Recreate optimizer
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
    train_hybrid_network(epochs=50, split_every=3)
