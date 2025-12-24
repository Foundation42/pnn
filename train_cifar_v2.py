"""
CIFAR-10 Growing Neural Field v2

Improvements:
1. More seed neurons (50)
2. Random injection - sprinkle new neurons in sparse/underperforming regions
3. More aggressive splitting
4. Deeper message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time


class GrowingFieldV2(nn.Module):
    """
    Improved growing field with:
    - Mitosis (splitting overloaded neurons)
    - Random injection (seeding new neurons in sparse regions)
    """

    def __init__(self,
                 volume_size=(100, 100, 100),
                 seed_neurons=50,
                 max_neurons=500,
                 input_size=3072,
                 output_size=10,
                 feature_dim=64,
                 connection_radius=20.0,
                 device=None):
        super().__init__()

        self.volume_size = torch.tensor(volume_size, dtype=torch.float32)
        self.seed_neurons = seed_neurons
        self.max_neurons = max_neurons
        self.input_size = input_size
        self.output_size = output_size
        self.feature_dim = feature_dim
        self.connection_radius = connection_radius

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Pre-allocate
        self.positions = nn.Parameter(torch.zeros(max_neurons, 3))
        self.input_weights = nn.Parameter(torch.zeros(max_neurons, input_size))
        self.features = nn.Parameter(torch.zeros(max_neurons, feature_dim))
        self.output_weights = nn.Parameter(torch.zeros(max_neurons, output_size))
        self.biases = nn.Parameter(torch.zeros(max_neurons))

        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.register_buffer('neuron_importance', torch.zeros(max_neurons))

        # Initialize seeds
        self._init_seeds(seed_neurons)

        self.total_splits = 0
        self.total_injections = 0

    def _init_seeds(self, n):
        """Initialize seed neurons spread across volume."""
        # Spread seeds across depth (x-axis)
        for i in range(n):
            # Position: spread across x, random y/z
            x = (i / n) * 80 + 10  # x: 10-90
            y = torch.rand(1).item() * 80 + 10
            z = torch.rand(1).item() * 80 + 10

            self.positions.data[i] = torch.tensor([x, y, z])
            self.input_weights.data[i] = torch.randn(self.input_size) * 0.02
            self.features.data[i] = torch.randn(self.feature_dim) * 0.1
            self.output_weights.data[i] = torch.randn(self.output_size) * 0.02
            self.alive_mask[i] = True

    @property
    def n_alive(self):
        return self.alive_mask.sum().item()

    def forward(self, x, n_iterations=3):
        batch_size = x.shape[0]
        device = x.device

        alive_idx = torch.where(self.alive_mask)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            return torch.zeros(batch_size, self.output_size, device=device)

        # Get parameters
        vol = self.volume_size.to(device)
        pos = self.positions[alive_idx]
        pos = torch.stack([
            torch.clamp(pos[:, 0], 0.1, vol[0].item() - 0.1),
            torch.clamp(pos[:, 1], 0.1, vol[1].item() - 0.1),
            torch.clamp(pos[:, 2], 0.1, vol[2].item() - 0.1),
        ], dim=1)

        feat = self.features[alive_idx]
        inp_w = self.input_weights[alive_idx]
        out_w = self.output_weights[alive_idx]
        bias = self.biases[alive_idx]

        # Distance-based connections
        dist = torch.cdist(pos, pos)
        conn_mask = (dist < self.connection_radius) & (dist > 0)

        # Connection weights from distance + feature similarity
        attenuation = torch.exp(-dist / self.connection_radius) * conn_mask.float()
        feat_norm = F.normalize(feat, dim=1, eps=1e-6)
        feat_sim = torch.clamp(torch.mm(feat_norm, feat_norm.T), -1, 1)
        conn_w = attenuation * (0.5 + 0.5 * feat_sim)
        conn_w = conn_w / (conn_w.sum(dim=1, keepdim=True) + 1e-6)

        # Input weighting by position (low x = more input)
        x_norm = pos[:, 0] / vol[0]
        input_weight = torch.exp(-2 * x_norm)
        input_weight = input_weight / (input_weight.sum() + 1e-6)

        # Project input
        activations = torch.mm(x, inp_w.T) * input_weight + bias

        # Message passing with residual
        for _ in range(n_iterations):
            msg = torch.mm(activations, conn_w.T)
            activations = F.relu(activations + 0.5 * msg)
            activations = torch.clamp(activations, max=50.0)

        # Track importance for growth decisions
        with torch.no_grad():
            importance = activations.abs().mean(dim=0)
            self.neuron_importance[alive_idx] = 0.9 * self.neuron_importance[alive_idx] + 0.1 * importance

        # Output weighting by position (high x = more output)
        output_weight = torch.exp(2 * (x_norm - 1))
        output_weight = output_weight / (output_weight.sum() + 1e-6)

        output = torch.mm(activations, out_w * output_weight.unsqueeze(1))

        return output

    def split_neurons(self, threshold=0.7):
        """Split high-importance neurons."""
        n_splits = 0
        alive_idx = torch.where(self.alive_mask)[0]

        if len(alive_idx) == 0:
            return 0

        importance = self.neuron_importance[alive_idx]
        max_imp = importance.max()

        if max_imp < 0.01:
            return 0

        norm_imp = importance / (max_imp + 1e-6)

        with torch.no_grad():
            for i, idx in enumerate(alive_idx):
                if norm_imp[i] > threshold and self.n_alive < self.max_neurons:
                    # Find empty slot
                    empty = torch.where(~self.alive_mask)[0]
                    if len(empty) == 0:
                        break

                    child_idx = empty[0].item()
                    parent_idx = idx.item()

                    # Child near parent
                    offset = torch.randn(3) * 5
                    self.positions.data[child_idx] = torch.clamp(
                        self.positions.data[parent_idx] + offset.to(self.device),
                        min=torch.tensor([1., 1., 1.], device=self.device),
                        max=self.volume_size.to(self.device) - 1
                    )

                    # Inherit with noise
                    noise = 0.1
                    self.input_weights.data[child_idx] = self.input_weights.data[parent_idx] + torch.randn_like(self.input_weights.data[parent_idx]) * noise
                    self.features.data[child_idx] = self.features.data[parent_idx] + torch.randn_like(self.features.data[parent_idx]) * noise
                    self.output_weights.data[child_idx] = self.output_weights.data[parent_idx] + torch.randn_like(self.output_weights.data[parent_idx]) * noise

                    self.alive_mask[child_idx] = True
                    self.neuron_importance[parent_idx] *= 0.5
                    n_splits += 1

        self.total_splits += n_splits
        return n_splits

    def inject_neurons(self, n_inject=5):
        """Inject new neurons in random/sparse regions."""
        n_injected = 0

        with torch.no_grad():
            alive_idx = torch.where(self.alive_mask)[0]
            alive_pos = self.positions[alive_idx].detach()

            for _ in range(n_inject):
                if self.n_alive >= self.max_neurons:
                    break

                empty = torch.where(~self.alive_mask)[0]
                if len(empty) == 0:
                    break

                new_idx = empty[0].item()

                # Random position, biased toward middle depth
                x = torch.rand(1).item() * 60 + 20  # 20-80
                y = torch.rand(1).item() * 80 + 10
                z = torch.rand(1).item() * 80 + 10

                # Check it's not too close to existing neurons
                new_pos = torch.tensor([x, y, z], device=self.device)
                if len(alive_pos) > 0:
                    dists = torch.norm(alive_pos.to(self.device) - new_pos, dim=1)
                    if dists.min() < 5:  # Too close, skip
                        continue

                self.positions.data[new_idx] = new_pos
                self.input_weights.data[new_idx] = torch.randn(self.input_size, device=self.device) * 0.02
                self.features.data[new_idx] = torch.randn(self.feature_dim, device=self.device) * 0.1
                self.output_weights.data[new_idx] = torch.randn(self.output_size, device=self.device) * 0.02
                self.alive_mask[new_idx] = True

                n_injected += 1

        self.total_injections += n_injected
        return n_injected

    def geometric_loss(self):
        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) < 2:
            return torch.tensor(0.0, device=self.device)

        pos = self.positions[alive_idx]

        # Spacing loss
        n_sample = min(200, len(alive_idx))
        i_idx = torch.randint(0, len(alive_idx), (n_sample,), device=self.device)
        j_idx = torch.randint(0, len(alive_idx), (n_sample,), device=self.device)
        valid = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]

        if len(i_idx) == 0:
            return torch.tensor(0.0, device=self.device)

        dists = torch.norm(pos[i_idx] - pos[j_idx], dim=1)
        spacing_loss = F.relu(3.0 - dists).mean()

        return spacing_loss * 0.1


def train_cifar_v2():
    print("=" * 70)
    print("  CIFAR-10 Growing Field v2")
    print("  More seeds + Random injection + Aggressive growth")
    print("=" * 70)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262))
    ])

    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=500, num_workers=2)

    # Model
    field = GrowingFieldV2(
        seed_neurons=50,
        max_neurons=400,
        input_size=3072,
        output_size=10,
        feature_dim=64,
        connection_radius=25.0
    )
    field = field.to(field.device)

    print(f"Seeds: {field.seed_neurons}, Max: {field.max_neurons}")
    print(f"Device: {field.device}")

    optimizer = torch.optim.AdamW(field.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    history = {'acc': [], 'neurons': [], 'epochs': []}
    best_acc = 0

    for epoch in range(60):
        field.train()
        epoch_loss = 0
        n_batches = 0
        t0 = time.time()

        for data, target in train_loader:
            data = data.view(-1, 3072).to(field.device)
            target = target.to(field.device)

            optimizer.zero_grad()
            out = field(data, n_iterations=4)
            loss = F.cross_entropy(out, target) + field.geometric_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Growth operations
        n_splits = 0
        n_inject = 0
        if (epoch + 1) % 2 == 0:
            n_splits = field.split_neurons(threshold=0.6)
        if (epoch + 1) % 3 == 0 and field.n_alive < 300:
            n_inject = field.inject_neurons(n_inject=3)

        # Test
        field.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 3072).to(field.device)
                target = target.to(field.device)
                pred = field(data, n_iterations=4).argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        acc = 100 * correct / total
        best_acc = max(best_acc, acc)

        history['acc'].append(acc)
        history['neurons'].append(field.n_alive)
        history['epochs'].append(epoch + 1)

        print(f"Epoch {epoch+1:2d} [{time.time()-t0:.1f}s] "
              f"Loss: {epoch_loss/n_batches:.3f} "
              f"Acc: {acc:.1f}% (best: {best_acc:.1f}%) "
              f"Neurons: {field.n_alive} (+{n_splits}s +{n_inject}i)")

    print(f"\nFinal: {field.n_alive} neurons, {best_acc:.1f}% accuracy")

    # Quick plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['epochs'], history['acc'], 'g-', lw=2)
    ax1.axhline(best_acc, color='g', ls='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'CIFAR-10: {best_acc:.1f}%')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['epochs'], history['neurons'], 'r-', lw=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Neurons')
    ax2.set_title(f'Growth: {field.seed_neurons} → {field.n_alive}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cifar_v2_results.png', dpi=150)
    print("Saved cifar_v2_results.png")

    return field, history, best_acc


if __name__ == "__main__":
    field, history, best_acc = train_cifar_v2()
