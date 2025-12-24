"""
Growing Neural Field - Neural Mitosis!

Start with SEEDS, grow through SPLITTING!

This inverts the paradigm:
- Merge approach: 500 → 393 (prune redundancy)
- Growth approach: 10 → ??? (grow complexity)

If the optimal structure is an ATTRACTOR, both should converge
to the same crystalline geometry!

This is how organisms develop:
- Embryo starts with few cells
- Cells divide under pressure
- Differentiation during division
- Final organism = optimal structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class GrowthStats:
    """Statistics about the growing field."""
    n_neurons: int
    n_splits: int
    total_load: float
    max_load: float
    volume_utilization: float
    n_clusters: int


class GrowingNeuralField(nn.Module):
    """
    A neural field that GROWS from seeds through mitosis!

    Neurons split when they become overloaded:
    - High activation magnitude (doing too much work)
    - Many strong connections (hub neurons)

    Children inherit parent's position (nearby) and weights (with noise).
    """

    def __init__(self,
                 volume_size: Tuple[float, float, float] = (100, 100, 100),
                 seed_neurons: int = 10,
                 max_neurons: int = 1000,
                 input_size: int = 784,
                 output_size: int = 10,
                 feature_dim: int = 32,
                 base_connection_radius: float = 20.0,
                 split_threshold: float = 0.8,
                 device: str = None):
        """
        Args:
            volume_size: Size of the 3D volume
            seed_neurons: Number of initial seed neurons
            max_neurons: Maximum neurons to grow to
            input_size: Input dimension
            output_size: Output dimension
            feature_dim: Feature dimension per neuron
            base_connection_radius: Radius for connections
            split_threshold: Load threshold to trigger split (0-1)
        """
        super().__init__()

        self.volume_size = torch.tensor(volume_size, dtype=torch.float32)
        self.seed_neurons = seed_neurons
        self.max_neurons = max_neurons
        self.input_size = input_size
        self.output_size = output_size
        self.feature_dim = feature_dim
        self.base_connection_radius = base_connection_radius
        self.split_threshold = split_threshold

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Pre-allocate for max_neurons (we'll use alive_mask to track active ones)
        # This avoids constant reallocation during growth

        # === NEURON PARAMETERS ===

        # Positions - seeds distributed strategically
        initial_positions = self._initialize_seed_positions(seed_neurons)
        self.positions = nn.Parameter(
            torch.zeros(max_neurons, 3)
        )
        self.positions.data[:seed_neurons] = initial_positions

        # Input weights
        self.input_weights = nn.Parameter(
            torch.zeros(max_neurons, input_size)
        )
        self.input_weights.data[:seed_neurons] = torch.randn(seed_neurons, input_size) * 0.1

        # Features (neuron type/function)
        self.features = nn.Parameter(
            torch.zeros(max_neurons, feature_dim)
        )
        self.features.data[:seed_neurons] = torch.randn(seed_neurons, feature_dim) * 0.1

        # Output weights
        self.output_weights = nn.Parameter(
            torch.zeros(max_neurons, output_size)
        )
        self.output_weights.data[:seed_neurons] = torch.randn(seed_neurons, output_size) * 0.1

        # Connection radii
        self.connection_radii = nn.Parameter(
            torch.ones(max_neurons) * base_connection_radius
        )

        # Activation thresholds
        self.thresholds = nn.Parameter(torch.zeros(max_neurons))

        # Alive mask - tracks which neurons exist
        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.alive_mask[:seed_neurons] = True

        # Load tracking (for split decisions)
        self.register_buffer('neuron_load', torch.zeros(max_neurons))
        self.register_buffer('activation_history', torch.zeros(max_neurons))

        # Growth statistics
        self.total_splits = 0
        self.growth_history = []

    def _initialize_seed_positions(self, n: int) -> torch.Tensor:
        """
        Initialize seed neurons strategically across the volume.

        Seeds should cover:
        - Input region (low x)
        - Output region (high x)
        - Processing region (middle)
        """
        positions = torch.zeros(n, 3)

        # Divide seeds into input/processing/output regions
        n_input = max(1, n // 4)
        n_output = max(1, n // 4)
        n_process = n - n_input - n_output

        # Input region seeds (x: 0-20)
        positions[:n_input, 0] = torch.rand(n_input) * 20
        positions[:n_input, 1] = torch.rand(n_input) * self.volume_size[1]
        positions[:n_input, 2] = torch.rand(n_input) * self.volume_size[2]

        # Output region seeds (x: 80-100)
        positions[n_input:n_input+n_output, 0] = 80 + torch.rand(n_output) * 20
        positions[n_input:n_input+n_output, 1] = torch.rand(n_output) * self.volume_size[1]
        positions[n_input:n_input+n_output, 2] = torch.rand(n_output) * self.volume_size[2]

        # Processing region seeds (x: 20-80)
        positions[n_input+n_output:, 0] = 20 + torch.rand(n_process) * 60
        positions[n_input+n_output:, 1] = torch.rand(n_process) * self.volume_size[1]
        positions[n_input+n_output:, 2] = torch.rand(n_process) * self.volume_size[2]

        return positions

    @property
    def n_alive(self) -> int:
        return self.alive_mask.sum().item()

    def forward(self, x: torch.Tensor, n_iterations: int = 2) -> torch.Tensor:
        """Forward pass through the growing field."""
        batch_size = x.shape[0]
        device = x.device

        alive_idx = torch.where(self.alive_mask)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            return torch.zeros(batch_size, self.output_size, device=device)

        # Get alive neuron parameters with clamping for stability
        vol_size = self.volume_size.to(device)
        raw_pos = self.positions[alive_idx]
        pos_x = torch.clamp(raw_pos[:, 0], min=0.1, max=vol_size[0].item() - 0.1)
        pos_y = torch.clamp(raw_pos[:, 1], min=0.1, max=vol_size[1].item() - 0.1)
        pos_z = torch.clamp(raw_pos[:, 2], min=0.1, max=vol_size[2].item() - 0.1)
        pos = torch.stack([pos_x, pos_y, pos_z], dim=1)

        feat = self.features[alive_idx]
        radii = torch.clamp(self.connection_radii[alive_idx], min=1.0, max=50.0)

        # Distance matrix
        dist_matrix = torch.cdist(pos, pos)

        # Connection weights
        max_radius = radii.max()
        connection_mask = dist_matrix < max_radius

        attenuation = torch.exp(-torch.clamp(dist_matrix / (radii.unsqueeze(1) + 1e-6), max=20.0))
        attenuation = attenuation * connection_mask.float()

        feat_norm = F.normalize(feat, dim=1, eps=1e-6)
        feat_sim = torch.matmul(feat_norm, feat_norm.T)

        conn_weights = attenuation * (0.3 + 0.7 * torch.clamp(feat_sim, -1, 1))
        row_sums = conn_weights.sum(dim=1, keepdim=True)
        conn_weights = conn_weights / (row_sums + 1e-6)

        # Input weighting based on position
        x_coords = torch.clamp(pos[:, 0] / vol_size[0], min=0.0, max=1.0)
        input_weight = torch.exp(-3 * x_coords)
        input_weight = input_weight / (input_weight.sum() + 1e-6)

        # Project input to neurons
        activations = torch.matmul(x, self.input_weights[alive_idx].T)
        activations = activations * input_weight

        # Message passing
        for _ in range(n_iterations):
            aggregated = torch.matmul(activations, conn_weights.T)
            activations = F.relu(activations + aggregated - self.thresholds[alive_idx])
            activations = torch.clamp(activations, max=100.0)

        # Track activation magnitude for load computation
        with torch.no_grad():
            activation_magnitude = activations.abs().mean(dim=0)
            self.activation_history[alive_idx] = (
                0.9 * self.activation_history[alive_idx] +
                0.1 * activation_magnitude
            )

        # Output weighting
        output_weight = torch.exp(3 * (x_coords - 1))
        output_weight = output_weight / (output_weight.sum() + 1e-6)
        output_weight = output_weight.unsqueeze(1)

        weighted_output = self.output_weights[alive_idx] * output_weight
        output = torch.matmul(activations, weighted_output)

        return output

    def compute_neuron_load(self) -> torch.Tensor:
        """
        Compute load for each alive neuron.

        Load = activation_magnitude * connection_count
        High load = neuron is doing too much work = should split
        """
        alive_idx = torch.where(self.alive_mask)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            return torch.zeros(0, device=self.device)

        with torch.no_grad():
            pos = self.positions[alive_idx]
            radii = self.connection_radii[alive_idx]

            # Count connections for each neuron
            dist_matrix = torch.cdist(pos, pos)
            connection_counts = (dist_matrix < radii.unsqueeze(1)).sum(dim=1).float()

            # Normalize connection counts
            max_connections = connection_counts.max() + 1e-6
            norm_connections = connection_counts / max_connections

            # Get activation history
            activations = self.activation_history[alive_idx]
            max_activation = activations.max() + 1e-6
            norm_activations = activations / max_activation

            # Load = activation * connections
            load = norm_activations * norm_connections

            # Also consider weight magnitude (high weights = important)
            weight_magnitude = self.input_weights[alive_idx].abs().mean(dim=1)
            weight_magnitude = weight_magnitude / (weight_magnitude.max() + 1e-6)

            load = load * (0.5 + 0.5 * weight_magnitude)

            # Store load
            self.neuron_load[alive_idx] = load

        return load

    def split_overloaded_neurons(self, force_split: bool = False) -> int:
        """
        Split neurons that are overloaded.

        MITOSIS: One neuron becomes two!
        - Children positioned near parent
        - Children inherit parent's weights with noise
        - Children specialize through continued training

        Returns number of splits performed.
        """
        n_splits = 0

        # Compute current load
        load = self.compute_neuron_load()

        alive_idx = torch.where(self.alive_mask)[0]

        # Find overloaded neurons
        overloaded_mask = load > self.split_threshold
        overloaded_idx = alive_idx[overloaded_mask]

        # If forcing split and none overloaded, split the highest-load neuron
        if force_split and len(overloaded_idx) == 0 and len(alive_idx) > 0:
            max_load_idx = load.argmax()
            overloaded_idx = alive_idx[max_load_idx:max_load_idx+1]

        with torch.no_grad():
            for parent_idx in overloaded_idx:
                parent_idx = parent_idx.item()

                # Check if we have room to grow
                if self.n_alive >= self.max_neurons:
                    break

                # Find empty slot for child
                empty_slots = torch.where(~self.alive_mask)[0]
                if len(empty_slots) == 0:
                    break

                child_idx = empty_slots[0].item()

                # === MITOSIS ===

                # Position: Child near parent with random offset
                offset_magnitude = 5.0  # 5mm offset
                offset = torch.randn(3) * offset_magnitude
                child_pos = self.positions[parent_idx] + offset.to(self.device)

                # Clamp to volume bounds
                child_pos = torch.clamp(
                    child_pos,
                    min=torch.tensor([0.1, 0.1, 0.1], device=self.device),
                    max=self.volume_size.to(self.device) - 0.1
                )

                # Weights: Inherit from parent with noise (differentiation!)
                noise_scale = 0.1

                self.positions.data[child_idx] = child_pos
                self.input_weights.data[child_idx] = (
                    self.input_weights.data[parent_idx] +
                    torch.randn_like(self.input_weights.data[parent_idx]) * noise_scale
                )
                self.features.data[child_idx] = (
                    self.features.data[parent_idx] +
                    torch.randn_like(self.features.data[parent_idx]) * noise_scale
                )
                self.output_weights.data[child_idx] = (
                    self.output_weights.data[parent_idx] +
                    torch.randn_like(self.output_weights.data[parent_idx]) * noise_scale
                )
                self.connection_radii.data[child_idx] = self.connection_radii.data[parent_idx]
                self.thresholds.data[child_idx] = self.thresholds.data[parent_idx]

                # Also slightly modify parent (both specialize!)
                self.input_weights.data[parent_idx] += (
                    torch.randn_like(self.input_weights.data[parent_idx]) * noise_scale * 0.5
                )
                self.features.data[parent_idx] += (
                    torch.randn_like(self.features.data[parent_idx]) * noise_scale * 0.5
                )

                # Activate child
                self.alive_mask[child_idx] = True

                # Reset load for both
                self.neuron_load[parent_idx] = 0
                self.neuron_load[child_idx] = 0
                self.activation_history[parent_idx] = 0
                self.activation_history[child_idx] = 0

                n_splits += 1

        self.total_splits += n_splits
        return n_splits

    def geometric_loss(self,
                       clustering_weight: float = 0.01,
                       spacing_weight: float = 0.1) -> torch.Tensor:
        """Geometric regularization for self-organization."""
        losses = []

        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) < 2:
            return torch.tensor(0.0, device=self.device)

        pos = self.positions[alive_idx]
        features = self.features[alive_idx]

        # Sample pairs for efficiency
        n_samples = min(500, len(alive_idx) * (len(alive_idx) - 1) // 2)
        i_idx = torch.randint(0, len(alive_idx), (n_samples,), device=self.device)
        j_idx = torch.randint(0, len(alive_idx), (n_samples,), device=self.device)

        valid = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]

        if len(i_idx) > 0:
            feature_similarity = F.cosine_similarity(features[i_idx], features[j_idx], dim=1)
            spatial_distance = torch.norm(pos[i_idx] - pos[j_idx], dim=1)

            # Similar features should be close
            clustering_loss = (feature_similarity * spatial_distance).mean()
            losses.append(clustering_weight * clustering_loss)

            # Minimum spacing
            min_spacing = 2.0
            spacing_violations = F.relu(min_spacing - spatial_distance)
            spacing_loss = spacing_violations.mean()
            losses.append(spacing_weight * spacing_loss)

        return sum(losses) if losses else torch.tensor(0.0, device=self.device)

    def get_stats(self) -> GrowthStats:
        """Get current growth statistics."""
        alive_idx = torch.where(self.alive_mask)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            return GrowthStats(0, self.total_splits, 0, 0, 0, 0)

        with torch.no_grad():
            load = self.compute_neuron_load()

            pos = self.positions[alive_idx]
            used_volume = (
                (pos[:, 0].max() - pos[:, 0].min()) *
                (pos[:, 1].max() - pos[:, 1].min()) *
                (pos[:, 2].max() - pos[:, 2].min())
            )
            total_volume = self.volume_size.prod()

            # Cluster estimate
            try:
                from sklearn.cluster import KMeans
                if n_alive >= 10:
                    kmeans = KMeans(n_clusters=min(10, n_alive // 3), random_state=42, n_init=3)
                    kmeans.fit(pos.cpu().numpy())
                    n_clusters = len(set(kmeans.labels_))
                else:
                    n_clusters = n_alive
            except:
                n_clusters = 1

        return GrowthStats(
            n_neurons=n_alive,
            n_splits=self.total_splits,
            total_load=load.sum().item() if len(load) > 0 else 0,
            max_load=load.max().item() if len(load) > 0 else 0,
            volume_utilization=(used_volume / total_volume).item() if total_volume > 0 else 0,
            n_clusters=n_clusters
        )

    def summary(self) -> str:
        """Print summary."""
        stats = self.get_stats()

        lines = [
            "=" * 60,
            "Growing Neural Field (Neural Mitosis)",
            "=" * 60,
            f"Volume: {self.volume_size[0]:.0f} x {self.volume_size[1]:.0f} x {self.volume_size[2]:.0f}",
            f"Seed neurons: {self.seed_neurons}",
            f"Current neurons: {stats.n_neurons}",
            f"Total splits: {stats.n_splits}",
            f"Growth factor: {stats.n_neurons / self.seed_neurons:.1f}x",
            "",
            "Structure:",
            f"  Max load: {stats.max_load:.3f}",
            f"  Volume utilization: {stats.volume_utilization:.1%}",
            f"  Clusters: {stats.n_clusters}",
            "=" * 60,
        ]
        return "\n".join(lines)


def train_growing_field(field: GrowingNeuralField,
                        train_loader,
                        test_loader,
                        epochs: int = 50,
                        split_every: int = 3,
                        lr: float = 0.01,
                        geometric_weight: float = 0.005,
                        verbose: bool = True) -> Dict:
    """
    Train a growing neural field with periodic splitting.
    """
    device = field.device
    field = field.to(device)

    # Use a learning rate that works with growing parameters
    optimizer = torch.optim.AdamW(field.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'test_acc': [],
        'n_neurons': [],
        'n_splits': [],
        'positions': [],
        'alive_masks': [],
        'volume_utilization': [],
        'epochs': [],
        'stats': [],
    }

    # Record initial state
    with torch.no_grad():
        history['positions'].append(field.positions.detach().cpu().numpy().copy())
        history['alive_masks'].append(field.alive_mask.detach().cpu().numpy().copy())
        history['n_neurons'].append(field.n_alive)
        history['n_splits'].append(0)
        history['epochs'].append(0)
        stats = field.get_stats()
        history['volume_utilization'].append(stats.volume_utilization)
        history['stats'].append(stats)

    if verbose:
        print("=" * 60)
        print("  Training Growing Neural Field")
        print("  Watch neurons multiply through MITOSIS!")
        print("=" * 60)
        print(f"  Seed neurons: {field.seed_neurons}")
        print(f"  Max neurons: {field.max_neurons}")
        print(f"  Split threshold: {field.split_threshold}")
        print("=" * 60)
        print()

    for epoch in range(epochs):
        field.train()
        epoch_loss = 0.0
        n_batches = 0

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, field.input_size).to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = field(data, n_iterations=2)

            class_loss = F.cross_entropy(output, target)
            geo_loss = field.geometric_loss()
            loss = class_loss + geometric_weight * geo_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Split overloaded neurons periodically
        n_new_splits = 0
        if (epoch + 1) % split_every == 0:
            # Force at least one split in early epochs if under-capacity
            force = (epoch < epochs // 2 and field.n_alive < field.max_neurons // 2)
            n_new_splits = field.split_overloaded_neurons(force_split=force)

        # Test accuracy
        field.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, field.input_size).to(device)
                target = target.to(device)
                output = field(data, n_iterations=2)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        test_acc = 100.0 * correct / total

        # Record history
        stats = field.get_stats()
        history['train_loss'].append(epoch_loss / n_batches)
        history['test_acc'].append(test_acc)
        history['n_neurons'].append(stats.n_neurons)
        history['n_splits'].append(field.total_splits)
        history['volume_utilization'].append(stats.volume_utilization)
        history['stats'].append(stats)

        # Record positions periodically
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                history['positions'].append(field.positions.detach().cpu().numpy().copy())
                history['alive_masks'].append(field.alive_mask.detach().cpu().numpy().copy())
                history['epochs'].append(epoch + 1)

        elapsed = time.time() - start_time

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} [{elapsed:.1f}s]")
            print(f"  Loss: {epoch_loss/n_batches:.4f}, Test Acc: {test_acc:.1f}%")
            print(f"  Neurons: {stats.n_neurons} (splits this epoch: {n_new_splits}, total: {field.total_splits})")
            print(f"  Load: max={stats.max_load:.3f}, Volume: {stats.volume_utilization:.1%}")
            print()

    if verbose:
        print("=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print(field.summary())

    return history


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    print("=" * 70)
    print("  GROWING NEURAL FIELD - NEURAL MITOSIS!")
    print("  Watch seeds grow into a neural network!")
    print("=" * 70)
    print()

    # Load data
    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, num_workers=2)

    # Create field with few seeds
    print("\nCreating field with 10 seed neurons...")
    field = GrowingNeuralField(
        volume_size=(100, 100, 100),
        seed_neurons=10,
        max_neurons=600,
        input_size=784,
        output_size=10,
        feature_dim=32,
        base_connection_radius=20.0,
        split_threshold=0.6  # Lower threshold = more splitting
    )

    print(f"Initial state:\n{field.summary()}")

    # Train with growth
    print("\nTraining with neural mitosis...")
    history = train_growing_field(
        field,
        train_loader,
        test_loader,
        epochs=40,
        split_every=2,  # Check for splits every 2 epochs
        lr=0.01,
        geometric_weight=0.005,
        verbose=True
    )

    print(f"\nFinal state:\n{field.summary()}")
    print(f"\nGrowth: {field.seed_neurons} -> {field.n_alive} neurons ({field.n_alive/field.seed_neurons:.1f}x)")
    print(f"Final accuracy: {history['test_acc'][-1]:.1f}%")
