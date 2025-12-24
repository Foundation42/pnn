"""
Continuous Neural Field

NOT a layered network. A CONTINUOUS VOLUME of neurons that self-organize!

Start with dense soup → Watch it crystallize through:
- Gradient descent (learning)
- Merge (combine similar nearby neurons)
- Prune (remove weak neurons)
- Split (divide overloaded hubs)

Uses spatial acceleration structures (octree) for O(n log n) computation!

This is Neural Architecture Search through PHYSICS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class FieldStats:
    """Statistics about the neural field state."""
    n_neurons: int
    n_connections: int
    volume_utilization: float
    mean_connectivity: float
    hierarchy_depth: float
    n_clusters: int


class Octree:
    """
    Spatial acceleration structure for O(log n) neighbor queries.

    Divides 3D space into nested octants.
    """

    def __init__(self, positions: torch.Tensor, indices: torch.Tensor,
                 bounds: Tuple[torch.Tensor, torch.Tensor],
                 max_depth: int = 8, min_neurons: int = 8):
        """
        Build octree from neuron positions.

        Args:
            positions: (N, 3) tensor of neuron positions
            indices: (N,) tensor of original neuron indices
            bounds: (min_corner, max_corner) of this node
            max_depth: Maximum tree depth
            min_neurons: Minimum neurons to split further
        """
        self.bounds_min, self.bounds_max = bounds
        self.center = (self.bounds_min + self.bounds_max) / 2
        self.is_leaf = True
        self.children = [None] * 8
        self.neuron_indices = indices
        self.positions = positions

        # Split if we have enough neurons and depth remaining
        if len(indices) > min_neurons and max_depth > 0:
            self._split(positions, indices, max_depth - 1, min_neurons)

    def _split(self, positions: torch.Tensor, indices: torch.Tensor,
               max_depth: int, min_neurons: int):
        """Split this node into 8 children."""
        self.is_leaf = False

        # Determine which octant each neuron belongs to
        octants = ((positions >= self.center).int() *
                   torch.tensor([[4, 2, 1]], device=positions.device)).sum(dim=1)

        for i in range(8):
            mask = octants == i
            if mask.sum() > 0:
                child_positions = positions[mask]
                child_indices = indices[mask]

                # Compute child bounds
                child_min = self.bounds_min.clone()
                child_max = self.bounds_max.clone()

                if i & 4:  # x >= center
                    child_min[0] = self.center[0]
                else:
                    child_max[0] = self.center[0]

                if i & 2:  # y >= center
                    child_min[1] = self.center[1]
                else:
                    child_max[1] = self.center[1]

                if i & 1:  # z >= center
                    child_min[2] = self.center[2]
                else:
                    child_max[2] = self.center[2]

                self.children[i] = Octree(
                    child_positions, child_indices,
                    (child_min, child_max),
                    max_depth, min_neurons
                )

        # Clear leaf data
        self.neuron_indices = None
        self.positions = None

    def query_radius(self, center: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Find all neuron indices within radius of center.
        O(log n) average case!
        """
        # Early exit if this node doesn't intersect the query sphere
        if not self._intersects_sphere(center, radius):
            return torch.tensor([], dtype=torch.long, device=center.device)

        if self.is_leaf:
            # Check neurons in this leaf
            if self.positions is None or len(self.positions) == 0:
                return torch.tensor([], dtype=torch.long, device=center.device)

            distances = torch.norm(self.positions - center, dim=1)
            mask = distances <= radius
            return self.neuron_indices[mask]

        # Recursively query children
        results = []
        for child in self.children:
            if child is not None:
                results.append(child.query_radius(center, radius))

        if results:
            return torch.cat(results)
        return torch.tensor([], dtype=torch.long, device=center.device)

    def _intersects_sphere(self, center: torch.Tensor, radius: float) -> bool:
        """Check if this node's bounding box intersects a sphere."""
        # Find closest point in box to sphere center
        closest = torch.clamp(center, self.bounds_min, self.bounds_max)
        distance = torch.norm(closest - center)
        return distance <= radius


class ContinuousNeuralField(nn.Module):
    """
    A continuous volume of neurons that self-organize through training.

    This is NOT a layered network!
    Neurons exist at continuous 3D positions and connect to nearby neighbors.
    The architecture EMERGES from gradient descent + spatial pressure.
    """

    def __init__(self,
                 volume_size: Tuple[float, float, float] = (100, 100, 100),
                 initial_neurons: int = 1000,
                 input_size: int = 784,
                 output_size: int = 10,
                 feature_dim: int = 32,
                 base_connection_radius: float = 15.0,
                 device: str = None):
        """
        Args:
            volume_size: Size of the 3D volume (x, y, z)
            initial_neurons: Number of neurons to start with
            input_size: Dimension of input signal
            output_size: Dimension of output signal
            feature_dim: Internal feature dimension of each neuron
            base_connection_radius: Base radius for local connections
        """
        super().__init__()

        self.volume_size = torch.tensor(volume_size, dtype=torch.float32)
        self.initial_neurons = initial_neurons
        self.input_size = input_size
        self.output_size = output_size
        self.feature_dim = feature_dim
        self.base_connection_radius = base_connection_radius

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # === NEURON PARAMETERS ===

        # Positions in 3D volume (THE KEY - these MOVE during training!)
        self.positions = nn.Parameter(self._initialize_positions(initial_neurons))

        # Each neuron has input weights (from input signal) and feature weights
        self.input_weights = nn.Parameter(
            torch.randn(initial_neurons, input_size) * 0.01
        )

        # Feature transform (neuron's "type" or "function")
        self.features = nn.Parameter(
            torch.randn(initial_neurons, feature_dim) * 0.1
        )

        # Output weights (to produce final output)
        self.output_weights = nn.Parameter(
            torch.randn(initial_neurons, output_size) * 0.01
        )

        # Connection radius per neuron (can vary - learned!)
        self.connection_radii = nn.Parameter(
            torch.ones(initial_neurons) * base_connection_radius
        )

        # Activation threshold per neuron
        self.thresholds = nn.Parameter(torch.zeros(initial_neurons))

        # Importance/alive mask (for soft pruning)
        self.register_buffer('alive_mask', torch.ones(initial_neurons, dtype=torch.bool))

        # Octree (rebuilt periodically)
        self.octree = None
        self._rebuild_octree()

        # Statistics tracking
        self.stats_history = []

    def _initialize_positions(self, n: int) -> torch.Tensor:
        """Initialize neuron positions throughout the volume."""
        positions = torch.rand(n, 3) * self.volume_size

        # Create slight bias toward edges for input/output specialization
        # 20% of neurons near input face (low x)
        n_input_region = n // 5
        positions[:n_input_region, 0] = torch.rand(n_input_region) * self.volume_size[0] * 0.2

        # 20% of neurons near output face (high x)
        n_output_region = n // 5
        positions[n_input_region:2*n_input_region, 0] = (
            self.volume_size[0] * 0.8 +
            torch.rand(n_output_region) * self.volume_size[0] * 0.2
        )

        return positions

    def _rebuild_octree(self):
        """Rebuild spatial acceleration structure."""
        with torch.no_grad():
            alive_indices = torch.where(self.alive_mask)[0]
            if len(alive_indices) == 0:
                self.octree = None
                return

            # Move to CPU for octree (spatial queries are CPU-bound anyway)
            alive_positions = self.positions[alive_indices].detach().cpu()
            alive_indices_cpu = alive_indices.cpu()

            bounds = (
                torch.zeros(3),
                self.volume_size.cpu()
            )

            self.octree = Octree(
                alive_positions, alive_indices_cpu,
                bounds, max_depth=8, min_neurons=8
            )

    def get_neighbors(self, neuron_idx: int) -> torch.Tensor:
        """Get indices of neurons within connection radius."""
        if self.octree is None:
            return torch.tensor([], dtype=torch.long, device=self.device)

        pos = self.positions[neuron_idx].detach().cpu()
        radius = self.connection_radii[neuron_idx].detach().cpu()

        neighbors = self.octree.query_radius(pos, radius.item())

        # Remove self
        neighbors = neighbors[neighbors != neuron_idx]

        return neighbors.to(self.device)

    def forward(self, x: torch.Tensor, n_iterations: int = 2) -> torch.Tensor:
        """
        Forward pass with simplified spatial computation.

        Uses a dense distance matrix for efficiency on GPU.
        For very large networks, switch to sparse octree-based computation.
        """
        batch_size = x.shape[0]
        device = x.device

        # Get alive neurons
        alive_idx = torch.where(self.alive_mask)[0]
        n_alive = len(alive_idx)

        if n_alive == 0:
            return torch.zeros(batch_size, self.output_size, device=device)

        # Get positions and features of alive neurons
        # CLAMP positions to stay within volume (prevents NaN)
        vol_size = self.volume_size.to(device)
        raw_pos = self.positions[alive_idx]
        # Use functional clamp without in-place operations
        pos_x = torch.clamp(raw_pos[:, 0], min=0.1, max=vol_size[0].item() - 0.1)
        pos_y = torch.clamp(raw_pos[:, 1], min=0.1, max=vol_size[1].item() - 0.1)
        pos_z = torch.clamp(raw_pos[:, 2], min=0.1, max=vol_size[2].item() - 0.1)
        pos = torch.stack([pos_x, pos_y, pos_z], dim=1)
        feat = self.features[alive_idx]
        radii = torch.clamp(self.connection_radii[alive_idx], min=1.0, max=50.0)

        # Compute pairwise distances (dense for now - O(n²) but GPU-friendly)
        dist_matrix = torch.cdist(pos, pos)  # (n_alive, n_alive)

        # Compute connection weights based on distance and features
        # Mask connections beyond radius
        max_radius = radii.max()
        connection_mask = dist_matrix < max_radius

        # Distance-based attenuation - clamp to prevent exp overflow
        attenuation = torch.exp(-torch.clamp(dist_matrix / (radii.unsqueeze(1) + 1e-6), max=20.0))
        attenuation = attenuation * connection_mask.float()

        # Feature similarity
        feat_norm = F.normalize(feat, dim=1, eps=1e-6)
        feat_sim = torch.matmul(feat_norm, feat_norm.T)  # (n_alive, n_alive)

        # Combined connection weights
        conn_weights = attenuation * (0.3 + 0.7 * torch.clamp(feat_sim, -1, 1))

        # Normalize with better epsilon
        row_sums = conn_weights.sum(dim=1, keepdim=True)
        conn_weights = conn_weights / (row_sums + 1e-6)

        # Initialize activations from input - use safer normalized x_coords
        x_coords = torch.clamp(pos[:, 0] / vol_size[0], min=0.0, max=1.0)

        # Input weighting: neurons near input (low x) receive more input
        # Use softer exponential to prevent numerical issues
        input_weight = torch.exp(-3 * x_coords)  # Softer decay
        input_weight = input_weight / (input_weight.sum() + 1e-6)

        # Project input to neurons
        activations = torch.matmul(x, self.input_weights[alive_idx].T)  # (batch, n_alive)
        activations = activations * input_weight

        # Message passing iterations with activation clamping
        for _ in range(n_iterations):
            # Aggregate from neighbors
            aggregated = torch.matmul(activations, conn_weights.T)  # (batch, n_alive)

            # Update with nonlinearity and clamp to prevent explosion
            activations = F.relu(activations + aggregated - self.thresholds[alive_idx])
            activations = torch.clamp(activations, max=100.0)  # Prevent explosion

        # Output: neurons near output (high x) contribute more
        output_weight = torch.exp(3 * (x_coords - 1))  # Softer curve
        output_weight = output_weight / (output_weight.sum() + 1e-6)
        output_weight = output_weight.unsqueeze(1)  # (n_alive, 1)

        # Weighted output projection
        weighted_output = self.output_weights[alive_idx] * output_weight
        output = torch.matmul(activations, weighted_output)

        return output

    def geometric_loss(self,
                       clustering_weight: float = 0.01,
                       spacing_weight: float = 0.1,
                       sparsity_weight: float = 0.001) -> torch.Tensor:
        """
        Loss terms that encourage self-organization.
        """
        losses = []

        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) < 2:
            return torch.tensor(0.0, device=self.device)

        pos = self.positions[alive_idx]
        features = self.features[alive_idx]

        # 1. Clustering pressure: similar features should be spatially close
        # Sample pairs for efficiency
        n_samples = min(500, len(alive_idx) * (len(alive_idx) - 1) // 2)
        i_idx = torch.randint(0, len(alive_idx), (n_samples,), device=self.device)
        j_idx = torch.randint(0, len(alive_idx), (n_samples,), device=self.device)

        # Avoid self-pairs
        valid = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]

        if len(i_idx) > 0:
            feature_similarity = F.cosine_similarity(features[i_idx], features[j_idx], dim=1)
            spatial_distance = torch.norm(pos[i_idx] - pos[j_idx], dim=1)

            # Similar features should be close, different features can be far
            clustering_loss = (feature_similarity * spatial_distance).mean()
            losses.append(clustering_weight * clustering_loss)

        # 2. Minimum spacing (prevent overlap)
        if len(i_idx) > 0:
            min_spacing = 2.0
            spacing_violations = F.relu(min_spacing - spatial_distance)
            spacing_loss = spacing_violations.mean()
            losses.append(spacing_weight * spacing_loss)

        # 3. Sparsity pressure (encourage efficient representation)
        feature_norms = features.norm(dim=1)
        sparsity_loss = -feature_norms.mean()  # Encourage some neurons to be weak
        losses.append(sparsity_weight * sparsity_loss)

        return sum(losses) if losses else torch.tensor(0.0, device=self.device)

    # === SELF-ORGANIZATION OPERATIONS ===

    def merge_nearby_neurons(self, distance_threshold: float = 2.0,
                              similarity_threshold: float = 0.9) -> int:
        """
        Merge neurons that are close AND similar.
        Returns number of merges performed.
        """
        n_merges = 0

        with torch.no_grad():
            alive_idx = torch.where(self.alive_mask)[0]

            for i in range(len(alive_idx)):
                idx_i = alive_idx[i]
                if not self.alive_mask[idx_i]:
                    continue

                pos_i = self.positions[idx_i].detach().cpu()
                feat_i = self.features[idx_i]

                # Find close neighbors
                neighbors = self.octree.query_radius(pos_i, distance_threshold)

                for idx_j in neighbors:
                    if idx_j <= idx_i or not self.alive_mask[idx_j]:
                        continue

                    # Check feature similarity
                    similarity = F.cosine_similarity(
                        feat_i.unsqueeze(0),
                        self.features[idx_j].unsqueeze(0)
                    ).item()

                    if similarity > similarity_threshold:
                        # MERGE: average into i, kill j
                        self.positions.data[idx_i] = (
                            self.positions[idx_i] + self.positions[idx_j]
                        ) / 2
                        self.features.data[idx_i] = (
                            self.features[idx_i] + self.features[idx_j]
                        ) / 2
                        self.input_weights.data[idx_i] = (
                            self.input_weights[idx_i] + self.input_weights[idx_j]
                        ) / 2
                        self.output_weights.data[idx_i] = (
                            self.output_weights[idx_i] + self.output_weights[idx_j]
                        ) / 2

                        self.alive_mask[idx_j] = False
                        n_merges += 1

            if n_merges > 0:
                self._rebuild_octree()

        return n_merges

    def prune_weak_neurons(self, importance_threshold: float = 0.1) -> int:
        """
        Remove neurons with low importance.
        Returns number of neurons pruned.
        """
        n_pruned = 0

        with torch.no_grad():
            alive_idx = torch.where(self.alive_mask)[0]

            # Compute importance: feature magnitude + output contribution
            for idx in alive_idx:
                feature_importance = self.features[idx].norm().item()
                output_importance = self.output_weights[idx].norm().item()
                input_importance = self.input_weights[idx].norm().item()

                # Connectivity importance
                neighbors = self.get_neighbors(idx.item())
                connectivity = len(neighbors) / self.initial_neurons

                total_importance = (
                    feature_importance +
                    output_importance +
                    input_importance +
                    connectivity * 10
                )

                if total_importance < importance_threshold:
                    self.alive_mask[idx] = False
                    n_pruned += 1

            if n_pruned > 0:
                self._rebuild_octree()

        return n_pruned

    def get_stats(self) -> FieldStats:
        """Get current field statistics."""
        with torch.no_grad():
            alive_idx = torch.where(self.alive_mask)[0]
            n_alive = len(alive_idx)

            if n_alive == 0:
                return FieldStats(0, 0, 0, 0, 0, 0)

            # Count connections
            n_connections = 0
            connectivities = []
            for idx in alive_idx[:min(100, n_alive)]:  # Sample for speed
                neighbors = self.get_neighbors(idx.item())
                n_connections += len(neighbors)
                connectivities.append(len(neighbors))

            # Volume utilization
            pos = self.positions[alive_idx]
            used_volume = (
                (pos[:, 0].max() - pos[:, 0].min()) *
                (pos[:, 1].max() - pos[:, 1].min()) *
                (pos[:, 2].max() - pos[:, 2].min())
            )
            total_volume = self.volume_size.prod()

            # Hierarchy depth (x-coordinate spread)
            x_spread = (pos[:, 0].max() - pos[:, 0].min()) / self.volume_size[0]

            # Estimate clusters using k-means
            try:
                from sklearn.cluster import KMeans
                if n_alive >= 10:
                    kmeans = KMeans(n_clusters=min(10, n_alive // 5), random_state=42, n_init=3)
                    kmeans.fit(pos.cpu().numpy())
                    n_clusters = len(set(kmeans.labels_))
                else:
                    n_clusters = 1
            except:
                n_clusters = 1

            return FieldStats(
                n_neurons=n_alive,
                n_connections=n_connections,
                volume_utilization=(used_volume / total_volume).item(),
                mean_connectivity=np.mean(connectivities) if connectivities else 0,
                hierarchy_depth=x_spread.item(),
                n_clusters=n_clusters
            )

    def reorganize(self,
                   merge_distance: float = 2.0,
                   merge_similarity: float = 0.9,
                   prune_threshold: float = 0.1) -> Dict[str, int]:
        """
        Perform all self-organization operations.
        """
        n_merged = self.merge_nearby_neurons(merge_distance, merge_similarity)
        n_pruned = self.prune_weak_neurons(prune_threshold)

        # Always rebuild octree after reorganization
        self._rebuild_octree()

        return {
            'merged': n_merged,
            'pruned': n_pruned,
            'alive': self.alive_mask.sum().item()
        }

    def summary(self) -> str:
        """Print summary of field state."""
        stats = self.get_stats()

        lines = [
            "=" * 60,
            "Continuous Neural Field",
            "=" * 60,
            f"Volume: {self.volume_size[0]:.0f} × {self.volume_size[1]:.0f} × {self.volume_size[2]:.0f}",
            f"Initial neurons: {self.initial_neurons}",
            f"Alive neurons: {stats.n_neurons}",
            f"Compression: {self.initial_neurons / max(1, stats.n_neurons):.1f}×",
            "",
            "Structure:",
            f"  Connections: ~{stats.n_connections}",
            f"  Mean connectivity: {stats.mean_connectivity:.1f}",
            f"  Volume utilization: {stats.volume_utilization:.1%}",
            f"  Hierarchy depth: {stats.hierarchy_depth:.2f}",
            f"  Clusters: {stats.n_clusters}",
            "=" * 60,
        ]
        return "\n".join(lines)


def train_continuous_field(field: ContinuousNeuralField,
                           train_loader,
                           test_loader,
                           epochs: int = 100,
                           reorganize_every: int = 5,
                           lr: float = 0.01,
                           geometric_weight: float = 0.01,
                           verbose: bool = True) -> Dict:
    """
    Train a continuous neural field with periodic reorganization.
    """
    device = field.device
    field = field.to(device)

    optimizer = torch.optim.AdamW(field.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'test_acc': [],
        'n_neurons': [],
        'stats': [],
        'positions': [],
        'epochs': [],
    }

    if verbose:
        print("=" * 60)
        print("  Training Continuous Neural Field")
        print("  Watch the neurons self-organize!")
        print("=" * 60)
        print(f"  Initial neurons: {field.alive_mask.sum().item()}")
        print(f"  Volume: {field.volume_size.tolist()}")
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
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(field.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Reorganize periodically
        reorg_stats = {'merged': 0, 'pruned': 0}
        if (epoch + 1) % reorganize_every == 0:
            reorg_stats = field.reorganize(
                merge_distance=3.0,
                merge_similarity=0.85,
                prune_threshold=0.05
            )

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
        history['stats'].append(stats)
        history['epochs'].append(epoch + 1)

        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                alive_idx = torch.where(field.alive_mask)[0]
                history['positions'].append(
                    field.positions[alive_idx].detach().cpu().numpy().copy()
                )

        elapsed = time.time() - start_time

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} [{elapsed:.1f}s]")
            print(f"  Loss: {epoch_loss/n_batches:.4f}, Test Acc: {test_acc:.1f}%")
            print(f"  Neurons: {stats.n_neurons} (merged: {reorg_stats['merged']}, pruned: {reorg_stats['pruned']})")
            print(f"  Structure: {stats.n_clusters} clusters, {stats.mean_connectivity:.1f} avg connectivity")
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

    print("Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    print("Creating Continuous Neural Field...")
    field = ContinuousNeuralField(
        volume_size=(100, 100, 100),
        initial_neurons=2000,
        input_size=784,
        output_size=10,
        feature_dim=32,
        base_connection_radius=15.0
    )

    print(f"Initial state:\n{field.summary()}")

    print("\nTraining with self-organization...")
    history = train_continuous_field(
        field,
        train_loader,
        test_loader,
        epochs=30,
        reorganize_every=3,
        lr=0.01,
        geometric_weight=0.005,
        verbose=True
    )

    print(f"\nFinal state:\n{field.summary()}")
    print(f"\nCompression: {field.initial_neurons}→{field.alive_mask.sum().item()} neurons")
    print(f"Final accuracy: {history['test_acc'][-1]:.1f}%")
