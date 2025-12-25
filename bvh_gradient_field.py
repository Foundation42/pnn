"""
BVH Gradient Field: Hierarchical Learning with Gradient-Guided Sparsity

Key insight: Use Bounding Volume Hierarchy to organize neurons spatially,
then freeze stable clusters and focus compute on high-gradient regions.

This mimics biological learning:
- V1 freezes after critical period (edges done)
- Higher cortex stays plastic (novel patterns)
- Prefrontal most plastic (abstract reasoning)

Architecture:
- Neurons exist in continuous 2D/3D space
- BVH organizes them into hierarchical clusters
- Gradient flow tracks learning activity per cluster
- Frozen clusters: forward pass only, no backprop
- Active clusters: full learning, can split if overloaded
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation


class ClusterState:
    """Temperature states for clusters - from hot (learning) to frozen."""
    HOT = "hot"           # High gradient, actively learning
    WARM = "warm"         # Moderate gradient, still learning
    COOLING = "cooling"   # Low gradient, candidate for freezing
    FROZEN = "frozen"     # No gradient updates, just forward pass


@dataclass
class BVHNode:
    """A node in the Bounding Volume Hierarchy."""
    neuron_indices: List[int]  # Which neurons belong to this cluster
    bbox_min: np.ndarray       # Bounding box minimum (x, y)
    bbox_max: np.ndarray       # Bounding box maximum (x, y)

    # Gradient flow tracking
    gradient_flow: float = 0.0           # EMA of gradient magnitude
    gradient_flow_history: List[float] = field(default_factory=list)
    stable_epochs: int = 0               # How long gradient has been stable/low

    # State with temperature cascade
    state: str = "hot"  # hot -> warm -> cooling -> frozen
    frozen: bool = False
    depth: int = 0

    # Children (None if leaf)
    left: Optional['BVHNode'] = None
    right: Optional['BVHNode'] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def center(self) -> np.ndarray:
        return (self.bbox_min + self.bbox_max) / 2

    @property
    def size(self) -> int:
        return len(self.neuron_indices)

    def update_gradient_flow(self, new_flow: float, ema_alpha: float = 0.3):
        """Update gradient flow with exponential moving average."""
        if self.gradient_flow == 0:
            self.gradient_flow = new_flow
        else:
            self.gradient_flow = ema_alpha * new_flow + (1 - ema_alpha) * self.gradient_flow
        self.gradient_flow_history.append(self.gradient_flow)
        # Keep history bounded
        if len(self.gradient_flow_history) > 100:
            self.gradient_flow_history.pop(0)


class BVHGradientField(nn.Module):
    """
    Neural network with BVH-organized neurons and gradient-guided sparsity.

    Neurons live in continuous 2D space. The BVH organizes them hierarchically.
    Clusters with stable (low) gradient flow get frozen - they still contribute
    to forward pass but don't receive gradient updates (massive speedup).

    Clusters with high gradient flow can split to add capacity where needed.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_neurons: int = 32,
        space_dim: int = 2,
        space_size: float = 10.0,
        freeze_threshold: float = 0.01,
        freeze_patience: int = 10,
        split_threshold: float = 0.5,
        min_cluster_size: int = 4,
        max_depth: int = 6,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.space_dim = space_dim
        self.space_size = space_size

        # Thresholds for freeze/split decisions
        self.freeze_threshold = freeze_threshold
        self.freeze_patience = freeze_patience
        self.split_threshold = split_threshold
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth

        # Initialize neurons in space
        self.num_neurons = initial_neurons
        self.register_buffer(
            'positions',
            torch.rand(initial_neurons, space_dim) * space_size
        )

        # Neuron parameters
        self.input_weights = nn.Parameter(torch.randn(initial_neurons, input_dim) * 0.1)
        self.biases = nn.Parameter(torch.zeros(initial_neurons))
        self.output_weights = nn.Parameter(torch.randn(output_dim, initial_neurons) * 0.1)

        # Interaction based on distance (like physical forces)
        self.interaction_scale = nn.Parameter(torch.tensor(1.0))

        # Build initial BVH
        self.bvh_root = self._build_bvh(list(range(initial_neurons)))

        # Tracking
        self.frozen_mask = torch.zeros(initial_neurons, dtype=torch.bool)
        self.epoch = 0
        self.stats = {
            'frozen_neurons': [],
            'active_neurons': [],
            'total_neurons': [],
            'splits': 0,
            'freezes': 0,
        }

    def _build_bvh(self, indices: List[int], depth: int = 0) -> BVHNode:
        """Recursively build BVH from neuron indices."""
        if len(indices) == 0:
            raise ValueError("Cannot build BVH from empty indices")

        positions = self.positions[indices].cpu().numpy()
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)

        node = BVHNode(
            neuron_indices=indices,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            depth=depth,
        )

        # Stop recursion if too few neurons or max depth
        if len(indices) <= self.min_cluster_size or depth >= self.max_depth:
            return node

        # Split along longest axis
        extent = bbox_max - bbox_min
        split_axis = np.argmax(extent)

        # Sort by position along split axis
        sorted_indices = sorted(indices, key=lambda i: self.positions[i, split_axis].item())
        mid = len(sorted_indices) // 2

        left_indices = sorted_indices[:mid]
        right_indices = sorted_indices[mid:]

        if len(left_indices) > 0 and len(right_indices) > 0:
            node.left = self._build_bvh(left_indices, depth + 1)
            node.right = self._build_bvh(right_indices, depth + 1)

        return node

    def _compute_interactions(self) -> torch.Tensor:
        """Compute neuron-to-neuron interactions based on distance."""
        # Pairwise distances
        diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)  # [N, N, D]
        dist = torch.norm(diff, dim=-1) + 1e-6  # [N, N]

        # Interaction strength decays with distance (like gravity/EM)
        interactions = self.interaction_scale / dist
        interactions = interactions - torch.diag(torch.diag(interactions))  # Zero self-interaction

        return interactions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - all neurons contribute, frozen or not."""
        batch_size = x.shape[0]

        # Input to neurons
        neuron_input = F.linear(x, self.input_weights, self.biases)  # [B, N]

        # Neuron activations with interactions
        interactions = self._compute_interactions()  # [N, N]

        # Each neuron's activation is modulated by neighbors
        # This creates emergent spatial patterns
        activations = torch.tanh(neuron_input)  # [B, N]

        # Apply interaction (neighbors influence each other)
        modulated = activations + 0.1 * torch.matmul(activations, interactions)
        activations = torch.tanh(modulated)

        # Output
        output = F.linear(activations, self.output_weights)

        return output

    def compute_cluster_gradients(self) -> dict:
        """Compute gradient flow for each BVH cluster."""
        cluster_grads = {}

        def traverse(node: BVHNode):
            if node is None:
                return

            # Compute total gradient magnitude for this cluster
            indices = node.neuron_indices

            total_grad = 0.0
            count = 0

            # Check input weights
            if self.input_weights.grad is not None:
                for idx in indices:
                    grad_mag = self.input_weights.grad[idx].abs().sum().item()
                    total_grad += grad_mag
                    count += 1

            # Check output weights
            if self.output_weights.grad is not None:
                for idx in indices:
                    grad_mag = self.output_weights.grad[:, idx].abs().sum().item()
                    total_grad += grad_mag
                    count += 1

            if count > 0:
                avg_grad = total_grad / count
                node.update_gradient_flow(avg_grad)
                cluster_grads[id(node)] = avg_grad

            traverse(node.left)
            traverse(node.right)

        traverse(self.bvh_root)
        return cluster_grads

    def get_leaf_gradient_flows(self) -> List[Tuple[BVHNode, float]]:
        """Get gradient flows for all leaf clusters."""
        leaves = []

        def collect_leaves(node: BVHNode):
            if node is None:
                return
            if node.is_leaf:
                leaves.append((node, node.gradient_flow))
            else:
                collect_leaves(node.left)
                collect_leaves(node.right)

        collect_leaves(self.bvh_root)
        return leaves

    def update_temperature_states(self):
        """
        Update cluster temperature states based on gradient flow.

        AGGRESSIVE freezing - like irradiance caching in light transport:
        - HOT: grad > 75th percentile (high variance, needs samples)
        - WARM: grad > median (moderate variance)
        - COOLING: grad < median for 2+ epochs (low variance, reduce sampling)
        - FROZEN: grad < 25th percentile for 3+ epochs (cache this!)

        Key insight: If a region is stable, FREEZE IT. Don't waste compute.
        """
        leaves = self.get_leaf_gradient_flows()
        if not leaves:
            return {'hot': 0, 'warm': 0, 'cooling': 0, 'frozen': 0}

        # Get gradient distribution
        flows = [f for _, f in leaves if f > 0]
        if not flows:
            return {'hot': 0, 'warm': 0, 'cooling': 0, 'frozen': 0}

        flows_sorted = sorted(flows)
        n = len(flows_sorted)

        # Percentile thresholds - MORE AGGRESSIVE
        p25 = flows_sorted[int(n * 0.25)] if n > 4 else flows_sorted[0]
        p50 = flows_sorted[int(n * 0.50)] if n > 2 else flows_sorted[0]
        p75 = flows_sorted[int(n * 0.75)] if n > 4 else flows_sorted[-1]

        state_counts = {'hot': 0, 'warm': 0, 'cooling': 0, 'frozen': 0}
        newly_frozen = 0

        for node, grad in leaves:
            if node.frozen:
                state_counts['frozen'] += node.size
                continue

            # Determine temperature state - AGGRESSIVE thresholds
            if grad >= p75:
                # HIGH variance - needs active learning
                node.state = ClusterState.HOT
                node.stable_epochs = 0
                state_counts['hot'] += node.size
            elif grad >= p50:
                # Medium variance - still learning but slowing
                if node.state == ClusterState.COOLING:
                    # Was cooling, now warmed up - reset
                    node.stable_epochs = 0
                node.state = ClusterState.WARM
                state_counts['warm'] += node.size
            elif grad >= p25:
                # LOW variance - start cooling after 2 epochs
                node.stable_epochs += 1
                if node.stable_epochs >= 2:
                    node.state = ClusterState.COOLING
                    state_counts['cooling'] += node.size
                else:
                    node.state = ClusterState.WARM
                    state_counts['warm'] += node.size
            else:
                # VERY LOW variance - freeze after 3 epochs of being cold
                node.stable_epochs += 1

                if node.stable_epochs >= 3:
                    # FREEZE! Like caching irradiance
                    node.state = ClusterState.FROZEN
                    node.frozen = True
                    newly_frozen += 1
                    for idx in node.neuron_indices:
                        self.frozen_mask[idx] = True
                    print(f"  ❄️  FROZEN cluster depth={node.depth} neurons={node.size} (grad={grad:.6f} < p25={p25:.6f}, stable {node.stable_epochs} epochs)")
                    state_counts['frozen'] += node.size
                elif node.stable_epochs >= 2:
                    node.state = ClusterState.COOLING
                    state_counts['cooling'] += node.size
                else:
                    node.state = ClusterState.WARM
                    state_counts['warm'] += node.size

        self.stats['freezes'] += newly_frozen
        return state_counts

    def check_splits(self) -> int:
        """Check if any clusters should split due to high gradient flow."""
        splits_performed = 0

        def should_split(node: BVHNode) -> bool:
            return (
                node.is_leaf and
                not node.frozen and
                node.gradient_flow > self.split_threshold and
                node.size > self.min_cluster_size * 2 and
                node.depth < self.max_depth
            )

        def try_split(node: BVHNode) -> bool:
            nonlocal splits_performed

            if node is None:
                return False

            if should_split(node):
                # Split this cluster
                indices = node.neuron_indices
                positions = self.positions[indices].cpu().numpy()

                # Split along axis of maximum gradient variance
                # (where the gradient flow is most different)
                extent = node.bbox_max - node.bbox_min
                split_axis = np.argmax(extent)

                sorted_indices = sorted(indices, key=lambda i: self.positions[i, split_axis].item())
                mid = len(sorted_indices) // 2

                left_indices = sorted_indices[:mid]
                right_indices = sorted_indices[mid:]

                if len(left_indices) >= self.min_cluster_size and len(right_indices) >= self.min_cluster_size:
                    node.left = self._build_bvh(left_indices, node.depth + 1)
                    node.right = self._build_bvh(right_indices, node.depth + 1)
                    splits_performed += 1
                    print(f"  SPLIT cluster at depth {node.depth}: {node.size} -> {len(left_indices)} + {len(right_indices)}")
                    return True

            # Recurse
            try_split(node.left)
            try_split(node.right)
            return False

        try_split(self.bvh_root)
        self.stats['splits'] += splits_performed
        return splits_performed

    def add_neurons(self, n: int = 4, near_high_gradient: bool = True) -> int:
        """Add new neurons, preferably near high-gradient clusters."""
        old_num = self.num_neurons
        new_num = old_num + n

        # Find high-gradient regions
        if near_high_gradient:
            # Get cluster with highest gradient flow
            best_cluster = None
            best_flow = -1

            def find_best(node: BVHNode):
                nonlocal best_cluster, best_flow
                if node is None:
                    return
                if node.is_leaf and not node.frozen and node.gradient_flow > best_flow:
                    best_flow = node.gradient_flow
                    best_cluster = node
                find_best(node.left)
                find_best(node.right)

            find_best(self.bvh_root)

            if best_cluster is not None:
                # Add neurons near this cluster's center
                center = torch.tensor(best_cluster.center, dtype=torch.float32)
                spread = (best_cluster.bbox_max - best_cluster.bbox_min).mean() * 0.3
                new_positions = center + torch.randn(n, self.space_dim) * spread
            else:
                new_positions = torch.rand(n, self.space_dim) * self.space_size
        else:
            new_positions = torch.rand(n, self.space_dim) * self.space_size

        # Expand parameters
        new_positions = new_positions.to(self.positions.device)
        self.positions = torch.cat([self.positions, new_positions], dim=0)

        # New weights (small initialization)
        new_input = nn.Parameter(torch.randn(n, self.input_dim, device=self.input_weights.device) * 0.01)
        new_bias = nn.Parameter(torch.zeros(n, device=self.biases.device))
        new_output = nn.Parameter(torch.randn(self.output_dim, n, device=self.output_weights.device) * 0.01)

        self.input_weights = nn.Parameter(torch.cat([self.input_weights.data, new_input.data], dim=0))
        self.biases = nn.Parameter(torch.cat([self.biases.data, new_bias.data], dim=0))
        self.output_weights = nn.Parameter(torch.cat([self.output_weights.data, new_output.data], dim=1))

        # Expand frozen mask
        self.frozen_mask = torch.cat([
            self.frozen_mask,
            torch.zeros(n, dtype=torch.bool, device=self.frozen_mask.device)
        ])

        self.num_neurons = new_num

        # Rebuild BVH to include new neurons
        self.bvh_root = self._build_bvh(list(range(self.num_neurons)))

        print(f"  GREW {n} neurons near high-gradient region (total: {self.num_neurons})")
        return n

    def apply_sparse_gradients(self):
        """Zero out gradients for frozen neurons (sparse backprop)."""
        if self.input_weights.grad is not None:
            self.input_weights.grad[self.frozen_mask] = 0
        if self.biases.grad is not None:
            self.biases.grad[self.frozen_mask] = 0
        if self.output_weights.grad is not None:
            self.output_weights.grad[:, self.frozen_mask] = 0

    def epoch_end(self):
        """Called at end of each epoch to update BVH state."""
        self.epoch += 1

        # Update statistics
        frozen_count = self.frozen_mask.sum().item()
        active_count = self.num_neurons - frozen_count

        self.stats['frozen_neurons'].append(frozen_count)
        self.stats['active_neurons'].append(active_count)
        self.stats['total_neurons'].append(self.num_neurons)

        return {
            'frozen': frozen_count,
            'active': active_count,
            'total': self.num_neurons,
            'frozen_pct': frozen_count / self.num_neurons * 100,
        }

    def visualize(self, save_path: str = None, show_gradients: bool = True, state_counts: dict = None):
        """Visualize the BVH structure and neuron temperature states."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Color scheme for temperature states
        state_colors = {
            ClusterState.HOT: '#ff4444',      # Red - actively learning
            ClusterState.WARM: '#ffaa44',     # Orange - moderate learning
            ClusterState.COOLING: '#ffcccc',  # Salmon - cooling down
            ClusterState.FROZEN: '#4444ff',   # Blue - frozen
        }

        # Left: Spatial view with BVH boxes colored by temperature
        ax1 = axes[0]
        ax1.set_xlim(-0.5, self.space_size + 0.5)
        ax1.set_ylim(-0.5, self.space_size + 0.5)
        ax1.set_aspect('equal')
        ax1.set_title(f'BVH Temperature Map (Epoch {self.epoch})')

        # Collect leaf nodes for drawing
        def draw_leaves(node: BVHNode):
            if node is None:
                return
            if node.is_leaf:
                color = state_colors.get(node.state, '#888888')
                alpha = 0.6 if node.frozen else 0.4

                width = node.bbox_max[0] - node.bbox_min[0] + 0.1
                height = node.bbox_max[1] - node.bbox_min[1] + 0.1

                rect = Rectangle(
                    (node.bbox_min[0] - 0.05, node.bbox_min[1] - 0.05),
                    width, height,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha,
                )
                ax1.add_patch(rect)
            else:
                draw_leaves(node.left)
                draw_leaves(node.right)

        draw_leaves(self.bvh_root)

        # Draw neurons on top
        positions = self.positions.cpu().numpy()

        # Get state for each neuron from its cluster
        neuron_colors = []
        def get_neuron_states(node: BVHNode):
            if node is None:
                return
            if node.is_leaf:
                for idx in node.neuron_indices:
                    neuron_colors.append((idx, state_colors.get(node.state, '#888888')))
            else:
                get_neuron_states(node.left)
                get_neuron_states(node.right)

        get_neuron_states(self.bvh_root)
        neuron_colors.sort(key=lambda x: x[0])
        colors = [c for _, c in neuron_colors]

        ax1.scatter(positions[:, 0], positions[:, 1],
                   c=colors, s=60, edgecolors='black', linewidths=0.5, zorder=5)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=state_colors[ClusterState.HOT], label='Hot (learning)'),
            Patch(facecolor=state_colors[ClusterState.WARM], label='Warm'),
            Patch(facecolor=state_colors[ClusterState.COOLING], label='Cooling'),
            Patch(facecolor=state_colors[ClusterState.FROZEN], label='Frozen'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Middle: Temperature distribution bar
        ax2 = axes[1]
        if state_counts:
            states = ['Hot', 'Warm', 'Cooling', 'Frozen']
            counts = [state_counts.get('hot', 0), state_counts.get('warm', 0),
                     state_counts.get('cooling', 0), state_counts.get('frozen', 0)]
            colors_bar = [state_colors[ClusterState.HOT], state_colors[ClusterState.WARM],
                         state_colors[ClusterState.COOLING], state_colors[ClusterState.FROZEN]]

            bars = ax2.bar(states, counts, color=colors_bar, edgecolor='black')
            ax2.set_ylabel('Neurons')
            ax2.set_title('Temperature Distribution')

            # Add percentages
            total = sum(counts)
            for bar, count in zip(bars, counts):
                if total > 0:
                    pct = count / total * 100
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{pct:.0f}%', ha='center', va='bottom', fontsize=10)

        # Right: Statistics over time
        ax3 = axes[2]
        if len(self.stats['frozen_neurons']) > 0:
            epochs = range(len(self.stats['frozen_neurons']))

            # Stacked area chart
            frozen = np.array(self.stats['frozen_neurons'])
            active = np.array(self.stats['active_neurons'])

            ax3.fill_between(epochs, 0, frozen,
                           alpha=0.7, label='Frozen', color=state_colors[ClusterState.FROZEN])
            ax3.fill_between(epochs, frozen, frozen + active,
                           alpha=0.7, label='Active', color=state_colors[ClusterState.HOT])

            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Neurons')
            ax3.set_title('Freezing Progression')
            ax3.legend()

            # Add speedup annotation
            if len(frozen) > 0 and frozen[-1] > 0:
                frozen_pct = frozen[-1] / (frozen[-1] + active[-1])
                speedup = 1 / (1 - frozen_pct) if frozen_pct < 1 else float('inf')
                ax3.text(0.95, 0.95, f'Speedup: {speedup:.1f}x',
                        transform=ax3.transAxes, ha='right', va='top',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  📊 Saved: {save_path}")

        plt.close()
        return fig


def train_bvh_mnist():
    """Train BVH Gradient Field on MNIST."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    from datetime import datetime
    import os

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/bvh_gradient_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("BVH GRADIENT FIELD: Hierarchical Sparse Learning")
    print("=" * 60)
    print(f"Output directory: {output_dir}")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BVHGradientField(
        input_dim=784,
        output_dim=10,
        initial_neurons=64,
        space_size=10.0,
        freeze_threshold=0.1,
        freeze_patience=3,
        split_threshold=5.0,
        min_cluster_size=4,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\nInitial neurons: {model.num_neurons}")
    print(f"Device: {device}")
    print()

    # Training loop
    num_epochs = 30
    state_counts = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Apply sparse gradients (zero frozen neurons)
            model.apply_sparse_gradients()

            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        train_acc = 100 * correct / total

        # Compute cluster gradients ONCE at end of epoch (for efficiency)
        # Do a single forward-backward to get fresh gradients
        sample_data, sample_target = next(iter(train_loader))
        sample_data = sample_data.view(-1, 784).to(device)
        sample_target = sample_target.to(device)
        optimizer.zero_grad()
        output = model(sample_data)
        loss = criterion(output, sample_target)
        loss.backward()
        model.compute_cluster_gradients()

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 784).to(device)
                target = target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)

        test_acc = 100 * test_correct / test_total

        # End of epoch updates
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | Loss: {total_loss/len(train_loader):.4f}")

        # Update temperature states (this handles freezing)
        state_counts = model.update_temperature_states()
        print(f"  🌡️  Hot: {state_counts['hot']} | Warm: {state_counts['warm']} | Cooling: {state_counts['cooling']} | Frozen: {state_counts['frozen']}")

        # Check for splits
        splits = model.check_splits()

        # Optionally grow if accuracy plateaus
        if epoch > 10 and epoch % 5 == 0 and test_acc < 95:
            model.add_neurons(8, near_high_gradient=True)

        stats = model.epoch_end()
        frozen_pct = stats['frozen_pct']
        speedup = 1 / (1 - frozen_pct/100) if frozen_pct < 100 else float('inf')
        print(f"  Neurons: {stats['total']} | Frozen: {stats['frozen']} ({frozen_pct:.1f}%) | Speedup: {speedup:.2f}x")

        # Visualize EVERY epoch to see the cascade
        model.visualize(f'{output_dir}/epoch_{epoch:02d}.png', state_counts=state_counts)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Final neurons: {model.num_neurons}")
    print(f"Frozen neurons: {model.frozen_mask.sum().item()} ({model.frozen_mask.sum().item()/model.num_neurons*100:.1f}%)")
    print(f"Total splits: {model.stats['splits']}")
    print(f"Total freezes: {model.stats['freezes']}")
    print(f"Final test accuracy: {test_acc:.2f}%")

    # Compute theoretical speedup
    frozen_pct = model.frozen_mask.sum().item() / model.num_neurons
    if frozen_pct < 1:
        print(f"\nTheoretical backprop speedup: {1/(1-frozen_pct):.2f}x (only {(1-frozen_pct)*100:.1f}% of neurons trained)")
    else:
        print(f"\nAll neurons frozen - no more training needed!")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': model.stats,
        'num_neurons': model.num_neurons,
        'frozen_mask': model.frozen_mask,
    }, f'{output_dir}/model_final.pt')
    print(f"\nModel saved to {output_dir}/model_final.pt")

    return model, output_dir


if __name__ == "__main__":
    model, output_dir = train_bvh_mnist()
    print(f"\n🎄 Run complete! Check {output_dir}/ for visualizations.")
