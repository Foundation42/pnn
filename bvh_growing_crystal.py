"""
BVH Growing Crystal: Neural networks that GROW and FREEZE simultaneously!

Start from a tiny seed (8 neurons), then:
- GROW neurons where gradient flow is high (need more capacity)
- FREEZE neurons where gradient flow is low (crystallized knowledge)

Like a crystal growing from solution:
- Hot regions = supersaturated, actively crystallizing
- Frozen regions = solid crystal, stable structure

Combined with light transport insight:
- High gradient = high variance = need more samples/neurons
- Low gradient = stable = cache it and move on
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from datetime import datetime
import os


class ClusterState:
    HOT = "hot"
    WARM = "warm"
    COOLING = "cooling"
    FROZEN = "frozen"


@dataclass
class BVHNode:
    neuron_indices: List[int]
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    gradient_flow: float = 0.0
    gradient_flow_history: List[float] = field(default_factory=list)
    stable_epochs: int = 0
    state: str = "hot"
    frozen: bool = False
    depth: int = 0
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
        if self.gradient_flow == 0:
            self.gradient_flow = new_flow
        else:
            self.gradient_flow = ema_alpha * new_flow + (1 - ema_alpha) * self.gradient_flow
        self.gradient_flow_history.append(self.gradient_flow)
        if len(self.gradient_flow_history) > 100:
            self.gradient_flow_history.pop(0)


class GrowingCrystalField(nn.Module):
    """
    Neural network that grows AND freezes simultaneously.

    Like crystal growth:
    - Start from seed (few neurons)
    - Grow in high-energy regions
    - Crystallize (freeze) in stable regions
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seed_neurons: int = 8,  # Start TINY
        space_dim: int = 2,
        space_size: float = 10.0,
        max_neurons: int = 128,
        grow_threshold_percentile: float = 0.9,  # Grow in top 10% gradient
        freeze_after_epochs: int = 2,  # AGGRESSIVE freezing
        min_cluster_size: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.space_dim = space_dim
        self.space_size = space_size
        self.max_neurons = max_neurons
        self.grow_threshold_percentile = grow_threshold_percentile
        self.freeze_after_epochs = freeze_after_epochs
        self.min_cluster_size = min_cluster_size

        # Start with seed neurons
        self.num_neurons = seed_neurons
        self.register_buffer(
            'positions',
            torch.rand(seed_neurons, space_dim) * space_size
        )

        # Neuron parameters
        self.input_weights = nn.Parameter(torch.randn(seed_neurons, input_dim) * 0.1)
        self.biases = nn.Parameter(torch.zeros(seed_neurons))
        self.output_weights = nn.Parameter(torch.randn(output_dim, seed_neurons) * 0.1)

        self.interaction_scale = nn.Parameter(torch.tensor(1.0))

        # Build initial BVH
        self.bvh_root = self._build_bvh(list(range(seed_neurons)))

        # Tracking
        self.frozen_mask = torch.zeros(seed_neurons, dtype=torch.bool)
        self.epoch = 0
        self.stats = {
            'frozen_neurons': [],
            'active_neurons': [],
            'total_neurons': [],
            'grown': [],
            'speedup': [],
        }

    def _build_bvh(self, indices: List[int], depth: int = 0) -> BVHNode:
        """Build BVH using SAH-like heuristic based on spatial compactness."""
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

        if len(indices) <= self.min_cluster_size or depth >= 6:
            return node

        # SAH-like split: try multiple candidates, pick best
        best_cost = float('inf')
        best_split = None

        for axis in range(self.space_dim):
            # Sort along this axis
            sorted_indices = sorted(indices, key=lambda i: self.positions[i, axis].item())

            # Try several split positions (not just median)
            n = len(sorted_indices)
            for split_pos in range(max(1, n//4), min(n, 3*n//4 + 1), max(1, n//8)):
                left_idx = sorted_indices[:split_pos]
                right_idx = sorted_indices[split_pos:]

                if len(left_idx) < self.min_cluster_size or len(right_idx) < self.min_cluster_size:
                    continue

                # Compute SAH-like cost: surface_area * count
                # For 2D: "surface area" = perimeter of bounding box
                left_pos = positions[[indices.index(i) for i in left_idx]]
                right_pos = positions[[indices.index(i) for i in right_idx]]

                left_extent = left_pos.max(axis=0) - left_pos.min(axis=0) + 1e-6
                right_extent = right_pos.max(axis=0) - right_pos.min(axis=0) + 1e-6

                # Cost = left_perimeter * left_count + right_perimeter * right_count
                left_cost = 2 * (left_extent[0] + left_extent[1]) * len(left_idx)
                right_cost = 2 * (right_extent[0] + right_extent[1]) * len(right_idx)
                total_cost = left_cost + right_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_split = (left_idx, right_idx)

        # Fallback to median split if SAH didn't find good split
        if best_split is None:
            extent = bbox_max - bbox_min
            split_axis = np.argmax(extent)
            sorted_indices = sorted(indices, key=lambda i: self.positions[i, split_axis].item())
            mid = len(sorted_indices) // 2
            best_split = (sorted_indices[:mid], sorted_indices[mid:])

        left_indices, right_indices = best_split

        if len(left_indices) > 0 and len(right_indices) > 0:
            node.left = self._build_bvh(left_indices, depth + 1)
            node.right = self._build_bvh(right_indices, depth + 1)

        return node

    def _compute_interactions(self) -> torch.Tensor:
        diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1) + 1e-6
        interactions = self.interaction_scale / dist
        interactions = interactions - torch.diag(torch.diag(interactions))
        return interactions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        neuron_input = F.linear(x, self.input_weights, self.biases)
        interactions = self._compute_interactions()
        activations = torch.tanh(neuron_input)
        modulated = activations + 0.1 * torch.matmul(activations, interactions)
        activations = torch.tanh(modulated)
        output = F.linear(activations, self.output_weights)
        return output

    def compute_cluster_gradients(self) -> dict:
        cluster_grads = {}

        def traverse(node: BVHNode):
            if node is None:
                return
            if node.is_leaf:
                indices = node.neuron_indices
                total_grad = 0.0
                count = 0

                if self.input_weights.grad is not None:
                    for idx in indices:
                        if idx < self.input_weights.grad.shape[0]:
                            grad_mag = self.input_weights.grad[idx].abs().sum().item()
                            total_grad += grad_mag
                            count += 1

                if self.output_weights.grad is not None:
                    for idx in indices:
                        if idx < self.output_weights.grad.shape[1]:
                            grad_mag = self.output_weights.grad[:, idx].abs().sum().item()
                            total_grad += grad_mag
                            count += 1

                if count > 0:
                    avg_grad = total_grad / count
                    node.update_gradient_flow(avg_grad)
                    cluster_grads[id(node)] = (node, avg_grad)

            traverse(node.left)
            traverse(node.right)

        traverse(self.bvh_root)
        return cluster_grads

    def get_leaf_nodes(self) -> List[BVHNode]:
        leaves = []
        def collect(node):
            if node is None:
                return
            if node.is_leaf:
                leaves.append(node)
            else:
                collect(node.left)
                collect(node.right)
        collect(self.bvh_root)
        return leaves

    def update_and_grow(self) -> dict:
        """
        AGGRESSIVE update: freeze cold regions, grow in hot regions.
        """
        leaves = self.get_leaf_nodes()
        if not leaves:
            return {'hot': 0, 'warm': 0, 'cooling': 0, 'frozen': 0, 'grown': 0}

        # Get gradient distribution
        flows = [(node, node.gradient_flow) for node in leaves if node.gradient_flow > 0 and not node.frozen]
        if not flows:
            frozen_count = sum(n.size for n in leaves if n.frozen)
            return {'hot': 0, 'warm': 0, 'cooling': 0, 'frozen': frozen_count, 'grown': 0}

        flows_sorted = sorted([f for _, f in flows])
        n = len(flows_sorted)

        # Percentile thresholds
        p25 = flows_sorted[int(n * 0.25)] if n > 4 else flows_sorted[0]
        p50 = flows_sorted[int(n * 0.50)] if n > 2 else flows_sorted[0]
        p75 = flows_sorted[int(n * 0.75)] if n > 4 else flows_sorted[-1]
        p90 = flows_sorted[int(n * 0.90)] if n > 10 else flows_sorted[-1]

        state_counts = {'hot': 0, 'warm': 0, 'cooling': 0, 'frozen': 0, 'grown': 0}
        nodes_to_grow = []

        for node in leaves:
            if node.frozen:
                state_counts['frozen'] += node.size
                continue

            grad = node.gradient_flow

            if grad >= p90 and self.num_neurons < self.max_neurons:
                # TOP 10% - candidate for growth!
                node.state = ClusterState.HOT
                node.stable_epochs = 0
                state_counts['hot'] += node.size
                nodes_to_grow.append(node)
            elif grad >= p75:
                node.state = ClusterState.HOT
                node.stable_epochs = 0
                state_counts['hot'] += node.size
            elif grad >= p50:
                if node.state == ClusterState.COOLING:
                    node.stable_epochs = 0
                node.state = ClusterState.WARM
                state_counts['warm'] += node.size
            elif grad >= p25:
                # Below median - start cooling
                node.stable_epochs += 1
                if node.stable_epochs >= self.freeze_after_epochs:
                    node.state = ClusterState.COOLING
                    state_counts['cooling'] += node.size
                else:
                    node.state = ClusterState.WARM
                    state_counts['warm'] += node.size
            else:
                # Bottom 25% - FREEZE after 2 epochs
                node.stable_epochs += 1
                if node.stable_epochs >= self.freeze_after_epochs:
                    # FREEZE!
                    node.state = ClusterState.FROZEN
                    node.frozen = True
                    for idx in node.neuron_indices:
                        if idx < len(self.frozen_mask):
                            self.frozen_mask[idx] = True
                    print(f"  ❄️  FROZEN {node.size} neurons (grad={grad:.4f}, stable {node.stable_epochs} epochs)")
                    state_counts['frozen'] += node.size
                else:
                    node.state = ClusterState.COOLING
                    state_counts['cooling'] += node.size

        # GROW in hottest regions
        if nodes_to_grow and self.num_neurons < self.max_neurons:
            # Grow 2-4 neurons near the hottest cluster
            hottest = max(nodes_to_grow, key=lambda n: n.gradient_flow)
            neurons_to_add = min(4, self.max_neurons - self.num_neurons)
            grown = self._grow_neurons(neurons_to_add, hottest)
            state_counts['grown'] = grown
            if grown > 0:
                print(f"  🌱 GREW {grown} neurons near hot cluster (total: {self.num_neurons})")

        return state_counts

    def _grow_neurons(self, n: int, near_cluster: BVHNode) -> int:
        """Add neurons near a high-gradient cluster."""
        if n <= 0:
            return 0

        old_num = self.num_neurons
        new_num = old_num + n

        # Position near the hot cluster
        center = torch.tensor(near_cluster.center, dtype=torch.float32, device=self.positions.device)
        spread = max(0.5, (near_cluster.bbox_max - near_cluster.bbox_min).mean() * 0.5)
        new_positions = center + torch.randn(n, self.space_dim, device=self.positions.device) * spread

        # Clamp to space
        new_positions = torch.clamp(new_positions, 0, self.space_size)

        # Expand position buffer
        self.positions = torch.cat([self.positions, new_positions], dim=0)

        # New weights - small initialization for stable growth
        device = self.input_weights.device
        new_input = torch.randn(n, self.input_dim, device=device) * 0.01
        new_bias = torch.zeros(n, device=device)
        new_output = torch.randn(self.output_dim, n, device=device) * 0.01

        self.input_weights = nn.Parameter(torch.cat([self.input_weights.data, new_input], dim=0))
        self.biases = nn.Parameter(torch.cat([self.biases.data, new_bias], dim=0))
        self.output_weights = nn.Parameter(torch.cat([self.output_weights.data, new_output], dim=1))

        # Expand frozen mask
        self.frozen_mask = torch.cat([
            self.frozen_mask,
            torch.zeros(n, dtype=torch.bool, device=self.frozen_mask.device)
        ])

        self.num_neurons = new_num

        # Rebuild BVH with new neurons
        self.bvh_root = self._build_bvh(list(range(self.num_neurons)))

        return n

    def apply_sparse_gradients(self):
        """Zero gradients for frozen neurons."""
        if self.input_weights.grad is not None:
            mask = self.frozen_mask[:self.input_weights.grad.shape[0]]
            self.input_weights.grad[mask] = 0
        if self.biases.grad is not None:
            mask = self.frozen_mask[:self.biases.grad.shape[0]]
            self.biases.grad[mask] = 0
        if self.output_weights.grad is not None:
            mask = self.frozen_mask[:self.output_weights.grad.shape[1]]
            self.output_weights.grad[:, mask] = 0

    def epoch_end(self) -> dict:
        self.epoch += 1
        frozen_count = self.frozen_mask.sum().item()
        active_count = self.num_neurons - frozen_count

        self.stats['frozen_neurons'].append(frozen_count)
        self.stats['active_neurons'].append(active_count)
        self.stats['total_neurons'].append(self.num_neurons)

        speedup = self.num_neurons / max(active_count, 1)
        self.stats['speedup'].append(speedup)

        return {
            'frozen': frozen_count,
            'active': active_count,
            'total': self.num_neurons,
            'frozen_pct': frozen_count / self.num_neurons * 100 if self.num_neurons > 0 else 0,
            'speedup': speedup,
        }

    def visualize(self, save_path: str, state_counts: dict = None):
        """Visualize the growing crystal."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        state_colors = {
            ClusterState.HOT: '#ff4444',
            ClusterState.WARM: '#ffaa44',
            ClusterState.COOLING: '#ffcccc',
            ClusterState.FROZEN: '#4444ff',
        }

        # Left: Spatial view
        ax1 = axes[0]
        ax1.set_xlim(-0.5, self.space_size + 0.5)
        ax1.set_ylim(-0.5, self.space_size + 0.5)
        ax1.set_aspect('equal')
        ax1.set_title(f'Growing Crystal (Epoch {self.epoch}, {self.num_neurons} neurons)')

        # Draw BVH leaves
        def draw_leaves(node: BVHNode):
            if node is None:
                return
            if node.is_leaf:
                color = state_colors.get(node.state, '#888888')
                alpha = 0.6 if node.frozen else 0.3
                width = node.bbox_max[0] - node.bbox_min[0] + 0.1
                height = node.bbox_max[1] - node.bbox_min[1] + 0.1
                rect = Rectangle(
                    (node.bbox_min[0] - 0.05, node.bbox_min[1] - 0.05),
                    width, height,
                    linewidth=2, edgecolor=color, facecolor=color, alpha=alpha,
                )
                ax1.add_patch(rect)
            else:
                draw_leaves(node.left)
                draw_leaves(node.right)

        draw_leaves(self.bvh_root)

        # Draw neurons
        positions = self.positions.cpu().numpy()

        neuron_colors = []
        def get_colors(node: BVHNode):
            if node is None:
                return
            if node.is_leaf:
                for idx in node.neuron_indices:
                    if idx < len(positions):
                        neuron_colors.append((idx, state_colors.get(node.state, '#888888')))
            else:
                get_colors(node.left)
                get_colors(node.right)

        get_colors(self.bvh_root)
        neuron_colors.sort(key=lambda x: x[0])

        if neuron_colors:
            colors = [c for _, c in neuron_colors]
            pos_subset = positions[:len(colors)]
            ax1.scatter(pos_subset[:, 0], pos_subset[:, 1],
                       c=colors, s=80, edgecolors='black', linewidths=0.5, zorder=5)

        legend_elements = [
            Patch(facecolor=state_colors[ClusterState.HOT], label='Hot (growing)'),
            Patch(facecolor=state_colors[ClusterState.WARM], label='Warm'),
            Patch(facecolor=state_colors[ClusterState.COOLING], label='Cooling'),
            Patch(facecolor=state_colors[ClusterState.FROZEN], label='Frozen'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Middle: Growth + Freeze over time
        ax2 = axes[1]
        if len(self.stats['total_neurons']) > 0:
            epochs = range(len(self.stats['total_neurons']))
            ax2.fill_between(epochs, 0, self.stats['frozen_neurons'],
                           alpha=0.7, label='Frozen', color=state_colors[ClusterState.FROZEN])
            ax2.fill_between(epochs, self.stats['frozen_neurons'],
                           self.stats['total_neurons'],
                           alpha=0.7, label='Active', color=state_colors[ClusterState.HOT])
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Neurons')
            ax2.set_title('Crystal Growth & Freezing')
            ax2.legend()

        # Right: Speedup over time
        ax3 = axes[2]
        if len(self.stats['speedup']) > 0:
            epochs = range(len(self.stats['speedup']))
            ax3.plot(epochs, self.stats['speedup'], 'b-', linewidth=2, label='Speedup')
            ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Speedup (x)')
            ax3.set_title(f'Training Efficiency (Current: {self.stats["speedup"][-1]:.1f}x)')
            ax3.legend()

            # Annotate current speedup
            ax3.text(0.95, 0.95, f'{self.stats["speedup"][-1]:.1f}x',
                    transform=ax3.transAxes, ha='right', va='top',
                    fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def train_growing_crystal():
    """Train a growing crystal neural network."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/growing_crystal_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GROWING CRYSTAL: Neural Growth + Freezing")
    print("=" * 60)
    print(f"Output: {output_dir}")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create model - START TINY!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GrowingCrystalField(
        input_dim=784,
        output_dim=10,
        seed_neurons=8,        # Start with just 8 neurons!
        max_neurons=96,        # Can grow up to 96
        freeze_after_epochs=2, # Aggressive freezing
        min_cluster_size=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🌱 Starting with {model.num_neurons} seed neurons")
    print(f"📈 Can grow up to {model.max_neurons} neurons")
    print(f"❄️  Freezing after {model.freeze_after_epochs} stable epochs")
    print(f"Device: {device}\n")

    num_epochs = 40
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Recreate optimizer if neurons were added
        if epoch > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            model.apply_sparse_gradients()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        train_acc = 100 * correct / total

        # Compute gradients for growth/freeze decisions
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
        best_acc = max(best_acc, test_acc)

        # Update: grow and freeze
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | Loss: {total_loss/len(train_loader):.4f}")

        state_counts = model.update_and_grow()
        stats = model.epoch_end()

        print(f"  🌡️  Hot: {state_counts['hot']} | Warm: {state_counts['warm']} | Cooling: {state_counts['cooling']} | Frozen: {state_counts['frozen']}")
        print(f"  📊 Total: {stats['total']} | Active: {stats['active']} | Frozen: {stats['frozen']} ({stats['frozen_pct']:.1f}%)")
        print(f"  ⚡ Speedup: {stats['speedup']:.2f}x")

        # Visualize
        model.visualize(f'{output_dir}/epoch_{epoch:02d}.png', state_counts)

    print("\n" + "=" * 60)
    print("FINAL RESULTS - GROWING CRYSTAL")
    print("=" * 60)
    print(f"Started with: 8 neurons")
    print(f"Grew to: {model.num_neurons} neurons")
    print(f"Frozen: {model.frozen_mask.sum().item()} ({model.frozen_mask.sum().item()/model.num_neurons*100:.1f}%)")
    print(f"Active: {model.num_neurons - model.frozen_mask.sum().item()}")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Final speedup: {model.stats['speedup'][-1]:.2f}x")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': model.stats,
        'num_neurons': model.num_neurons,
    }, f'{output_dir}/model_final.pt')

    # Create GIF
    try:
        import imageio.v2 as imageio
        from PIL import Image
        images = []
        for i in range(num_epochs):
            img_path = f'{output_dir}/epoch_{i:02d}.png'
            if os.path.exists(img_path):
                img = Image.open(img_path)
                images.append(np.array(img))
        if images:
            imageio.mimsave(f'{output_dir}/crystal_growth.gif', images, duration=400, loop=0)
            print(f"\n🎬 Animation saved to {output_dir}/crystal_growth.gif")
    except:
        pass

    return model, output_dir


if __name__ == "__main__":
    model, output_dir = train_growing_crystal()
    print(f"\n🎄 Crystal grown! Check {output_dir}/")
