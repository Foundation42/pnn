"""
Growing Crystal on CIFAR-10: Can the crystal paradigm scale?

CIFAR-10 is much harder than MNIST:
- 32x32x3 = 3072 input dimensions (vs 784)
- Color images with complex textures
- 10 classes but much more visual variety

Let's see if growth + freezing still works!
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


class CIFARCrystalField(nn.Module):
    """
    Growing Crystal for CIFAR-10.

    Key differences from MNIST version:
    - Larger input dimension (3072 vs 784)
    - More neurons needed (start with 16, grow to 256)
    - Add a simple conv frontend to reduce dimensionality
    """

    def __init__(
        self,
        seed_neurons: int = 16,
        space_dim: int = 2,
        space_size: float = 10.0,
        max_neurons: int = 256,
        freeze_after_epochs: int = 3,
        min_cluster_size: int = 2,
    ):
        super().__init__()

        self.space_dim = space_dim
        self.space_size = space_size
        self.max_neurons = max_neurons
        self.freeze_after_epochs = freeze_after_epochs
        self.min_cluster_size = min_cluster_size

        # Simple conv frontend to reduce CIFAR dimensions
        # 32x32x3 -> 16x16x16 -> 8x8x32 -> 512
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # After conv: 32 * 8 * 8 = 2048 -> project to smaller dim
        self.feature_dim = 256
        self.feature_proj = nn.Linear(32 * 8 * 8, self.feature_dim)

        # Crystal neurons
        self.num_neurons = seed_neurons
        self.register_buffer(
            'positions',
            torch.rand(seed_neurons, space_dim) * space_size
        )

        # Neuron parameters
        self.input_weights = nn.Parameter(torch.randn(seed_neurons, self.feature_dim) * 0.1)
        self.biases = nn.Parameter(torch.zeros(seed_neurons))
        self.output_weights = nn.Parameter(torch.randn(10, seed_neurons) * 0.1)

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
            'train_acc': [],
            'test_acc': [],
        }

    def _build_bvh(self, indices: List[int], depth: int = 0) -> BVHNode:
        """Build BVH using SAH-like heuristic for compact clusters."""
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
            sorted_indices = sorted(indices, key=lambda i: self.positions[i, axis].item())
            n = len(sorted_indices)

            for split_pos in range(max(1, n//4), min(n, 3*n//4 + 1), max(1, n//8)):
                left_idx = sorted_indices[:split_pos]
                right_idx = sorted_indices[split_pos:]

                if len(left_idx) < self.min_cluster_size or len(right_idx) < self.min_cluster_size:
                    continue

                left_pos = positions[[indices.index(i) for i in left_idx]]
                right_pos = positions[[indices.index(i) for i in right_idx]]

                left_extent = left_pos.max(axis=0) - left_pos.min(axis=0) + 1e-6
                right_extent = right_pos.max(axis=0) - right_pos.min(axis=0) + 1e-6

                # Cost = perimeter * count (SAH for 2D)
                left_cost = 2 * (left_extent[0] + left_extent[1]) * len(left_idx)
                right_cost = 2 * (right_extent[0] + right_extent[1]) * len(right_idx)
                total_cost = left_cost + right_cost

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_split = (left_idx, right_idx)

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
        # Conv frontend
        x = self.pool(F.relu(self.conv1(x)))  # 32->16
        x = self.pool(F.relu(self.conv2(x)))  # 16->8
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.feature_proj(x))  # Project to feature_dim

        # Crystal neurons
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
        """Update temperatures and optionally grow/freeze."""
        leaves = self.get_leaf_nodes()
        if not leaves:
            return {'hot': 0, 'warm': 0, 'cooling': 0, 'frozen': 0, 'grown': 0}

        flows = [(node, node.gradient_flow) for node in leaves if node.gradient_flow > 0 and not node.frozen]
        if not flows:
            frozen_count = sum(n.size for n in leaves if n.frozen)
            return {'hot': 0, 'warm': 0, 'cooling': 0, 'frozen': frozen_count, 'grown': 0}

        flows_sorted = sorted([f for _, f in flows])
        n = len(flows_sorted)

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
                node.stable_epochs += 1
                if node.stable_epochs >= self.freeze_after_epochs:
                    node.state = ClusterState.COOLING
                    state_counts['cooling'] += node.size
                else:
                    node.state = ClusterState.WARM
                    state_counts['warm'] += node.size
            else:
                node.stable_epochs += 1
                if node.stable_epochs >= self.freeze_after_epochs:
                    node.state = ClusterState.FROZEN
                    node.frozen = True
                    for idx in node.neuron_indices:
                        if idx < len(self.frozen_mask):
                            self.frozen_mask[idx] = True
                    print(f"  ❄️  FROZEN {node.size} neurons (grad={grad:.4f})")
                    state_counts['frozen'] += node.size
                else:
                    node.state = ClusterState.COOLING
                    state_counts['cooling'] += node.size

        # Grow in hottest regions
        if nodes_to_grow and self.num_neurons < self.max_neurons:
            hottest = max(nodes_to_grow, key=lambda n: n.gradient_flow)
            neurons_to_add = min(8, self.max_neurons - self.num_neurons)  # Grow faster for CIFAR
            grown = self._grow_neurons(neurons_to_add, hottest)
            state_counts['grown'] = grown
            if grown > 0:
                print(f"  🌱 GREW {grown} neurons (total: {self.num_neurons})")

        return state_counts

    def _grow_neurons(self, n: int, near_cluster: BVHNode) -> int:
        if n <= 0:
            return 0

        old_num = self.num_neurons
        new_num = old_num + n

        center = torch.tensor(near_cluster.center, dtype=torch.float32, device=self.positions.device)
        spread = max(0.5, (near_cluster.bbox_max - near_cluster.bbox_min).mean() * 0.5)
        new_positions = center + torch.randn(n, self.space_dim, device=self.positions.device) * spread
        new_positions = torch.clamp(new_positions, 0, self.space_size)

        self.positions = torch.cat([self.positions, new_positions], dim=0)

        device = self.input_weights.device
        new_input = torch.randn(n, self.feature_dim, device=device) * 0.01
        new_bias = torch.zeros(n, device=device)
        new_output = torch.randn(10, n, device=device) * 0.01

        self.input_weights = nn.Parameter(torch.cat([self.input_weights.data, new_input], dim=0))
        self.biases = nn.Parameter(torch.cat([self.biases.data, new_bias], dim=0))
        self.output_weights = nn.Parameter(torch.cat([self.output_weights.data, new_output], dim=1))

        self.frozen_mask = torch.cat([
            self.frozen_mask,
            torch.zeros(n, dtype=torch.bool, device=self.frozen_mask.device)
        ])

        self.num_neurons = new_num
        self.bvh_root = self._build_bvh(list(range(self.num_neurons)))

        return n

    def apply_sparse_gradients(self):
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        state_colors = {
            ClusterState.HOT: '#ff4444',
            ClusterState.WARM: '#ffaa44',
            ClusterState.COOLING: '#ffcccc',
            ClusterState.FROZEN: '#4444ff',
        }

        ax1 = axes[0]
        ax1.set_xlim(-0.5, self.space_size + 0.5)
        ax1.set_ylim(-0.5, self.space_size + 0.5)
        ax1.set_aspect('equal')
        ax1.set_title(f'CIFAR Crystal (Epoch {self.epoch}, {self.num_neurons} neurons)')

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
                       c=colors, s=60, edgecolors='black', linewidths=0.5, zorder=5)

        legend_elements = [
            Patch(facecolor=state_colors[ClusterState.HOT], label='Hot'),
            Patch(facecolor=state_colors[ClusterState.WARM], label='Warm'),
            Patch(facecolor=state_colors[ClusterState.COOLING], label='Cooling'),
            Patch(facecolor=state_colors[ClusterState.FROZEN], label='Frozen'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right')

        # Middle: Growth + Freeze
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

        # Right: Accuracy + Speedup
        ax3 = axes[2]
        if len(self.stats['test_acc']) > 0:
            epochs = range(len(self.stats['test_acc']))
            ax3.plot(epochs, self.stats['test_acc'], 'g-', linewidth=2, label='Test Acc')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_title(f'Accuracy (Best: {max(self.stats["test_acc"]):.1f}%)')

            ax3_twin = ax3.twinx()
            ax3_twin.plot(epochs, self.stats['speedup'], 'b--', linewidth=2, label='Speedup')
            ax3_twin.set_ylabel('Speedup (x)', color='blue')

            ax3.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def train_cifar_crystal():
    """Train growing crystal on CIFAR-10."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/cifar_crystal_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GROWING CRYSTAL: CIFAR-10")
    print("=" * 60)
    print(f"Output: {output_dir}")

    # CIFAR transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CIFARCrystalField(
        seed_neurons=16,
        max_neurons=256,
        freeze_after_epochs=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🌱 Starting with {model.num_neurons} seed neurons")
    print(f"📈 Can grow up to {model.max_neurons} neurons")
    print(f"Device: {device}\n")

    num_epochs = 60
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # Recreate optimizer if neurons were added
        if epoch > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=scheduler.get_last_lr()[0])

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

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
        scheduler.step()

        # Compute gradients for growth/freeze
        sample_data, sample_target = next(iter(train_loader))
        sample_data, sample_target = sample_data.to(device), sample_target.to(device)
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
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)

        test_acc = 100 * test_correct / test_total
        best_acc = max(best_acc, test_acc)

        model.stats['train_acc'].append(train_acc)
        model.stats['test_acc'].append(test_acc)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"  Train: {train_acc:.1f}% | Test: {test_acc:.1f}% (Best: {best_acc:.1f}%)")

        state_counts = model.update_and_grow()
        stats = model.epoch_end()

        print(f"  🌡️  Hot: {state_counts['hot']} | Warm: {state_counts['warm']} | Cooling: {state_counts['cooling']} | Frozen: {state_counts['frozen']}")
        print(f"  📊 Total: {stats['total']} | Active: {stats['active']} | Frozen: {stats['frozen']} ({stats['frozen_pct']:.1f}%)")
        print(f"  ⚡ Speedup: {stats['speedup']:.2f}x")

        model.visualize(f'{output_dir}/epoch_{epoch:02d}.png', state_counts)

    print("\n" + "=" * 60)
    print("FINAL RESULTS - CIFAR-10 CRYSTAL")
    print("=" * 60)
    print(f"Started with: 16 neurons")
    print(f"Grew to: {model.num_neurons} neurons")
    print(f"Frozen: {model.frozen_mask.sum().item()} ({model.frozen_mask.sum().item()/model.num_neurons*100:.1f}%)")
    print(f"Active: {model.num_neurons - model.frozen_mask.sum().item()}")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Final speedup: {model.stats['speedup'][-1]:.2f}x")

    torch.save({
        'model_state_dict': model.state_dict(),
        'stats': model.stats,
        'num_neurons': model.num_neurons,
        'best_acc': best_acc,
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
            imageio.mimsave(f'{output_dir}/crystal_growth.gif', images, duration=300, loop=0)
            print(f"\n🎬 Animation: {output_dir}/crystal_growth.gif")
    except:
        pass

    return model, output_dir


if __name__ == "__main__":
    model, output_dir = train_cifar_crystal()
    print(f"\n🎄 CIFAR crystal grown! Check {output_dir}/")
