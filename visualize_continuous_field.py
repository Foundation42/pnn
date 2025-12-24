"""
Visualize Continuous Neural Field Evolution

Watch a dense neural soup CRYSTALLIZE into structure!
The ultimate "matter condensing out of a training loop" visualization.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib import cm
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from continuous_field import ContinuousNeuralField, FieldStats


def train_and_record(field: ContinuousNeuralField,
                     train_loader,
                     test_loader,
                     epochs: int = 30,
                     reorganize_every: int = 3,
                     lr: float = 0.01,
                     geometric_weight: float = 0.005,
                     record_every: int = 1,
                     verbose: bool = True):
    """
    Train and record detailed history for visualization.
    """
    device = field.device
    field = field.to(device)

    optimizer = torch.optim.AdamW(field.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'test_acc': [],
        'n_neurons': [],
        'positions': [],  # List of position arrays
        'alive_masks': [],  # Which neurons are alive
        'features': [],  # Neuron features for coloring
        'stats': [],
        'epochs': [],
        'merges': [],
        'prunes': [],
        'volume_utilization': [],
        'n_clusters': [],
    }

    # Record initial state
    with torch.no_grad():
        alive_idx = torch.where(field.alive_mask)[0]
        history['positions'].append(field.positions.detach().cpu().numpy().copy())
        history['alive_masks'].append(field.alive_mask.detach().cpu().numpy().copy())
        history['features'].append(field.features.detach().cpu().numpy().copy())
        history['epochs'].append(0)
        stats = field.get_stats()
        history['n_neurons'].append(stats.n_neurons)
        history['volume_utilization'].append(stats.volume_utilization)
        history['n_clusters'].append(stats.n_clusters)

    if verbose:
        print("=" * 60)
        print("  Training Continuous Neural Field for Visualization")
        print("  Watch the neural soup crystallize!")
        print("=" * 60)
        print(f"  Initial neurons: {field.alive_mask.sum().item()}")
        print(f"  Volume: {field.volume_size.tolist()}")
        print("=" * 60)
        print()

    total_merges = 0
    total_prunes = 0

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
            total_merges += reorg_stats['merged']
            total_prunes += reorg_stats['pruned']

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
        history['train_loss'].append(epoch_loss / n_batches)
        history['test_acc'].append(test_acc)
        history['merges'].append(total_merges)
        history['prunes'].append(total_prunes)

        # Record positions at regular intervals
        if (epoch + 1) % record_every == 0:
            with torch.no_grad():
                history['positions'].append(field.positions.detach().cpu().numpy().copy())
                history['alive_masks'].append(field.alive_mask.detach().cpu().numpy().copy())
                history['features'].append(field.features.detach().cpu().numpy().copy())
                history['epochs'].append(epoch + 1)

                stats = field.get_stats()
                history['n_neurons'].append(stats.n_neurons)
                history['volume_utilization'].append(stats.volume_utilization)
                history['n_clusters'].append(stats.n_clusters)
                history['stats'].append(stats)

        elapsed = time.time() - start_time

        if verbose:
            stats = field.get_stats()
            print(f"Epoch {epoch+1:3d}/{epochs} [{elapsed:.1f}s]")
            print(f"  Loss: {epoch_loss/n_batches:.4f}, Test Acc: {test_acc:.1f}%")
            print(f"  Neurons: {stats.n_neurons} (total merged: {total_merges}, pruned: {total_prunes})")
            print(f"  Structure: {stats.n_clusters} clusters, volume: {stats.volume_utilization:.1%}")
            print()

    return history


def create_3d_condensation_animation(history: dict,
                                      volume_size: tuple,
                                      output_path: str = 'neural_field_condensation.gif',
                                      fps: int = 3,
                                      dpi: int = 100):
    """
    Create a 3D animation of the neural field condensing.
    """
    positions_list = history['positions']
    alive_masks_list = history['alive_masks']
    features_list = history['features']
    epochs = history['epochs']

    n_frames = len(positions_list)

    fig = plt.figure(figsize=(14, 10))

    # Main 3D plot
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')

    # Top-down view (XY)
    ax_xy = fig.add_subplot(2, 2, 2)

    # Side view (XZ)
    ax_xz = fig.add_subplot(2, 2, 3)

    # Stats panel
    ax_stats = fig.add_subplot(2, 2, 4)

    def update(frame_idx):
        ax_3d.clear()
        ax_xy.clear()
        ax_xz.clear()
        ax_stats.clear()

        positions = positions_list[frame_idx]
        alive_mask = alive_masks_list[frame_idx]
        features = features_list[frame_idx]
        epoch = epochs[frame_idx]

        # Get alive positions
        alive_pos = positions[alive_mask]
        alive_feat = features[alive_mask]

        if len(alive_pos) == 0:
            return

        # Compute colors based on x-position (depth in network)
        x_normalized = alive_pos[:, 0] / volume_size[0]
        colors = plt.cm.viridis(x_normalized)

        # Size based on feature magnitude (importance)
        feat_magnitude = np.linalg.norm(alive_feat, axis=1)
        sizes = 20 + 30 * (feat_magnitude / (feat_magnitude.max() + 1e-8))

        # 3D scatter
        ax_3d.scatter(alive_pos[:, 0], alive_pos[:, 1], alive_pos[:, 2],
                      c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=0.3)

        ax_3d.set_xlim(0, volume_size[0])
        ax_3d.set_ylim(0, volume_size[1])
        ax_3d.set_zlim(0, volume_size[2])
        ax_3d.set_xlabel('X (Input → Output)')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f'Epoch {epoch}: {len(alive_pos)} Neurons', fontsize=12, fontweight='bold')

        # Add bounding box
        ax_3d.plot([0, volume_size[0]], [0, 0], [0, 0], 'k--', alpha=0.2)
        ax_3d.plot([0, 0], [0, volume_size[1]], [0, 0], 'k--', alpha=0.2)
        ax_3d.plot([0, 0], [0, 0], [0, volume_size[2]], 'k--', alpha=0.2)

        # Top-down view (XY projection)
        ax_xy.scatter(alive_pos[:, 0], alive_pos[:, 1], c=colors, s=sizes*0.5, alpha=0.6)
        ax_xy.set_xlim(0, volume_size[0])
        ax_xy.set_ylim(0, volume_size[1])
        ax_xy.set_xlabel('X (Depth)')
        ax_xy.set_ylabel('Y')
        ax_xy.set_title('Top-Down View (XY)', fontsize=10)
        ax_xy.set_aspect('equal')
        ax_xy.grid(True, alpha=0.3)

        # Side view (XZ projection)
        ax_xz.scatter(alive_pos[:, 0], alive_pos[:, 2], c=colors, s=sizes*0.5, alpha=0.6)
        ax_xz.set_xlim(0, volume_size[0])
        ax_xz.set_ylim(0, volume_size[2])
        ax_xz.set_xlabel('X (Depth)')
        ax_xz.set_ylabel('Z')
        ax_xz.set_title('Side View (XZ)', fontsize=10)
        ax_xz.set_aspect('equal')
        ax_xz.grid(True, alpha=0.3)

        # Stats panel
        ax_stats.axis('off')

        # Compute stats for this frame
        if frame_idx < len(history['n_neurons']):
            n_neurons = history['n_neurons'][frame_idx]
            vol_util = history['volume_utilization'][frame_idx]
            n_clusters = history['n_clusters'][frame_idx]

            # Get test accuracy if available
            if frame_idx > 0 and frame_idx <= len(history['test_acc']):
                test_acc = history['test_acc'][frame_idx - 1]
            else:
                test_acc = 0
        else:
            n_neurons = len(alive_pos)
            vol_util = 0
            n_clusters = 0
            test_acc = 0

        stats_text = f"""
CONTINUOUS NEURAL FIELD

Epoch: {epoch}

STRUCTURE
  Neurons: {n_neurons:,}
  Clusters: {n_clusters}
  Volume: {vol_util:.1%}

PERFORMANCE
  Test Acc: {test_acc:.1f}%

SELF-ORGANIZATION
  Total Merged: {history['merges'][max(0, frame_idx-1)] if frame_idx > 0 else 0}
  Total Pruned: {history['prunes'][max(0, frame_idx-1)] if frame_idx > 0 else 0}
"""

        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                      fontsize=11, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Create animation
    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    plt.tight_layout()

    # Save
    print(f"Saving to {output_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()

    print(f"Animation saved to {output_path}")


def create_evolution_dashboard(history: dict,
                                volume_size: tuple,
                                output_path: str = 'neural_field_evolution.png',
                                dpi: int = 150):
    """
    Create a dashboard showing the neural field evolution.
    """
    fig = plt.figure(figsize=(16, 12))

    # Title
    fig.suptitle('Continuous Neural Field Evolution\n"Matter Condensing from Training"',
                 fontsize=14, fontweight='bold', y=0.98)

    # 1. Initial vs Final 3D comparison
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')

    # Initial state
    pos_init = history['positions'][0]
    alive_init = history['alive_masks'][0]
    alive_pos_init = pos_init[alive_init]

    if len(alive_pos_init) > 0:
        colors_init = plt.cm.viridis(alive_pos_init[:, 0] / volume_size[0])
        ax1.scatter(alive_pos_init[:, 0], alive_pos_init[:, 1], alive_pos_init[:, 2],
                    c=colors_init, s=10, alpha=0.5)
    ax1.set_title(f'Initial: {len(alive_pos_init)} neurons', fontsize=10)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_xlim(0, volume_size[0]); ax1.set_ylim(0, volume_size[1]); ax1.set_zlim(0, volume_size[2])

    # Final state
    pos_final = history['positions'][-1]
    alive_final = history['alive_masks'][-1]
    alive_pos_final = pos_final[alive_final]

    if len(alive_pos_final) > 0:
        colors_final = plt.cm.viridis(alive_pos_final[:, 0] / volume_size[0])
        ax2.scatter(alive_pos_final[:, 0], alive_pos_final[:, 1], alive_pos_final[:, 2],
                    c=colors_final, s=20, alpha=0.7)
    ax2.set_title(f'Final: {len(alive_pos_final)} neurons', fontsize=10)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_xlim(0, volume_size[0]); ax2.set_ylim(0, volume_size[1]); ax2.set_zlim(0, volume_size[2])

    # 2. Neuron count over time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(history['epochs'][:len(history['n_neurons'])], history['n_neurons'], 'b-', linewidth=2)
    ax3.fill_between(history['epochs'][:len(history['n_neurons'])], history['n_neurons'], alpha=0.3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Alive Neurons')
    ax3.set_title('Neural Field Compression', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Add compression ratio
    if len(history['n_neurons']) > 0:
        compression = history['n_neurons'][0] / max(1, history['n_neurons'][-1])
        ax3.text(0.95, 0.95, f'{compression:.1f}x compression',
                 transform=ax3.transAxes, ha='right', va='top',
                 fontsize=10, fontweight='bold')

    # 3. Training progress
    ax4 = fig.add_subplot(2, 3, 4)
    ax4_twin = ax4.twinx()

    epochs_train = list(range(1, len(history['train_loss']) + 1))
    ax4.plot(epochs_train, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    ax4_twin.plot(epochs_train, history['test_acc'], 'g-', linewidth=2, label='Test Acc')

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='blue')
    ax4_twin.set_ylabel('Test Accuracy (%)', color='green')
    ax4.set_title('Training Progress', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 4. Volume utilization
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(history['epochs'][:len(history['volume_utilization'])],
             [v * 100 for v in history['volume_utilization']], 'm-', linewidth=2)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Volume Utilization (%)')
    ax5.set_title('Spatial Compaction', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 5. Cluster evolution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(history['epochs'][:len(history['n_clusters'])],
             history['n_clusters'], 'c-', linewidth=2, marker='o')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Number of Clusters')
    ax6.set_title('Structure Emergence', fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Dashboard saved to {output_path}")


def create_depth_histogram_animation(history: dict,
                                      volume_size: tuple,
                                      output_path: str = 'depth_evolution.gif',
                                      fps: int = 3,
                                      dpi: int = 100):
    """
    Animate how neurons distribute across depth (X axis) over time.
    Shows the emergence of layer-like structure.
    """
    positions_list = history['positions']
    alive_masks_list = history['alive_masks']
    epochs = history['epochs']

    n_frames = len(positions_list)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def update(frame_idx):
        axes[0].clear()
        axes[1].clear()

        positions = positions_list[frame_idx]
        alive_mask = alive_masks_list[frame_idx]
        epoch = epochs[frame_idx]

        alive_pos = positions[alive_mask]

        if len(alive_pos) == 0:
            return

        # Depth histogram (X axis)
        axes[0].hist(alive_pos[:, 0], bins=20, range=(0, volume_size[0]),
                     color='steelblue', edgecolor='white', alpha=0.7)
        axes[0].set_xlabel('X Position (Input → Output)')
        axes[0].set_ylabel('Neuron Count')
        axes[0].set_title(f'Epoch {epoch}: Depth Distribution', fontsize=12)
        axes[0].set_xlim(0, volume_size[0])

        # Mark input/output regions
        axes[0].axvline(x=volume_size[0] * 0.1, color='green', linestyle='--', label='Input region')
        axes[0].axvline(x=volume_size[0] * 0.9, color='red', linestyle='--', label='Output region')
        axes[0].legend(fontsize=8)

        # Radial distribution from center
        center = np.array(volume_size) / 2
        distances = np.linalg.norm(alive_pos - center, axis=1)
        max_dist = np.linalg.norm(center)

        axes[1].hist(distances, bins=20, range=(0, max_dist),
                     color='coral', edgecolor='white', alpha=0.7)
        axes[1].set_xlabel('Distance from Volume Center')
        axes[1].set_ylabel('Neuron Count')
        axes[1].set_title(f'Epoch {epoch}: Radial Distribution', fontsize=12)
        axes[1].set_xlim(0, max_dist)

        fig.suptitle(f'Neural Field Structure Evolution - {len(alive_pos)} neurons',
                     fontsize=12, fontweight='bold')

    print(f"Creating depth histogram animation...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    plt.tight_layout()
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close()

    print(f"Depth evolution animation saved to {output_path}")


def visualize_continuous_field(epochs: int = 30,
                                initial_neurons: int = 500,
                                volume_size: tuple = (100, 100, 100),
                                reorganize_every: int = 3):
    """
    Complete visualization pipeline for continuous neural field.
    """
    print("=" * 70)
    print("  CONTINUOUS NEURAL FIELD VISUALIZATION")
    print("  Watch Matter Condense from a Training Loop!")
    print("=" * 70)
    print()

    # Load data
    print("Loading MNIST data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, num_workers=2)

    # Create field
    print(f"\nCreating neural field with {initial_neurons} neurons...")
    field = ContinuousNeuralField(
        volume_size=volume_size,
        initial_neurons=initial_neurons,
        input_size=784,
        output_size=10,
        feature_dim=32,
        base_connection_radius=15.0
    )

    print(f"Initial state:\n{field.summary()}")

    # Train and record
    print(f"\nTraining for {epochs} epochs (recording every epoch)...")
    history = train_and_record(
        field,
        train_loader,
        test_loader,
        epochs=epochs,
        reorganize_every=reorganize_every,
        lr=0.01,
        geometric_weight=0.005,
        record_every=1,
        verbose=True
    )

    print(f"\nFinal state:\n{field.summary()}")

    # Create visualizations
    print("\n" + "=" * 70)
    print("  Creating Visualizations")
    print("=" * 70)

    # 1. Main 3D condensation animation
    print("\n1. Creating 3D condensation animation...")
    create_3d_condensation_animation(
        history,
        volume_size,
        output_path='neural_field_condensation.gif',
        fps=3,
        dpi=100
    )

    # 2. Evolution dashboard
    print("\n2. Creating evolution dashboard...")
    create_evolution_dashboard(
        history,
        volume_size,
        output_path='neural_field_evolution.png',
        dpi=150
    )

    # 3. Depth histogram animation
    print("\n3. Creating depth distribution animation...")
    create_depth_histogram_animation(
        history,
        volume_size,
        output_path='depth_evolution.gif',
        fps=3,
        dpi=100
    )

    print("\n" + "=" * 70)
    print("  VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - neural_field_condensation.gif  (3D evolution)")
    print("  - neural_field_evolution.png     (dashboard)")
    print("  - depth_evolution.gif            (structure emergence)")
    print()
    print(f"Final results:")
    print(f"  Neurons: {field.initial_neurons} → {field.alive_mask.sum().item()}")
    print(f"  Accuracy: {history['test_acc'][-1]:.1f}%")
    print()

    return field, history


if __name__ == "__main__":
    field, history = visualize_continuous_field(
        epochs=30,
        initial_neurons=500,
        volume_size=(100, 100, 100),
        reorganize_every=3
    )
