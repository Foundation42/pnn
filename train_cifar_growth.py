"""
CIFAR-10 Growing Neural Field

MNIST needed only 18 neurons for 95.8% accuracy.
How many neurons does COLOR VISION need?

CIFAR-10:
- 32x32x3 = 3072 inputs (4x more than MNIST!)
- 10 classes: airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck
- Much harder than MNIST - state of art is ~99% with huge networks

Let's see what the MINIMAL architecture looks like!
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from growing_field import GrowingNeuralField, train_growing_field


def train_cifar_growing(seed_neurons: int = 20,
                        max_neurons: int = 500,
                        epochs: int = 50,
                        split_threshold: float = 0.4,
                        verbose: bool = True):
    """
    Train a growing neural field on CIFAR-10.
    """
    print("=" * 70)
    print("  CIFAR-10 GROWING NEURAL FIELD")
    print("  How many neurons does color vision need?")
    print("=" * 70)
    print()

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=500, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Input size: 32x32x3 = 3072")
    print()

    # Create field - CIFAR needs more capacity
    print(f"Creating field with {seed_neurons} seed neurons...")
    field = GrowingNeuralField(
        volume_size=(100, 100, 100),
        seed_neurons=seed_neurons,
        max_neurons=max_neurons,
        input_size=3072,  # 32x32x3
        output_size=10,
        feature_dim=64,  # Larger features for color
        base_connection_radius=25.0,  # Larger radius
        split_threshold=split_threshold
    )

    print(f"Initial state:\n{field.summary()}")

    # Custom training with recording
    device = field.device
    field = field.to(device)

    optimizer = torch.optim.AdamW(field.parameters(), lr=0.005)  # Lower LR for harder task
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'test_acc': [],
        'n_neurons': [],
        'n_splits': [],
        'positions': [],
        'alive_masks': [],
        'epochs': [],
    }

    # Record initial state
    with torch.no_grad():
        history['positions'].append(field.positions.detach().cpu().numpy().copy())
        history['alive_masks'].append(field.alive_mask.detach().cpu().numpy().copy())
        history['n_neurons'].append(field.n_alive)
        history['n_splits'].append(0)
        history['epochs'].append(0)

    best_acc = 0

    if verbose:
        print("\n" + "=" * 70)
        print("  Training - Watch neurons multiply!")
        print("=" * 70 + "\n")

    for epoch in range(epochs):
        field.train()
        epoch_loss = 0.0
        n_batches = 0

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            # Flatten CIFAR images
            data = data.view(-1, 3072).to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = field(data, n_iterations=3)  # More iterations for harder task

            class_loss = F.cross_entropy(output, target)
            geo_loss = field.geometric_loss()
            loss = class_loss + 0.005 * geo_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(field.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Split every 2 epochs
        n_new_splits = 0
        if (epoch + 1) % 2 == 0:
            # More aggressive splitting in early epochs
            force = (epoch < epochs // 2 and field.n_alive < max_neurons // 3)
            n_new_splits = field.split_overloaded_neurons(force_split=force)

        # Test accuracy
        field.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(-1, 3072).to(device)
                target = target.to(device)
                output = field(data, n_iterations=3)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        test_acc = 100.0 * correct / total
        best_acc = max(best_acc, test_acc)

        # Record history
        history['train_loss'].append(epoch_loss / n_batches)
        history['test_acc'].append(test_acc)
        history['n_neurons'].append(field.n_alive)
        history['n_splits'].append(field.total_splits)

        # Record positions
        with torch.no_grad():
            history['positions'].append(field.positions.detach().cpu().numpy().copy())
            history['alive_masks'].append(field.alive_mask.detach().cpu().numpy().copy())
            history['epochs'].append(epoch + 1)

        elapsed = time.time() - start_time

        if verbose:
            stats = field.get_stats()
            print(f"Epoch {epoch+1:3d}/{epochs} [{elapsed:.1f}s]")
            print(f"  Loss: {epoch_loss/n_batches:.4f}, Test Acc: {test_acc:.1f}% (best: {best_acc:.1f}%)")
            print(f"  Neurons: {field.n_alive} (splits: +{n_new_splits}, total: {field.total_splits})")
            print(f"  Volume: {stats.volume_utilization:.1%}, Clusters: {stats.n_clusters}")
            print()

    print("=" * 70)
    print("  Training Complete!")
    print("=" * 70)
    print(f"\n{field.summary()}")

    return field, history, best_acc


def create_cifar_visualization(field, history, output_path='cifar_growth.png'):
    """Create visualization of CIFAR growth."""
    fig = plt.figure(figsize=(14, 10))

    fig.suptitle('CIFAR-10 Growing Neural Field\nHow Many Neurons Does Color Vision Need?',
                 fontsize=14, fontweight='bold', y=0.98)

    # 1. Final 3D structure
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    alive_idx = torch.where(field.alive_mask)[0]
    pos = field.positions[alive_idx].detach().cpu().numpy()

    if len(pos) > 0:
        colors = plt.cm.plasma(pos[:, 0] / 100)
        sizes = 100 - len(pos)
        sizes = max(30, min(100, sizes))
        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=colors, s=sizes, alpha=0.8)

    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100); ax1.set_zlim(0, 100)
    ax1.set_xlabel('X (Depth)'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title(f'Final Structure: {len(pos)} neurons', fontsize=11)

    # 2. Neuron growth curve
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history['epochs'], history['n_neurons'], 'r-', linewidth=2)
    ax2.fill_between(history['epochs'], history['n_neurons'], alpha=0.3, color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Number of Neurons')
    ax2.set_title('Neural Mitosis (Growth)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add growth annotation
    start = history['n_neurons'][0]
    end = history['n_neurons'][-1]
    ax2.text(0.95, 0.95, f'{start} → {end}\n({end/start:.1f}x growth)',
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Accuracy curve
    ax3 = fig.add_subplot(2, 2, 3)
    epochs = list(range(1, len(history['test_acc']) + 1))
    ax3.plot(epochs, history['test_acc'], 'g-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.set_title('CIFAR-10 Accuracy', fontsize=11)
    ax3.grid(True, alpha=0.3)

    best_acc = max(history['test_acc'])
    ax3.axhline(y=best_acc, color='g', linestyle='--', alpha=0.5)
    ax3.text(0.95, 0.05, f'Best: {best_acc:.1f}%',
             transform=ax3.transAxes, ha='right', va='bottom',
             fontsize=11, fontweight='bold')

    # 4. Comparison with MNIST
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Compute stats
    stats = field.get_stats()

    comparison_text = f"""
CIFAR-10 vs MNIST COMPARISON

                 MNIST        CIFAR-10
                 -----        --------
Input Size:      784          3072 (4x)
Final Neurons:   18           {end}
Best Accuracy:   95.8%        {best_acc:.1f}%
Growth Factor:   1.8x         {end/start:.1f}x

EFFICIENCY:
MNIST:   {95.8/18:.2f}% per neuron
CIFAR:   {best_acc/end:.2f}% per neuron

STRUCTURE:
Volume Used:     ~12%         {stats.volume_utilization:.1%}
Clusters:        6            {stats.n_clusters}

CONCLUSION:
Color vision needs {end/18:.1f}x more neurons than grayscale!
But still way less than traditional networks.
"""

    ax4.text(0.1, 0.95, comparison_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to {output_path}")


if __name__ == "__main__":
    # Train CIFAR
    field, history, best_acc = train_cifar_growing(
        seed_neurons=20,
        max_neurons=500,
        epochs=50,
        split_threshold=0.4,
        verbose=True
    )

    # Create visualization
    create_cifar_visualization(field, history, 'cifar_growth.png')

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"""
    CIFAR-10 Growing Neural Field:
      Seeds: 20 neurons
      Final: {field.n_alive} neurons
      Growth: {field.n_alive / 20:.1f}x
      Best Accuracy: {best_acc:.1f}%

    Compare to MNIST:
      MNIST: 18 neurons → 95.8%
      CIFAR: {field.n_alive} neurons → {best_acc:.1f}%

    Color vision needs {field.n_alive / 18:.1f}x more neurons than grayscale!
    """)
