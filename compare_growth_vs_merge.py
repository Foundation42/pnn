"""
Compare Growth vs Merge Approaches

The Ultimate Question: Does the optimal geometry exist as an attractor?

If so, both approaches should converge to similar structures:
- Merge: 500 → ~400 neurons (prune redundancy)
- Growth: 10 → ??? neurons (grow complexity)

Let's find out!
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

from continuous_field import ContinuousNeuralField, train_continuous_field
from growing_field import GrowingNeuralField, train_growing_field


def run_comparison(epochs: int = 30,
                   merge_neurons: int = 500,
                   seed_neurons: int = 10,
                   verbose: bool = True):
    """
    Run both approaches and compare results.
    """
    print("=" * 70)
    print("  GROWTH vs MERGE: The Attractor Hypothesis")
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

    results = {}

    # === MERGE APPROACH ===
    print("\n" + "=" * 70)
    print("  APPROACH 1: MERGE (Start Dense, Prune Down)")
    print("=" * 70)

    merge_field = ContinuousNeuralField(
        volume_size=(100, 100, 100),
        initial_neurons=merge_neurons,
        input_size=784,
        output_size=10,
        feature_dim=32,
        base_connection_radius=15.0
    )

    print(f"Starting with {merge_neurons} neurons...")

    merge_history = train_continuous_field(
        merge_field,
        train_loader,
        test_loader,
        epochs=epochs,
        reorganize_every=3,
        lr=0.01,
        geometric_weight=0.005,
        verbose=verbose
    )

    results['merge'] = {
        'field': merge_field,
        'history': merge_history,
        'start_neurons': merge_neurons,
        'end_neurons': merge_field.alive_mask.sum().item(),
        'final_acc': merge_history['test_acc'][-1],
    }

    # === GROWTH APPROACH ===
    print("\n" + "=" * 70)
    print("  APPROACH 2: GROWTH (Start Sparse, Split Up)")
    print("=" * 70)

    growth_field = GrowingNeuralField(
        volume_size=(100, 100, 100),
        seed_neurons=seed_neurons,
        max_neurons=600,
        input_size=784,
        output_size=10,
        feature_dim=32,
        base_connection_radius=15.0,
        split_threshold=0.5  # Lower threshold for more splitting
    )

    print(f"Starting with {seed_neurons} seed neurons...")

    growth_history = train_growing_field(
        growth_field,
        train_loader,
        test_loader,
        epochs=epochs,
        split_every=2,
        lr=0.01,
        geometric_weight=0.005,
        verbose=verbose
    )

    results['growth'] = {
        'field': growth_field,
        'history': growth_history,
        'start_neurons': seed_neurons,
        'end_neurons': growth_field.n_alive,
        'final_acc': growth_history['test_acc'][-1],
    }

    return results


def create_comparison_visualization(results: dict, output_path: str = 'growth_vs_merge.png'):
    """
    Create side-by-side comparison visualization.
    """
    fig = plt.figure(figsize=(16, 12))

    fig.suptitle('Growth vs Merge: Two Paths to the Same Attractor?',
                 fontsize=14, fontweight='bold', y=0.98)

    # Extract data
    merge = results['merge']
    growth = results['growth']

    merge_field = merge['field']
    growth_field = growth['field']

    # 1. Final 3D structures side by side
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')

    # Merge final structure
    merge_alive = torch.where(merge_field.alive_mask)[0]
    merge_pos = merge_field.positions[merge_alive].detach().cpu().numpy()
    if len(merge_pos) > 0:
        colors = plt.cm.viridis(merge_pos[:, 0] / 100)
        ax1.scatter(merge_pos[:, 0], merge_pos[:, 1], merge_pos[:, 2],
                    c=colors, s=15, alpha=0.6)
    ax1.set_title(f'MERGE: {merge["start_neurons"]} → {merge["end_neurons"]} neurons\n'
                  f'Acc: {merge["final_acc"]:.1f}%', fontsize=10)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 100); ax1.set_zlim(0, 100)

    # Growth final structure
    growth_alive = torch.where(growth_field.alive_mask)[0]
    growth_pos = growth_field.positions[growth_alive].detach().cpu().numpy()
    if len(growth_pos) > 0:
        colors = plt.cm.plasma(growth_pos[:, 0] / 100)
        ax2.scatter(growth_pos[:, 0], growth_pos[:, 1], growth_pos[:, 2],
                    c=colors, s=50, alpha=0.8)
    ax2.set_title(f'GROWTH: {growth["start_neurons"]} → {growth["end_neurons"]} neurons\n'
                  f'Acc: {growth["final_acc"]:.1f}%', fontsize=10)
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 100); ax2.set_zlim(0, 100)

    # 2. Neuron count over time
    ax3 = fig.add_subplot(2, 3, 3)

    merge_epochs = list(range(len(merge['history']['n_neurons'])))
    growth_epochs = list(range(len(growth['history']['n_neurons'])))

    ax3.plot(merge_epochs, merge['history']['n_neurons'], 'b-', linewidth=2, label='Merge')
    ax3.plot(growth_epochs, growth['history']['n_neurons'], 'r-', linewidth=2, label='Growth')
    ax3.axhline(y=merge['end_neurons'], color='b', linestyle='--', alpha=0.5)
    ax3.axhline(y=growth['end_neurons'], color='r', linestyle='--', alpha=0.5)

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Number of Neurons')
    ax3.set_title('Neuron Count Evolution', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 3. Accuracy comparison
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(range(1, len(merge['history']['test_acc']) + 1),
             merge['history']['test_acc'], 'b-', linewidth=2, label='Merge')
    ax4.plot(range(1, len(growth['history']['test_acc']) + 1),
             growth['history']['test_acc'], 'r-', linewidth=2, label='Growth')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_title('Accuracy Comparison', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 4. Depth distribution comparison (X-axis histogram)
    ax5 = fig.add_subplot(2, 3, 5)

    if len(merge_pos) > 0:
        ax5.hist(merge_pos[:, 0], bins=20, range=(0, 100), alpha=0.5,
                 color='blue', label=f'Merge (n={len(merge_pos)})')
    if len(growth_pos) > 0:
        ax5.hist(growth_pos[:, 0], bins=20, range=(0, 100), alpha=0.5,
                 color='red', label=f'Growth (n={len(growth_pos)})')

    ax5.set_xlabel('X Position (Depth)')
    ax5.set_ylabel('Neuron Count')
    ax5.set_title('Depth Distribution Comparison', fontsize=10)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 5. Summary statistics
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Compute volume utilization for both
    if len(merge_pos) > 0:
        merge_vol = ((merge_pos[:, 0].max() - merge_pos[:, 0].min()) *
                     (merge_pos[:, 1].max() - merge_pos[:, 1].min()) *
                     (merge_pos[:, 2].max() - merge_pos[:, 2].min())) / (100**3)
    else:
        merge_vol = 0

    if len(growth_pos) > 0:
        growth_vol = ((growth_pos[:, 0].max() - growth_pos[:, 0].min()) *
                      (growth_pos[:, 1].max() - growth_pos[:, 1].min()) *
                      (growth_pos[:, 2].max() - growth_pos[:, 2].min())) / (100**3)
    else:
        growth_vol = 0

    summary_text = f"""
COMPARISON SUMMARY

              MERGE          GROWTH
              -----          ------
Start:        {merge['start_neurons']:4d}           {growth['start_neurons']:4d}
End:          {merge['end_neurons']:4d}           {growth['end_neurons']:4d}
Change:       {merge['end_neurons'] - merge['start_neurons']:+4d}           {growth['end_neurons'] - growth['start_neurons']:+4d}

Accuracy:     {merge['final_acc']:5.1f}%         {growth['final_acc']:5.1f}%
Volume:       {merge_vol*100:5.1f}%         {growth_vol*100:5.1f}%

EFFICIENCY (Acc per Neuron):
Merge:   {merge['final_acc']/merge['end_neurons']:.2f}% per neuron
Growth:  {growth['final_acc']/growth['end_neurons']:.2f}% per neuron

CONCLUSION:
{"Both converge to similar volume (~" + f"{(merge_vol+growth_vol)/2*100:.0f}%)!" if abs(merge_vol - growth_vol) < 0.1 else "Different final structures"}
"""

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nComparison saved to {output_path}")


def create_growth_animation(results: dict, output_path: str = 'growth_evolution.gif',
                            fps: int = 3):
    """
    Create animation of the growth process.
    """
    growth = results['growth']
    history = growth['history']

    positions_list = history['positions']
    alive_masks_list = history['alive_masks']
    epochs = history['epochs']

    n_frames = len(positions_list)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    def update(frame_idx):
        ax1.clear()
        ax2.clear()

        positions = positions_list[frame_idx]
        alive_mask = alive_masks_list[frame_idx]
        epoch = epochs[frame_idx]

        alive_pos = positions[alive_mask]
        n_alive = len(alive_pos)

        if n_alive > 0:
            colors = plt.cm.plasma(alive_pos[:, 0] / 100)
            sizes = 100 - n_alive  # Larger dots when fewer neurons
            sizes = max(20, min(100, sizes))

            ax1.scatter(alive_pos[:, 0], alive_pos[:, 1], alive_pos[:, 2],
                        c=colors, s=sizes, alpha=0.8)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_zlim(0, 100)
        ax1.set_xlabel('X (Depth)')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Epoch {epoch}: {n_alive} Neurons\n'
                      f'(Started with {growth["start_neurons"]} seeds)',
                      fontsize=11, fontweight='bold')

        # Neuron count plot
        current_epochs = history['epochs'][:frame_idx+1]
        current_neurons = history['n_neurons'][:frame_idx+1]

        ax2.plot(current_epochs, current_neurons, 'r-', linewidth=2)
        ax2.fill_between(current_epochs, current_neurons, alpha=0.3, color='red')

        ax2.set_xlim(0, max(epochs))
        ax2.set_ylim(0, max(history['n_neurons']) * 1.2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number of Neurons')
        ax2.set_title('Growth Through Mitosis', fontsize=11)
        ax2.grid(True, alpha=0.3)

        # Add growth factor annotation
        if n_alive > 0:
            growth_factor = n_alive / growth['start_neurons']
            ax2.text(0.95, 0.95, f'{growth_factor:.1f}x growth',
                     transform=ax2.transAxes, ha='right', va='top',
                     fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        fig.suptitle('NEURAL MITOSIS: Seeds Growing into Structure',
                     fontsize=12, fontweight='bold')

    print(f"Creating growth animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    plt.tight_layout()
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=100)
    plt.close()

    print(f"Growth animation saved to {output_path}")


def main():
    """Run the complete comparison."""
    print("\n" + "=" * 70)
    print("  THE ATTRACTOR HYPOTHESIS")
    print("  Do both paths lead to the same geometry?")
    print("=" * 70)
    print()

    # Run both approaches
    results = run_comparison(
        epochs=30,
        merge_neurons=500,
        seed_neurons=10,
        verbose=True
    )

    # Create visualizations
    print("\n" + "=" * 70)
    print("  Creating Visualizations")
    print("=" * 70)

    create_comparison_visualization(results, 'growth_vs_merge.png')
    create_growth_animation(results, 'growth_evolution.gif', fps=3)

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)

    merge = results['merge']
    growth = results['growth']

    print(f"""
    MERGE APPROACH:
      {merge['start_neurons']} → {merge['end_neurons']} neurons
      Final accuracy: {merge['final_acc']:.1f}%
      Efficiency: {merge['final_acc']/merge['end_neurons']:.2f}% per neuron

    GROWTH APPROACH:
      {growth['start_neurons']} → {growth['end_neurons']} neurons
      Final accuracy: {growth['final_acc']:.1f}%
      Efficiency: {growth['final_acc']/growth['end_neurons']:.2f}% per neuron

    WINNER: {"GROWTH" if growth['final_acc']/growth['end_neurons'] > merge['final_acc']/merge['end_neurons'] else "MERGE"} (higher efficiency)

    Output files:
      - growth_vs_merge.png
      - growth_evolution.gif
    """)

    return results


if __name__ == "__main__":
    results = main()
