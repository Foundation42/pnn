"""
Epic visualization for MNIST Unleashed.

256 neurons, 4 layers, 200mm board.
Watch intelligence crystallize into its natural form!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D


# Epic color scheme
COLORS = {
    'background': '#0a0a0a',
    'board': '#1a472a',
    'copper': '#b87333',
    'gold': '#ffd700',
    'silkscreen': '#ffffff',
    'input_grid': '#1a5a3a',
    'output': '#4ecdc4',
    # Layer colors (gradient from input to output)
    'layer_0': '#ff6b6b',  # Red (near input)
    'layer_1': '#ffa07a',  # Light salmon
    'layer_2': '#98d8c8',  # Teal
    'layer_3': '#4ecdc4',  # Cyan (near output)
}

LAYER_COLORS = ['#ff6b6b', '#ffa07a', '#98d8c8', '#4ecdc4', '#a29bfe', '#74b9ff']


def create_unleashed_animation(history: dict,
                                board_size: tuple = (200, 200),
                                n_layers: int = 4,
                                input_grid_size: float = 50,
                                save_path: str = "unleashed_condensation.gif",
                                fps: int = 20,
                                duration: float = 15.0) -> None:
    """
    Create epic animation of 256-neuron condensation across 4 layers.
    """
    board_width, board_height = board_size
    n_frames = int(fps * duration)
    n_history = len(history['epochs'])

    # Input positions
    input_positions = []
    grid_start_x = 10
    grid_start_y = (board_height - input_grid_size) / 2
    pixel_spacing = input_grid_size / 28

    for row in range(28):
        for col in range(28):
            x = grid_start_x + col * pixel_spacing
            y = grid_start_y + (27 - row) * pixel_spacing
            input_positions.append([x, y])
    input_positions = np.array(input_positions)

    # Output positions
    output_positions = []
    output_x = board_width - 10
    output_spacing = (board_height - 40) / 9
    for i in range(10):
        y = 20 + i * output_spacing
        output_positions.append([output_x, y])
    output_positions = np.array(output_positions)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(COLORS['background'])

    # Layout: Main PCB view (large), metrics on right
    gs = fig.add_gridspec(3, 3, width_ratios=[2.5, 1, 1], height_ratios=[1, 1, 1],
                          hspace=0.25, wspace=0.25)

    # Main PCB view (spans all rows on left)
    ax_pcb = fig.add_subplot(gs[:, 0])
    ax_pcb.set_facecolor(COLORS['board'])

    # Metrics
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[0, 2])
    ax_layers = fig.add_subplot(gs[1, 1])
    ax_spacing = fig.add_subplot(gs[1, 2])
    ax_traj = fig.add_subplot(gs[2, 1:])

    for ax in [ax_loss, ax_acc, ax_layers, ax_spacing, ax_traj]:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333333')

    # Titles
    ax_loss.set_title('Loss', color='white', fontsize=10)
    ax_acc.set_title('Accuracy', color='white', fontsize=10)
    ax_layers.set_title('Layer Distribution', color='white', fontsize=10)
    ax_spacing.set_title('Min Spacing', color='white', fontsize=10)
    ax_traj.set_title('Neuron Trajectories (sample)', color='white', fontsize=10)

    # === PCB Setup ===
    ax_pcb.set_xlim(-5, board_width + 5)
    ax_pcb.set_ylim(-5, board_height + 5)
    ax_pcb.set_aspect('equal')
    ax_pcb.axis('off')

    # Board outline
    board_rect = Rectangle((0, 0), board_width, board_height,
                            linewidth=3, edgecolor=COLORS['copper'],
                            facecolor=COLORS['board'])
    ax_pcb.add_patch(board_rect)

    # Input region
    input_region = Rectangle((5, grid_start_y - 5), input_grid_size + 10, input_grid_size + 10,
                              facecolor=COLORS['input_grid'], alpha=0.3)
    ax_pcb.add_patch(input_region)

    # Title and info
    title_text = ax_pcb.text(board_width / 2, board_height + 3,
                             'MNIST UNLEASHED - 256 Neurons × 4 Layers',
                             ha='center', va='bottom', color=COLORS['silkscreen'],
                             fontsize=14, fontweight='bold')
    epoch_text = ax_pcb.text(5, -3, '', ha='left', va='top', color='white', fontsize=11)
    acc_text = ax_pcb.text(board_width - 5, -3, '', ha='right', va='top',
                           color=COLORS['output'], fontsize=11)

    # Static elements
    ax_pcb.scatter(input_positions[:, 0], input_positions[:, 1],
                   s=1, c=COLORS['copper'], alpha=0.4)
    ax_pcb.scatter(output_positions[:, 0], output_positions[:, 1],
                   s=80, c=COLORS['output'], edgecolors='white', linewidth=1.5, zorder=10)

    for i, (x, y) in enumerate(output_positions):
        ax_pcb.text(x + 3, y, str(i), ha='left', va='center',
                    color=COLORS['silkscreen'], fontsize=9, fontweight='bold')

    # Labels
    ax_pcb.text(10 + input_grid_size/2, grid_start_y - 8, '28×28 Input',
                ha='center', color=COLORS['silkscreen'], fontsize=9)

    # Hidden neurons scatter (will update)
    hidden_scatters = []

    # Initialize line objects
    line_loss, = ax_loss.plot([], [], color='#ff6b6b', linewidth=2)
    line_class, = ax_loss.plot([], [], color='#4ecdc4', linewidth=1.5, alpha=0.7)
    ax_loss.set_xlabel('Epoch', color='white', fontsize=8)

    line_train, = ax_acc.plot([], [], color='#4ecdc4', linewidth=2, label='Train')
    line_test, = ax_acc.plot([], [], color='#ffe66d', linewidth=2, label='Test')
    ax_acc.axhline(98, color='#666666', linestyle='--', alpha=0.5)
    ax_acc.legend(loc='lower right', facecolor='#2a2a2a', labelcolor='white', fontsize=7)
    ax_acc.set_xlabel('Epoch', color='white', fontsize=8)

    # Layer bar chart
    layer_bars = ax_layers.bar(range(n_layers), [0]*n_layers,
                               color=LAYER_COLORS[:n_layers], edgecolor='white', linewidth=0.5)
    ax_layers.set_xticks(range(n_layers))
    ax_layers.set_xticklabels([f'L{i}' for i in range(n_layers)])
    ax_layers.set_ylim(0, 100)

    # Spacing line
    line_spacing, = ax_spacing.plot([], [], color='#ff6b6b', linewidth=2)
    ax_spacing.axhline(0.15, color='#4ecdc4', linestyle='--', alpha=0.7, label='Min allowed')
    ax_spacing.legend(loc='upper right', facecolor='#2a2a2a', labelcolor='white', fontsize=7)
    ax_spacing.set_xlabel('Epoch', color='white', fontsize=8)
    ax_spacing.set_ylabel('mm', color='white', fontsize=8)

    # Trajectory (sample 32 neurons)
    n_hidden = history['hidden_positions'][0].shape[0]
    sample_idx = np.linspace(0, n_hidden-1, min(32, n_hidden), dtype=int)
    traj_colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sample_idx)))
    traj_lines = []
    for i, _ in enumerate(sample_idx):
        line, = ax_traj.plot([], [], color=traj_colors[i], linewidth=0.8, alpha=0.6)
        traj_lines.append(line)

    ax_traj.set_xlim(60, 185)
    ax_traj.set_ylim(0, board_height)
    ax_traj.set_xlabel('X (mm)', color='white', fontsize=8)
    ax_traj.set_ylabel('Y (mm)', color='white', fontsize=8)

    def interpolate_frame(frame_idx):
        progress = frame_idx / max(1, n_frames - 1)
        # Ease-in-out
        if progress < 0.5:
            progress = 2 * progress * progress
        else:
            progress = 1 - pow(-2 * progress + 2, 2) / 2
        return min(int(progress * (n_history - 1)), n_history - 1)

    def update(frame):
        nonlocal hidden_scatters

        idx = interpolate_frame(frame)
        epoch = history['epochs'][idx]

        # Get current state
        hidden_pos = history['hidden_positions'][idx]
        hidden_layers = history['hidden_layers'][idx]

        # Clear old scatters
        for sc in hidden_scatters:
            sc.remove()
        hidden_scatters = []

        # Draw neurons colored by layer
        for layer in range(n_layers):
            mask = (hidden_layers >= layer - 0.5) & (hidden_layers < layer + 0.5)
            if mask.sum() > 0:
                # Size varies by layer (deeper = larger)
                size = 30 + layer * 10
                sc = ax_pcb.scatter(hidden_pos[mask, 0], hidden_pos[mask, 1],
                                    s=size, c=LAYER_COLORS[layer],
                                    edgecolors='white', linewidth=0.5,
                                    alpha=0.8, zorder=5 + layer)
                hidden_scatters.append(sc)

        # Update text
        epoch_text.set_text(f'Epoch: {epoch}')
        if idx < len(history['test_acc']):
            acc_text.set_text(f'Test: {history["test_acc"][idx]:.1f}%')

        # Update plots
        data_idx = min(idx, len(history['train_loss']) - 1)
        epochs_data = history['epochs'][1:data_idx + 2]

        if len(epochs_data) > 0 and data_idx < len(history['train_loss']):
            losses = history['train_loss'][:data_idx + 1]
            class_losses = history['classification_loss'][:data_idx + 1]
            line_loss.set_data(epochs_data, losses)
            line_class.set_data(epochs_data, class_losses)
            if len(epochs_data) > 1:
                ax_loss.set_xlim(0, max(epochs_data) * 1.05)
                ax_loss.set_ylim(0, max(max(losses), 0.1) * 1.1)

            train_accs = history['train_acc'][:data_idx + 1]
            test_accs = history['test_acc'][:data_idx + 1]
            line_train.set_data(epochs_data, train_accs)
            line_test.set_data(epochs_data, test_accs)
            if len(epochs_data) > 1:
                ax_acc.set_xlim(0, max(epochs_data) * 1.05)
                ax_acc.set_ylim(max(80, min(min(train_accs), min(test_accs)) - 2), 100.5)

            # Spacing
            spacings = history['spacing_min'][:data_idx + 1]
            line_spacing.set_data(epochs_data, spacings)
            if len(epochs_data) > 1:
                ax_spacing.set_xlim(0, max(epochs_data) * 1.05)
                ax_spacing.set_ylim(0, max(spacings) * 1.2)

        # Layer distribution
        layer_counts = [((hidden_layers >= i - 0.5) & (hidden_layers < i + 0.5)).sum()
                        for i in range(n_layers)]
        for bar, count in zip(layer_bars, layer_counts):
            bar.set_height(count)
        ax_layers.set_ylim(0, max(layer_counts) * 1.2 if max(layer_counts) > 0 else 100)

        # Trajectories
        for i, neuron_idx in enumerate(sample_idx):
            positions = np.array([history['hidden_positions'][j][neuron_idx]
                                  for j in range(idx + 1)])
            traj_lines[i].set_data(positions[:, 0], positions[:, 1])

        return hidden_scatters + [epoch_text, acc_text, line_loss, line_class,
                                   line_train, line_test, line_spacing] + list(layer_bars) + traj_lines

    print(f"Creating animation ({n_frames} frames)...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    print(f"Saving to {save_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    print("Done!")

    plt.close(fig)


def plot_unleashed_layout(model, save_path: str = "unleashed_layout.png"):
    """
    Create detailed multi-layer layout visualization.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor(COLORS['background'])

    board_width = model.board_width
    board_height = model.board_height

    # === Left: Top-down view (all layers overlaid) ===
    ax1 = axes[0]
    ax1.set_facecolor(COLORS['board'])

    board_rect = Rectangle((0, 0), board_width, board_height,
                            linewidth=3, edgecolor=COLORS['copper'],
                            facecolor=COLORS['board'])
    ax1.add_patch(board_rect)

    # Input grid
    input_grid_size = model.input_grid_size
    grid_start_y = (board_height - input_grid_size) / 2
    input_region = Rectangle((5, grid_start_y - 5), input_grid_size + 10, input_grid_size + 10,
                              facecolor=COLORS['input_grid'], alpha=0.3)
    ax1.add_patch(input_region)

    # Get positions
    hidden_pos = model.hidden_positions.detach().cpu().numpy()
    hidden_layers = model.hidden_layers.detach().cpu().numpy()

    # Draw neurons by layer
    for layer in range(model.n_layers):
        mask = (hidden_layers >= layer - 0.5) & (hidden_layers < layer + 0.5)
        if mask.sum() > 0:
            size = 40 + layer * 15
            ax1.scatter(hidden_pos[mask, 0], hidden_pos[mask, 1],
                        s=size, c=LAYER_COLORS[layer],
                        edgecolors='white', linewidth=1,
                        alpha=0.9, zorder=5 + layer,
                        label=f'Layer {layer} ({mask.sum()})')

    # Input/output
    input_positions = model.input_positions.cpu().numpy()
    output_positions = model.output_positions.cpu().numpy()

    ax1.scatter(input_positions[:, 0], input_positions[:, 1],
                s=2, c=COLORS['copper'], alpha=0.5)
    ax1.scatter(output_positions[:, 0], output_positions[:, 1],
                s=100, c=COLORS['output'], edgecolors='white', linewidth=2, zorder=10)

    for i, (x, y) in enumerate(output_positions):
        ax1.text(x + 4, y, str(i), ha='left', va='center',
                 color=COLORS['silkscreen'], fontsize=10, fontweight='bold')

    ax1.set_xlim(-5, board_width + 5)
    ax1.set_ylim(-5, board_height + 5)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', facecolor='#2a2a2a', edgecolor='#555555',
               labelcolor='white', fontsize=9)
    ax1.set_title('MNIST UNLEASHED - Top View (All Layers)', color='white', fontsize=14)
    ax1.axis('off')

    # === Right: Layer histogram + stats ===
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a1a')

    # Layer distribution
    layer_counts = [((hidden_layers >= i - 0.5) & (hidden_layers < i + 0.5)).sum()
                    for i in range(model.n_layers)]

    bars = ax2.bar(range(model.n_layers), layer_counts,
                   color=LAYER_COLORS[:model.n_layers], edgecolor='white', linewidth=2)

    ax2.set_xticks(range(model.n_layers))
    ax2.set_xticklabels([f'Layer {i}' for i in range(model.n_layers)], fontsize=11)
    ax2.set_ylabel('Number of Neurons', color='white', fontsize=12)
    ax2.tick_params(colors='white')
    ax2.set_title('Layer Distribution', color='white', fontsize=14)

    for spine in ax2.spines.values():
        spine.set_color('#444444')

    # Add count labels
    for bar, count in zip(bars, layer_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(count), ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')

    # Stats text
    stats = model.get_layout_stats()
    stats_text = f"""
Architecture: 784 → {model.hidden_size} → 10
Board: {board_width}mm × {board_height}mm
Layers: {model.n_layers}

Min spacing: {stats['min_spacing']:.3f}mm
Mean spacing: {stats['mean_spacing']:.1f}mm
Total traces: {stats['total_trace_length_m']:.1f}m

X range: {stats['hidden_x_range'][0]:.0f} - {stats['hidden_x_range'][1]:.0f}mm
Y range: {stats['hidden_y_range'][0]:.0f} - {stats['hidden_y_range'][1]:.0f}mm
Mean layer: {stats['mean_layer']:.2f}
"""
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
             ha='right', va='top', color='white', fontsize=10,
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#2a2a2a', edgecolor='#555555'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    print(f"Layout saved to {save_path}")
    plt.close(fig)


def plot_unleashed_progress(history: dict, save_path: str = "unleashed_progress.png"):
    """
    Create comprehensive training progress visualization.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle('MNIST UNLEASHED - Training Progress',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    for ax in axes.flat:
        ax.set_facecolor('#2a2a2a')
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#444444')
        ax.grid(True, alpha=0.2, color='white')

    n_data = len(history['train_loss'])
    epochs = history['epochs'][1:n_data + 1]

    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], color='#ff6b6b', linewidth=2, label='Total')
    ax1.plot(epochs, history['classification_loss'], color='#4ecdc4', linewidth=2, label='Classification')
    ax1.plot(epochs, history['layout_loss'], color='#ffe66d', linewidth=2, linestyle='--', label='Layout')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.set_title('Loss Curves', color='white', fontsize=12)
    ax1.legend(facecolor='#3a3a3a', labelcolor='white', fontsize=9)
    ax1.set_yscale('log')

    # 2. Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], color='#4ecdc4', linewidth=2, label='Train')
    ax2.plot(epochs, history['test_acc'], color='#ffe66d', linewidth=2, label='Test')
    ax2.axhline(98, color='#666666', linestyle='--', alpha=0.5, label='98% target')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Accuracy (%)', color='white')
    ax2.set_title('Accuracy', color='white', fontsize=12)
    ax2.legend(facecolor='#3a3a3a', labelcolor='white', fontsize=9, loc='lower right')
    ax2.set_ylim(85, 100.5)

    # 3. Minimum spacing over time
    ax3 = axes[0, 2]
    ax3.plot(epochs, history['spacing_min'], color='#ff6b6b', linewidth=2)
    ax3.axhline(0.15, color='#4ecdc4', linestyle='--', linewidth=2, label='Min allowed (0.15mm)')
    ax3.set_xlabel('Epoch', color='white')
    ax3.set_ylabel('Spacing (mm)', color='white')
    ax3.set_title('Minimum Neuron Spacing', color='white', fontsize=12)
    ax3.legend(facecolor='#3a3a3a', labelcolor='white', fontsize=9)

    # 4. Neuron trajectories
    ax4 = axes[1, 0]
    positions = np.array(history['hidden_positions'])
    n_hidden = positions.shape[1]

    # Sample neurons
    sample_idx = np.linspace(0, n_hidden - 1, min(64, n_hidden), dtype=int)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sample_idx)))

    for i, idx in enumerate(sample_idx):
        traj = positions[:, idx, :]
        ax4.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.5, linewidth=0.8)
        ax4.scatter(traj[0, 0], traj[0, 1], color=colors[i], s=15, marker='x', linewidth=1)
        ax4.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=25, edgecolor='white', linewidth=0.5)

    ax4.set_xlabel('X (mm)', color='white')
    ax4.set_ylabel('Y (mm)', color='white')
    ax4.set_title('Hidden Neuron Trajectories (64 sample)', color='white', fontsize=12)
    ax4.set_xlim(60, 190)
    ax4.set_ylim(0, 200)

    # 5. Layer evolution
    ax5 = axes[1, 1]
    layers_history = np.array(history['hidden_layers'])
    n_layers = 4

    for layer in range(n_layers):
        counts = [(layers_history[t] >= layer - 0.5).sum() - (layers_history[t] >= layer + 0.5).sum()
                  for t in range(len(layers_history))]
        ax5.plot(history['epochs'][:len(counts)], counts,
                 color=LAYER_COLORS[layer], linewidth=2, label=f'Layer {layer}')

    ax5.set_xlabel('Epoch', color='white')
    ax5.set_ylabel('Neuron Count', color='white')
    ax5.set_title('Layer Distribution Over Time', color='white', fontsize=12)
    ax5.legend(facecolor='#3a3a3a', labelcolor='white', fontsize=9)

    # 6. Total trace length
    ax6 = axes[1, 2]
    ax6.plot(epochs, history['trace_length'], color='#ffd700', linewidth=2)
    ax6.set_xlabel('Epoch', color='white')
    ax6.set_ylabel('Trace Length (m)', color='white')
    ax6.set_title('Total Trace Length', color='white', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    print(f"Progress chart saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    from train_unleashed import train_unleashed, save_model

    print("Training MNIST Unleashed...")
    model, history = train_unleashed(epochs=50, hidden_size=256, verbose=True)

    print("\nGenerating visualizations...")
    create_unleashed_animation(history, save_path="unleashed_condensation.gif", duration=15)
    plot_unleashed_layout(model, save_path="unleashed_layout.png")
    plot_unleashed_progress(history, save_path="unleashed_progress.png")

    save_model(model, "mnist_unleashed.pt")
    print("\nComplete!")
