"""
Visualization for MNIST Physical Neural Network.

Shows the beautiful condensation of 64 hidden neurons
while learning to classify handwritten digits.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection, EllipseCollection
import matplotlib.colors as mcolors


# PCB-inspired color scheme
COLORS = {
    'background': '#0a0a0a',
    'board': '#1a472a',
    'copper': '#b87333',
    'gold': '#ffd700',
    'solder_mask': '#2d5a3d',
    'silkscreen': '#ffffff',
    'input_grid': '#1a5a3a',
    'hidden': '#ff6b6b',
    'output': '#4ecdc4',
    'trace_pos': '#ffd700',
    'trace_neg': '#ff8c00',
}


def create_mnist_animation(history: dict,
                           board_size: tuple = (100, 100),
                           input_grid_size: float = 30,
                           save_path: str = "mnist_condensation.gif",
                           fps: int = 20,
                           duration: float = 12.0) -> None:
    """
    Create animation of MNIST network condensation.

    Args:
        history: Training history from train_mnist()
        board_size: Board dimensions in mm
        input_grid_size: Size of input grid in mm
        save_path: Output file path
        fps: Frames per second
        duration: Animation duration in seconds
    """
    board_width, board_height = board_size
    n_frames = int(fps * duration)
    n_history = len(history['epochs'])

    # Generate input positions (28x28 grid)
    input_positions = []
    grid_start_x = 5
    grid_start_y = (board_height - input_grid_size) / 2
    pixel_spacing = input_grid_size / 28

    for row in range(28):
        for col in range(28):
            x = grid_start_x + col * pixel_spacing
            y = grid_start_y + (27 - row) * pixel_spacing
            input_positions.append([x, y])
    input_positions = np.array(input_positions)

    # Output positions (right edge)
    output_positions = []
    output_x = board_width - 5
    output_spacing = (board_height - 20) / 9
    for i in range(10):
        y = 10 + i * output_spacing
        output_positions.append([output_x, y])
    output_positions = np.array(output_positions)

    # Create figure
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS['background'])

    # Layout: PCB view on left (larger), metrics on right
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1],
                          hspace=0.3, wspace=0.3)

    # Main PCB view
    ax_pcb = fig.add_subplot(gs[:, 0])
    ax_pcb.set_facecolor(COLORS['board'])

    # Loss curve
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_loss.set_facecolor('#1a1a1a')

    # Accuracy curve
    ax_acc = fig.add_subplot(gs[0, 2])
    ax_acc.set_facecolor('#1a1a1a')

    # Hidden neuron trajectory
    ax_traj = fig.add_subplot(gs[1, 1:])
    ax_traj.set_facecolor('#1a1a1a')

    # Style helper
    def style_axis(ax, title):
        ax.set_title(title, color='white', fontsize=11, pad=8)
        ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333333')

    style_axis(ax_loss, 'Loss')
    style_axis(ax_acc, 'Accuracy')
    style_axis(ax_traj, 'Hidden Neuron Trajectories')

    # === Setup PCB view ===
    ax_pcb.set_xlim(-2, board_width + 2)
    ax_pcb.set_ylim(-2, board_height + 2)
    ax_pcb.set_aspect('equal')
    ax_pcb.axis('off')

    # Board outline
    board_rect = Rectangle((0, 0), board_width, board_height,
                            linewidth=2, edgecolor=COLORS['copper'],
                            facecolor=COLORS['board'])
    ax_pcb.add_patch(board_rect)

    # Input grid region (show as lighter area)
    input_region = Rectangle((3, grid_start_y - 2), input_grid_size + 4, input_grid_size + 4,
                              facecolor=COLORS['input_grid'], alpha=0.3, zorder=1)
    ax_pcb.add_patch(input_region)

    # Title
    title_text = ax_pcb.text(board_width / 2, board_height + 1,
                             'MNIST Physical Neural Network',
                             ha='center', va='bottom', color=COLORS['silkscreen'],
                             fontsize=14, fontweight='bold')

    # Epoch and accuracy display
    epoch_text = ax_pcb.text(5, -1, '', ha='left', va='top',
                             color='white', fontsize=11)
    acc_text = ax_pcb.text(board_width - 5, -1, '', ha='right', va='top',
                           color=COLORS['output'], fontsize=11)

    # Draw input pixels (static, small dots)
    input_scatter = ax_pcb.scatter(input_positions[:, 0], input_positions[:, 1],
                                   s=2, c=COLORS['copper'], alpha=0.5, zorder=2)

    # Hidden neurons (will be updated)
    hidden_scatter = ax_pcb.scatter([], [], s=80, c=COLORS['hidden'],
                                    edgecolors='white', linewidth=1, zorder=5)

    # Output neurons (static)
    output_scatter = ax_pcb.scatter(output_positions[:, 0], output_positions[:, 1],
                                    s=100, c=COLORS['output'],
                                    edgecolors='white', linewidth=1.5, zorder=5)

    # Output labels
    for i, (x, y) in enumerate(output_positions):
        ax_pcb.text(x + 2, y, str(i), ha='left', va='center',
                    color=COLORS['silkscreen'], fontsize=9, fontweight='bold')

    # Labels
    ax_pcb.text(5 + input_grid_size/2, grid_start_y - 4, '28×28 Input',
                ha='center', va='top', color=COLORS['silkscreen'], fontsize=9)
    ax_pcb.text(board_width - 8, board_height / 2, 'Output\n0-9',
                ha='center', va='center', color=COLORS['silkscreen'], fontsize=9)

    # Initialize line plots
    line_loss, = ax_loss.plot([], [], color='#ff6b6b', linewidth=2, label='Total')
    line_class, = ax_loss.plot([], [], color='#4ecdc4', linewidth=1.5, label='Class', alpha=0.7)
    ax_loss.legend(loc='upper right', facecolor='#2a2a2a', labelcolor='white', fontsize=8)
    ax_loss.set_xlabel('Epoch', color='white', fontsize=9)

    line_train, = ax_acc.plot([], [], color='#4ecdc4', linewidth=2, label='Train')
    line_test, = ax_acc.plot([], [], color='#ffe66d', linewidth=2, label='Test')
    ax_acc.legend(loc='lower right', facecolor='#2a2a2a', labelcolor='white', fontsize=8)
    ax_acc.set_xlabel('Epoch', color='white', fontsize=9)
    ax_acc.set_ylabel('%', color='white', fontsize=9)

    # Trajectory setup
    n_hidden = history['hidden_positions'][0].shape[0]
    traj_colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_hidden))
    trajectory_lines = []
    for i in range(n_hidden):
        line, = ax_traj.plot([], [], color=traj_colors[i], linewidth=1, alpha=0.6)
        trajectory_lines.append(line)

    ax_traj.set_xlim(35, 95)
    ax_traj.set_ylim(0, board_height)
    ax_traj.set_xlabel('X (mm)', color='white', fontsize=9)
    ax_traj.set_ylabel('Y (mm)', color='white', fontsize=9)

    # Connection lines (sample - not all 784×64, that's too many)
    connection_lines = None

    def interpolate_frame(frame_idx):
        """Map frame to history index with easing."""
        progress = frame_idx / max(1, n_frames - 1)
        # Ease-in-out for smoother animation
        if progress < 0.5:
            progress = 2 * progress * progress
        else:
            progress = 1 - pow(-2 * progress + 2, 2) / 2
        history_idx = int(progress * (n_history - 1))
        return min(history_idx, n_history - 1)

    def update(frame):
        nonlocal connection_lines

        idx = interpolate_frame(frame)
        epoch = history['epochs'][idx]

        # Get data
        hidden_pos = history['hidden_positions'][idx]

        # Update hidden neurons
        hidden_scatter.set_offsets(hidden_pos)

        # Update text
        epoch_text.set_text(f'Epoch: {epoch}')
        if idx < len(history['test_acc']):
            acc_text.set_text(f'Test Acc: {history["test_acc"][idx]:.1f}%')

        # Update loss plot
        epochs = history['epochs'][:idx + 1]
        if len(history['train_loss']) > idx:
            losses = history['train_loss'][:idx + 1]
            class_losses = history['classification_loss'][:idx + 1]
            line_loss.set_data(epochs, losses)
            line_class.set_data(epochs, class_losses)
            if len(epochs) > 1:
                ax_loss.set_xlim(0, max(epochs) * 1.05)
                ax_loss.set_ylim(0, max(losses) * 1.1)

        # Update accuracy plot
        if len(history['train_acc']) > idx:
            train_accs = history['train_acc'][:idx + 1]
            test_accs = history['test_acc'][:idx + 1]
            line_train.set_data(epochs, train_accs)
            line_test.set_data(epochs, test_accs)
            if len(epochs) > 1:
                ax_acc.set_xlim(0, max(epochs) * 1.05)
                ax_acc.set_ylim(max(0, min(min(train_accs), min(test_accs)) - 5), 100)

        # Update trajectories
        for i in range(n_hidden):
            traj = np.array([history['hidden_positions'][j][i] for j in range(idx + 1)])
            trajectory_lines[i].set_data(traj[:, 0], traj[:, 1])

        # Draw sample connections (from hidden to output only - cleaner)
        if connection_lines is not None:
            connection_lines.remove()

        # Sample strongest connections
        if idx < len(history['weights_W2_sample']):
            W2 = history['weights_W2_sample'][idx]
            segments = []
            colors = []

            for j in range(len(hidden_pos)):
                for k in range(10):
                    weight = W2[j, k]
                    if abs(weight) > 0.3:  # Only show strong connections
                        x1, y1 = hidden_pos[j]
                        x2, y2 = output_positions[k]
                        segments.append([(x1, y1), (x2, y2)])
                        alpha = min(1.0, abs(weight) / 2)
                        if weight > 0:
                            colors.append((*mcolors.to_rgb(COLORS['trace_pos']), alpha * 0.5))
                        else:
                            colors.append((*mcolors.to_rgb(COLORS['trace_neg']), alpha * 0.5))

            if segments:
                connection_lines = LineCollection(segments, colors=colors, linewidths=1, zorder=3)
                ax_pcb.add_collection(connection_lines)
            else:
                connection_lines = None

        return [hidden_scatter, epoch_text, acc_text, line_loss, line_class,
                line_train, line_test] + trajectory_lines

    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    print(f"Saving to {save_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    print("Animation saved!")

    plt.close(fig)


def plot_mnist_layout(model, save_path: str = "mnist_layout.png"):
    """
    Create detailed plot of final MNIST PCB layout.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['board'])

    board_width = model.board_width
    board_height = model.board_height
    input_grid_size = model.input_grid_size

    # Board
    board_rect = Rectangle((0, 0), board_width, board_height,
                            linewidth=3, edgecolor=COLORS['copper'],
                            facecolor=COLORS['board'])
    ax.add_patch(board_rect)

    # Title
    ax.text(board_width / 2, board_height - 3,
            'MNIST Physical Neural Network',
            ha='center', va='top', color=COLORS['silkscreen'],
            fontsize=16, fontweight='bold', fontfamily='monospace')

    ax.text(board_width / 2, board_height - 8,
            f'{model.input_size} → {model.hidden_size} → {model.output_size}',
            ha='center', va='top', color=COLORS['copper'],
            fontsize=12, fontfamily='monospace')

    # Get positions
    input_pos = model.input_positions.cpu().numpy()
    hidden_pos = model.hidden_positions.detach().cpu().numpy()
    output_pos = model.output_positions.cpu().numpy()

    # Draw input grid
    grid_start_y = (board_height - input_grid_size) / 2
    input_region = Rectangle((3, grid_start_y - 2), input_grid_size + 4, input_grid_size + 4,
                              facecolor=COLORS['input_grid'], alpha=0.3)
    ax.add_patch(input_region)

    ax.scatter(input_pos[:, 0], input_pos[:, 1], s=3, c=COLORS['copper'], alpha=0.6)
    ax.text(5 + input_grid_size/2, grid_start_y - 5, '28×28 Input Pixels',
            ha='center', va='top', color=COLORS['silkscreen'], fontsize=10)

    # Draw hidden neurons
    ax.scatter(hidden_pos[:, 0], hidden_pos[:, 1], s=100, c=COLORS['hidden'],
               edgecolors='white', linewidth=2, zorder=10)

    # Draw output neurons with labels
    ax.scatter(output_pos[:, 0], output_pos[:, 1], s=150, c=COLORS['output'],
               edgecolors='white', linewidth=2, zorder=10)

    for i, (x, y) in enumerate(output_pos):
        ax.text(x, y, str(i), ha='center', va='center',
                color='white', fontsize=10, fontweight='bold', zorder=11)

    # Draw sample connections (hidden → output)
    W2 = model.W2.detach().cpu().numpy()
    for j in range(len(hidden_pos)):
        for k in range(10):
            weight = W2[j, k]
            if abs(weight) > 0.5:
                x1, y1 = hidden_pos[j]
                x2, y2 = output_pos[k]
                color = COLORS['trace_pos'] if weight > 0 else COLORS['trace_neg']
                alpha = min(0.8, abs(weight) / 3)
                lw = 0.5 + abs(weight) * 0.5
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, zorder=3)

    # Stats
    stats = model.get_layout_stats()
    stats_text = f"Min spacing: {stats['min_spacing']:.1f}mm\nTrace length: {stats['total_trace_length_mm']/1000:.1f}m"
    ax.text(5, 5, stats_text, ha='left', va='bottom',
            color=COLORS['silkscreen'], fontsize=9, fontfamily='monospace')

    ax.set_xlim(-5, board_width + 5)
    ax.set_ylim(-5, board_height + 5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    print(f"Layout saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    from train_mnist import train_mnist, save_model

    print("Training MNIST network...")
    model, history = train_mnist(epochs=20, hidden_size=64, verbose=True)

    print("\nGenerating visualizations...")
    create_mnist_animation(history, save_path="mnist_condensation.gif", duration=10)
    plot_mnist_layout(model, save_path="mnist_layout.png")

    save_model(model, "mnist_physical_network.pt")
    print("\nDone!")
