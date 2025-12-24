"""
Visualization of Physical Neural Network training.

Creates animations showing the "condensation" process as neurons
move from random positions to their optimal physical layout.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


# Style configuration for PCB aesthetics
PCB_COLORS = {
    'board': '#1a472a',       # Dark green (FR4)
    'copper': '#b87333',       # Copper orange
    'copper_light': '#d4a574', # Light copper
    'solder_mask': '#2d5a3d',  # Green solder mask
    'silkscreen': '#ffffff',   # White silkscreen
    'input': '#4a90d9',        # Blue for inputs
    'hidden': '#d94a4a',       # Red for hidden
    'output': '#4ad94a',       # Green for outputs
    'trace_positive': '#ffcc00',  # Gold for positive weights
    'trace_negative': '#cc6600',  # Dark copper for negative
}


def create_condensation_animation(history: dict,
                                   board_size: tuple = (50, 50),
                                   input_positions: np.ndarray = None,
                                   output_positions: np.ndarray = None,
                                   save_path: str = "condensation.gif",
                                   fps: int = 30,
                                   duration: float = 10.0) -> FuncAnimation:
    """
    Create an animation of the network condensing during training.

    Args:
        history: Training history from train_xor()
        board_size: Board dimensions (width, height) in mm
        input_positions: Fixed input neuron positions
        output_positions: Fixed output neuron positions
        save_path: Path to save the animation
        fps: Frames per second
        duration: Total animation duration in seconds

    Returns:
        FuncAnimation object
    """
    # Default positions if not provided
    if input_positions is None:
        input_positions = np.array([[2.0, 16.67], [2.0, 33.33]])
    if output_positions is None:
        output_positions = np.array([[48.0, 25.0]])

    board_width, board_height = board_size
    n_frames = int(fps * duration)
    n_history = len(history['epochs'])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0a')

    # Configure left plot (loss curves)
    ax1.set_facecolor('#1a1a1a')
    ax1.set_title('Training Progress', color='white', fontsize=14, pad=10)
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#333333')

    # Configure right plot (PCB layout)
    ax2.set_facecolor(PCB_COLORS['board'])
    ax2.set_title('PCB Layout Condensation', color='white', fontsize=14, pad=10)
    ax2.set_xlim(-2, board_width + 2)
    ax2.set_ylim(-2, board_height + 2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (mm)', color='white')
    ax2.set_ylabel('Y (mm)', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#333333')

    # Draw board outline
    board_rect = Rectangle((0, 0), board_width, board_height,
                            linewidth=2, edgecolor=PCB_COLORS['copper'],
                            facecolor=PCB_COLORS['board'], zorder=0)
    ax2.add_patch(board_rect)

    # Add grid pattern (like PCB silkscreen)
    for x in range(0, int(board_width) + 1, 10):
        ax2.axvline(x, color='#2a5a3a', linewidth=0.5, alpha=0.3)
    for y in range(0, int(board_height) + 1, 10):
        ax2.axhline(y, color='#2a5a3a', linewidth=0.5, alpha=0.3)

    # Initialize line objects for loss curves
    line_class, = ax1.plot([], [], color='#ff6b6b', linewidth=2,
                            label='Classification Loss')
    line_layout, = ax1.plot([], [], color='#4ecdc4', linewidth=2,
                             label='Layout Loss')
    line_total, = ax1.plot([], [], color='#ffe66d', linewidth=2,
                            label='Total Loss', linestyle='--')
    ax1.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='#444444',
               labelcolor='white')

    # Initialize neuron scatter plots
    input_scatter = ax2.scatter([], [], s=200, c=PCB_COLORS['input'],
                                 edgecolors='white', linewidth=2, zorder=5,
                                 label='Input')
    hidden_scatter = ax2.scatter([], [], s=250, c=PCB_COLORS['hidden'],
                                  edgecolors='white', linewidth=2, zorder=5,
                                  label='Hidden')
    output_scatter = ax2.scatter([], [], s=200, c=PCB_COLORS['output'],
                                  edgecolors='white', linewidth=2, zorder=5,
                                  label='Output')

    ax2.legend(loc='upper left', facecolor='#2a2a2a', edgecolor='#444444',
               labelcolor='white')

    # Text annotations
    epoch_text = ax2.text(board_width / 2, board_height + 1, '',
                          ha='center', va='bottom', color='white',
                          fontsize=12, fontweight='bold')
    accuracy_text = ax2.text(board_width - 2, board_height + 1, '',
                             ha='right', va='bottom', color='#4ad94a',
                             fontsize=11)

    # Line collection for traces (will be updated each frame)
    trace_collection = None

    def interpolate_frame(frame_idx):
        """Map animation frame to training history index."""
        progress = frame_idx / max(1, n_frames - 1)
        history_idx = int(progress * (n_history - 1))
        return min(history_idx, n_history - 1)

    def get_trace_segments(hidden_pos, weights_W1, weights_W2):
        """Generate line segments for all traces."""
        segments = []
        colors = []
        widths = []

        # Input → Hidden traces
        for i in range(len(input_positions)):
            for j in range(len(hidden_pos)):
                x1, y1 = input_positions[i]
                x2, y2 = hidden_pos[j]
                weight = weights_W1[i, j]

                # Manhattan routing: horizontal then vertical
                segments.append([(x1, y1), (x2, y1), (x2, y2)])

                # Color based on weight sign
                if weight > 0:
                    colors.append(PCB_COLORS['trace_positive'])
                else:
                    colors.append(PCB_COLORS['trace_negative'])

                # Width based on weight magnitude
                width = 0.5 + abs(weight) * 0.8
                widths.append(width)

        # Hidden → Output traces
        for j in range(len(hidden_pos)):
            for k in range(len(output_positions)):
                x1, y1 = hidden_pos[j]
                x2, y2 = output_positions[k]
                weight = weights_W2[j, k]

                segments.append([(x1, y1), (x1, y2), (x2, y2)])

                if weight > 0:
                    colors.append(PCB_COLORS['trace_positive'])
                else:
                    colors.append(PCB_COLORS['trace_negative'])

                width = 0.5 + abs(weight) * 0.8
                widths.append(width)

        return segments, colors, widths

    def update(frame):
        nonlocal trace_collection

        # Get history index for this frame
        hist_idx = interpolate_frame(frame)
        epoch = history['epochs'][hist_idx]

        # Update loss curves
        epochs = history['epochs'][:hist_idx + 1]
        class_loss = history['classification_loss'][:hist_idx + 1]
        layout_loss = history['layout_loss'][:hist_idx + 1]
        total_loss = history['total_loss'][:hist_idx + 1]

        line_class.set_data(epochs, class_loss)
        line_layout.set_data(epochs, layout_loss)
        line_total.set_data(epochs, total_loss)

        # Adjust axes
        if len(epochs) > 1:
            ax1.set_xlim(0, max(epochs) * 1.05)
            all_losses = class_loss + layout_loss + total_loss
            ax1.set_ylim(0, max(all_losses) * 1.1)

        # Get current positions and weights
        hidden_pos = history['hidden_positions'][hist_idx]
        weights_W1 = history['weights_W1'][hist_idx]
        weights_W2 = history['weights_W2'][hist_idx]
        accuracy = history['accuracy'][hist_idx]

        # Remove old traces
        if trace_collection is not None:
            trace_collection.remove()

        # Draw new traces
        segments, colors, widths = get_trace_segments(hidden_pos, weights_W1, weights_W2)

        # Create line segments for each trace
        all_segments = []
        all_colors = []

        for seg, color, width in zip(segments, colors, widths):
            # Convert 3-point Manhattan path to line segments
            for i in range(len(seg) - 1):
                all_segments.append([seg[i], seg[i + 1]])
                all_colors.append(color)

        if all_segments:
            trace_collection = LineCollection(all_segments, colors=all_colors,
                                               linewidths=[w for w, s in zip(widths, segments) for _ in range(2)],
                                               alpha=0.7, zorder=2)
            ax2.add_collection(trace_collection)

        # Update neuron positions
        input_scatter.set_offsets(input_positions)
        hidden_scatter.set_offsets(hidden_pos)
        output_scatter.set_offsets(output_positions)

        # Update text
        epoch_text.set_text(f'Epoch: {epoch}')
        accuracy_text.set_text(f'Accuracy: {accuracy:.0%}')

        return [line_class, line_layout, line_total, input_scatter,
                hidden_scatter, output_scatter, epoch_text, accuracy_text]

    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    # Save animation
    print(f"Saving animation to {save_path}...")
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    print(f"Animation saved!")

    plt.close(fig)
    return anim


def plot_final_layout(model, save_path: str = "final_layout.png"):
    """
    Create a detailed plot of the final PCB layout.

    Args:
        model: Trained PhysicalNeuralNetwork
        save_path: Path to save the image
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor(PCB_COLORS['board'])

    board_width = model.board_width
    board_height = model.board_height

    # Draw board
    board_rect = Rectangle((0, 0), board_width, board_height,
                            linewidth=3, edgecolor=PCB_COLORS['copper'],
                            facecolor=PCB_COLORS['board'], zorder=0)
    ax.add_patch(board_rect)

    # Add title as silkscreen text
    ax.text(board_width / 2, board_height - 2,
            'XOR Physical Neural Network',
            ha='center', va='top', color=PCB_COLORS['silkscreen'],
            fontsize=14, fontweight='bold', fontfamily='monospace')

    # Draw grid
    for x in range(0, int(board_width) + 1, 5):
        ax.axvline(x, color='#2a5a3a', linewidth=0.5, alpha=0.3)
    for y in range(0, int(board_height) + 1, 5):
        ax.axhline(y, color='#2a5a3a', linewidth=0.5, alpha=0.3)

    # Get trace info
    traces = model.get_trace_info()

    # Draw traces
    for trace in traces['input_to_hidden'] + traces['hidden_to_output']:
        x1, y1 = trace['from']
        x2, y2 = trace['to']
        weight = trace['weight']
        width_mm = trace['width_mm']

        # Color based on weight
        color = PCB_COLORS['trace_positive'] if weight > 0 else PCB_COLORS['trace_negative']

        # Line width scaled for visualization
        lw = max(1, width_mm * 2)

        # Manhattan routing
        ax.plot([x1, x2], [y1, y1], color=color, linewidth=lw, solid_capstyle='round', alpha=0.8)
        ax.plot([x2, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle='round', alpha=0.8)

    # Draw neurons as pads
    input_pos = model.input_positions.numpy()
    hidden_pos = model.hidden_positions.detach().numpy()
    output_pos = model.output_positions.numpy()

    # Input pads
    for i, (x, y) in enumerate(input_pos):
        circle = Circle((x, y), 2, facecolor=PCB_COLORS['copper'],
                         edgecolor='white', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y - 4, f'IN{i}', ha='center', va='top',
                color=PCB_COLORS['silkscreen'], fontsize=9, fontweight='bold')

    # Hidden neurons
    for j, (x, y) in enumerate(hidden_pos):
        circle = Circle((x, y), 2.5, facecolor=PCB_COLORS['hidden'],
                         edgecolor='white', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, f'H{j}', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')

    # Output pad
    for k, (x, y) in enumerate(output_pos):
        circle = Circle((x, y), 2, facecolor=PCB_COLORS['output'],
                         edgecolor='white', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y + 4, f'OUT{k}', ha='center', va='bottom',
                color=PCB_COLORS['silkscreen'], fontsize=9, fontweight='bold')

    # Add dimensions
    ax.text(board_width / 2, -3, f'{board_width}mm',
            ha='center', va='top', color='white', fontsize=10)
    ax.text(-3, board_height / 2, f'{board_height}mm',
            ha='right', va='center', color='white', fontsize=10, rotation=90)

    # Configure axes
    ax.set_xlim(-5, board_width + 5)
    ax.set_ylim(-5, board_height + 5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    print(f"Final layout saved to {save_path}")
    plt.close(fig)


def plot_training_progress(history: dict, save_path: str = "training_progress.png"):
    """
    Create a static plot of training progress.

    Args:
        history: Training history from train_xor()
        save_path: Path to save the image
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a1a')

    for ax in axes.flat:
        ax.set_facecolor('#2a2a2a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    epochs = history['epochs']

    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['classification_loss'], color='#ff6b6b',
             linewidth=2, label='Classification')
    ax1.plot(epochs, history['layout_loss'], color='#4ecdc4',
             linewidth=2, label='Layout')
    ax1.plot(epochs, history['total_loss'], color='#ffe66d',
             linewidth=2, linestyle='--', label='Total')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.set_title('Loss Curves', color='white', fontsize=12)
    ax1.legend(facecolor='#3a3a3a', edgecolor='#555555', labelcolor='white')
    ax1.set_yscale('log')

    # Plot 2: Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, [a * 100 for a in history['accuracy']],
             color='#4ad94a', linewidth=2)
    ax2.axhline(100, color='#666666', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Accuracy (%)', color='white')
    ax2.set_title('Training Accuracy', color='white', fontsize=12)
    ax2.set_ylim(0, 105)

    # Plot 3: Neuron position trajectory
    ax3 = axes[1, 0]
    positions = np.array(history['hidden_positions'])
    n_hidden = positions.shape[1]

    colors = plt.cm.viridis(np.linspace(0, 1, n_hidden))
    for j in range(n_hidden):
        trajectory = positions[:, j, :]
        ax3.plot(trajectory[:, 0], trajectory[:, 1], color=colors[j],
                 alpha=0.7, linewidth=1.5)
        ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], color=colors[j],
                    s=100, edgecolor='white', linewidth=2, zorder=5)
        ax3.scatter(trajectory[0, 0], trajectory[0, 1], color=colors[j],
                    s=50, marker='x', linewidth=2, zorder=5)

    ax3.set_xlabel('X (mm)', color='white')
    ax3.set_ylabel('Y (mm)', color='white')
    ax3.set_title('Hidden Neuron Trajectories', color='white', fontsize=12)
    ax3.set_xlim(0, 50)
    ax3.set_ylim(0, 50)

    # Plot 4: Weight evolution
    ax4 = axes[1, 1]
    weights_W1 = np.array(history['weights_W1'])
    weights_W2 = np.array(history['weights_W2'])

    # Flatten and plot
    for i in range(weights_W1.shape[1]):
        for j in range(weights_W1.shape[2]):
            ax4.plot(epochs, weights_W1[:, i, j], alpha=0.5, linewidth=1)

    ax4.axhline(0, color='#666666', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch', color='white')
    ax4.set_ylabel('Weight Value', color='white')
    ax4.set_title('Weight Evolution (Layer 1)', color='white', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                edgecolor='none', bbox_inches='tight')
    print(f"Training progress saved to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    from train import train_xor

    print("Training network...")
    model, history = train_xor(epochs=3000, layout_weight=0.1, verbose=True)

    print("\nGenerating visualizations...")

    # Create condensation animation
    input_pos = model.input_positions.numpy()
    output_pos = model.output_positions.numpy()

    create_condensation_animation(
        history,
        board_size=(model.board_width, model.board_height),
        input_positions=input_pos,
        output_positions=output_pos,
        save_path="condensation.gif",
        fps=30,
        duration=8.0
    )

    # Create static plots
    plot_final_layout(model, save_path="final_layout.png")
    plot_training_progress(history, save_path="training_progress.png")

    print("\nAll visualizations complete!")
