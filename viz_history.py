"""
Visualize training history from JSON
Creates consistent charts and animations from saved history
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def load_history(path):
    """Load history from JSONL or JSON file"""
    if path.endswith('.jsonl'):
        # JSONL format - one JSON object per line
        history = {'loss': [], 'neurons': [], 'frozen': [], 'samples': [], 'phases': []}
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    history['loss'].append(data['loss'])
                    history['neurons'].append(data['neurons'])
                    history['frozen'].append(data['frozen'])
                    history['samples'].append(data.get('sample', ''))
                    history['phases'].append(data.get('phase', ''))

        # Try to load config from companion file
        config_path = path.replace('history.jsonl', 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                history['config'] = config_data.get('config', {})
        return history
    else:
        # Legacy JSON format
        with open(path, 'r') as f:
            return json.load(f)


def plot_training_curves(history, output_path):
    """Create 4-panel training analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['loss']) + 1)

    # Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # Neuron growth
    ax = axes[0, 1]
    neurons = history['neurons']
    frozen = history['frozen']
    active = [n - f for n, f in zip(neurons, frozen)]

    ax.plot(epochs, neurons, 'b-', label='Total', linewidth=2)
    ax.plot(epochs, active, 'g-', label='Active', linewidth=2)
    ax.plot(epochs, frozen, 'gray', label='Frozen', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Count')
    ax.set_title('Neuron Growth & Freezing')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frozen percentage
    ax = axes[1, 0]
    frozen_pct = [100 * f / n if n > 0 else 0 for f, n in zip(frozen, neurons)]
    ax.plot(epochs, frozen_pct, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frozen %')
    ax.set_title('Crystallization Progress')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Add phase markers
    total_epochs = len(epochs)
    for pct, label in [(0.2, 'GROW'), (0.5, 'grow+freeze'), (0.8, 'FREEZE')]:
        ax.axvline(x=total_epochs * pct, color='red', linestyle='--', alpha=0.5)
        ax.text(total_epochs * pct, 95, label, fontsize=8, ha='center')

    # Speedup over time
    ax = axes[1, 1]
    speedups = [n / max(n - f, 1) for n, f in zip(neurons, frozen)]
    ax.plot(epochs, speedups, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Speedup')
    ax.set_title('Theoretical Speedup')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_sample_evolution(history, output_path, num_samples=10):
    """Show how generated samples evolved during training"""
    samples = history.get('samples', [])
    if not samples:
        print("No samples in history")
        return

    total = len(samples)
    indices = np.linspace(0, total - 1, num_samples, dtype=int)

    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2 * num_samples))

    for i, idx in enumerate(indices):
        ax = axes[i]
        epoch = idx + 1
        loss = history['loss'][idx]
        neurons = history['neurons'][idx]
        frozen = history['frozen'][idx]

        sample = samples[idx][:150]  # Truncate for display

        ax.text(0.02, 0.5, f"Epoch {epoch} | Loss: {loss:.3f} | N: {neurons} | F: {frozen}\n{sample}",
                fontsize=9, family='monospace', verticalalignment='center',
                transform=ax.transAxes, wrap=True)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved sample evolution to {output_path}")


def print_summary(history):
    """Print summary statistics"""
    config = history.get('config', {})

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if config:
        print(f"\nConfiguration:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    print(f"\nTraining Results:")
    print(f"  Total epochs: {len(history['loss'])}")
    print(f"  Initial loss: {history['loss'][0]:.4f}")
    print(f"  Final loss: {history['loss'][-1]:.4f}")
    print(f"  Best loss: {min(history['loss']):.4f}")

    print(f"\nNeuron Growth:")
    print(f"  Initial neurons: {history['neurons'][0]}")
    print(f"  Final neurons: {history['neurons'][-1]}")
    print(f"  Growth: {history['neurons'][-1] / history['neurons'][0]:.1f}x")

    print(f"\nCrystallization:")
    final_frozen = history['frozen'][-1]
    final_neurons = history['neurons'][-1]
    print(f"  Frozen: {final_frozen} ({100*final_frozen/final_neurons:.1f}%)")
    print(f"  Active: {final_neurons - final_frozen}")
    print(f"  Speedup: {final_neurons / max(final_neurons - final_frozen, 1):.1f}x")

    # Find when freezing started
    freeze_start = next((i for i, f in enumerate(history['frozen']) if f > 0), None)
    if freeze_start:
        print(f"  Freezing started: epoch {freeze_start + 1}")

    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python viz_history.py <history.json> [output_dir]")
        print("\nExample: python viz_history.py runs/crystal_shakespeare_xxx/history.json")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(json_path)

    history = load_history(json_path)

    print_summary(history)

    # Generate visualizations
    plot_training_curves(history, os.path.join(output_dir, 'training_curves.png'))
    plot_sample_evolution(history, os.path.join(output_dir, 'sample_evolution.png'))


if __name__ == "__main__":
    main()
