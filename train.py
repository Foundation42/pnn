"""
Train XOR network with physical layout optimization.

The network learns to solve XOR while simultaneously optimizing
the physical placement of neurons for PCB manufacturing.
"""

import torch
import torch.nn as nn
import numpy as np
from model import PhysicalNeuralNetwork


def train_xor(epochs: int = 5000,
              layout_weight: float = 0.1,
              lr_weights: float = 0.01,
              lr_positions: float = 0.5,
              hidden_size: int = 4,
              board_size: tuple = (50, 50),
              verbose: bool = True) -> tuple:
    """
    Train XOR network with physical layout optimization.

    Args:
        epochs: Number of training epochs
        layout_weight: Weight for layout loss (α in total_loss = class_loss + α * layout_loss)
        lr_weights: Learning rate for network weights
        lr_positions: Learning rate for neuron positions
        hidden_size: Number of hidden neurons
        board_size: PCB board dimensions (width, height) in mm
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, training_history)
    """
    # XOR dataset
    X = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ], dtype=torch.float32)

    y = torch.tensor([
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ], dtype=torch.float32)

    # Create network
    model = PhysicalNeuralNetwork(
        input_size=2,
        hidden_size=hidden_size,
        output_size=1,
        board_size=board_size
    )

    # Optimizer with different learning rates for weights vs positions
    optimizer = torch.optim.Adam([
        {'params': [model.W1, model.W2, model.b1, model.b2], 'lr': lr_weights},
        {'params': [model.hidden_positions], 'lr': lr_positions}
    ])

    # Learning rate scheduler - reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500
    )

    # Training history (for visualization)
    history = {
        'classification_loss': [],
        'layout_loss': [],
        'total_loss': [],
        'accuracy': [],
        'hidden_positions': [],
        'weights_W1': [],
        'weights_W2': [],
        'epochs': []
    }

    if verbose:
        print("=" * 50)
        print("Training XOR with Physical Layout Optimization")
        print("=" * 50)
        print(f"Board size: {board_size[0]}mm × {board_size[1]}mm")
        print(f"Hidden neurons: {hidden_size}")
        print(f"Layout weight (α): {layout_weight}")
        print(f"LR (weights): {lr_weights}, LR (positions): {lr_positions}")
        print("=" * 50)
        print()

    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X)

        # Classification loss (Binary Cross Entropy)
        classification_loss = nn.BCELoss()(y_pred, y)

        # Layout loss (physical constraints)
        layout_loss = model.layout_loss()

        # Combined loss
        total_loss = classification_loss + layout_weight * layout_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Update scheduler
        scheduler.step(total_loss)

        # Track metrics
        with torch.no_grad():
            accuracy = ((y_pred > 0.5).float() == y).float().mean().item()

            # Record history (sample every 10 epochs to save memory)
            if epoch % 10 == 0 or epoch == epochs - 1:
                history['classification_loss'].append(classification_loss.item())
                history['layout_loss'].append(layout_loss.item())
                history['total_loss'].append(total_loss.item())
                history['accuracy'].append(accuracy)
                history['hidden_positions'].append(
                    model.hidden_positions.detach().clone().numpy()
                )
                history['weights_W1'].append(model.W1.detach().clone().numpy())
                history['weights_W2'].append(model.W2.detach().clone().numpy())
                history['epochs'].append(epoch)

            # Track best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch

        # Print progress
        if verbose and (epoch % 500 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:5d}/{epochs}:")
            print(f"  Classification Loss: {classification_loss.item():.4f}")
            print(f"  Layout Loss:         {layout_loss.item():.4f}")
            print(f"  Total Loss:          {total_loss.item():.4f}")
            print(f"  Accuracy:            {accuracy:.2%}")
            print()

    if verbose:
        print("=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Best accuracy: {best_accuracy:.2%} at epoch {best_epoch}")
        print()
        print(model.summary())
        print()

        # Final test
        print("XOR Truth Table Verification:")
        y_final = model(X)
        for inp, target, pred in zip(X, y, y_final):
            correct = "✓" if (pred.item() > 0.5) == (target.item() > 0.5) else "✗"
            print(f"  {inp.numpy()} → {pred.item():.3f} "
                  f"(target: {target.item():.0f}) {correct}")

    return model, history


def save_model(model: PhysicalNeuralNetwork, filepath: str):
    """Save trained model to file."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'output_size': model.output_size,
        'board_size': (model.board_width, model.board_height)
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> PhysicalNeuralNetwork:
    """Load trained model from file."""
    checkpoint = torch.load(filepath)
    model = PhysicalNeuralNetwork(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        output_size=checkpoint['output_size'],
        board_size=checkpoint['board_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    # Train with default parameters
    model, history = train_xor(
        epochs=5000,
        layout_weight=0.1,
        lr_weights=0.01,
        lr_positions=0.3,
        hidden_size=4,
        board_size=(50, 50)
    )

    # Save model
    save_model(model, "xor_physical_network.pt")

    # Print final trace info
    print("\n=== Final Trace Specifications ===")
    traces = model.get_trace_info()

    print("\nInput → Hidden traces:")
    for t in traces['input_to_hidden']:
        print(f"  I{t['from_idx']} → H{t['to_idx']}: "
              f"w={t['weight']:+.2f}, "
              f"width={t['width_mm']:.2f}mm, "
              f"length={t['length_mm']:.1f}mm")

    print("\nHidden → Output traces:")
    for t in traces['hidden_to_output']:
        print(f"  H{t['from_idx']} → O{t['to_idx']}: "
              f"w={t['weight']:+.2f}, "
              f"width={t['width_mm']:.2f}mm, "
              f"length={t['length_mm']:.1f}mm")
