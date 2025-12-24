"""
Train MNIST with Physical Neural Network.

Watch 64 hidden neurons condense into optimal positions
while learning to classify handwritten digits!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
from model_mnist import MNISTPhysicalNetwork


def train_mnist(epochs: int = 50,
                hidden_size: int = 64,
                layout_weight: float = 0.01,
                lr_weights: float = 0.001,
                lr_positions: float = 0.1,
                batch_size: int = 128,
                board_size: tuple = (100, 100),
                save_every: int = 1,
                verbose: bool = True) -> tuple:
    """
    Train MNIST classifier with physical layout optimization.

    Args:
        epochs: Number of training epochs
        hidden_size: Number of hidden neurons
        layout_weight: Weight for layout loss (α)
        lr_weights: Learning rate for network weights
        lr_positions: Learning rate for neuron positions
        batch_size: Training batch size
        board_size: PCB dimensions in mm
        save_every: Save positions every N epochs (for animation)
        verbose: Print progress

    Returns:
        (model, history)
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 60)
        print("  MNIST Physical Neural Network Training")
        print("  Intelligence crystallizing into copper geometry...")
        print("=" * 60)
        print(f"  Device: {device}")
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Hidden neurons: {hidden_size}")
        print(f"  Board size: {board_size[0]}mm × {board_size[1]}mm")
        print(f"  Layout weight: {layout_weight}")
        print("=" * 60)
        print()

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False,
                             num_workers=4, pin_memory=True)

    if verbose:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print()

    # Create model
    model = MNISTPhysicalNetwork(
        hidden_size=hidden_size,
        board_size=board_size
    ).to(device)

    # Optimizer with different LRs
    optimizer = torch.optim.AdamW([
        {'params': [model.W1, model.W2, model.b1, model.b2], 'lr': lr_weights, 'weight_decay': 1e-4},
        {'params': [model.hidden_positions], 'lr': lr_positions, 'weight_decay': 0}
    ])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'layout_loss': [],
        'classification_loss': [],
        'hidden_positions': [],
        'epochs': [],
        'weights_W1_sample': [],  # Sample of weights for viz
        'weights_W2_sample': [],
    }

    # Initial positions
    history['hidden_positions'].append(model.hidden_positions.detach().cpu().numpy().copy())
    history['epochs'].append(0)

    best_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_layout_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Classification loss
            class_loss = F.cross_entropy(output, target)

            # Layout loss
            layout_loss = model.layout_loss()

            # Combined loss
            loss = class_loss + layout_weight * layout_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_class_loss += class_loss.item()
            epoch_layout_loss += layout_loss.item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        scheduler.step()

        # Epoch statistics
        n_batches = len(train_loader)
        train_loss = epoch_loss / n_batches
        train_class_loss = epoch_class_loss / n_batches
        train_layout_loss = epoch_layout_loss / n_batches
        train_acc = 100.0 * correct / total

        # Test accuracy
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

        test_acc = 100.0 * test_correct / test_total

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        elapsed = time.time() - start_time

        # Save history
        if epoch % save_every == 0 or epoch == epochs - 1:
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['classification_loss'].append(train_class_loss)
            history['layout_loss'].append(train_layout_loss)
            history['hidden_positions'].append(model.hidden_positions.detach().cpu().numpy().copy())
            history['epochs'].append(epoch + 1)

            # Sample weights for visualization
            history['weights_W1_sample'].append(model.W1.detach().cpu().numpy()[:100, :].copy())
            history['weights_W2_sample'].append(model.W2.detach().cpu().numpy().copy())

        if verbose:
            stats = model.get_layout_stats()
            print(f"Epoch {epoch+1:3d}/{epochs} [{elapsed:.1f}s]")
            print(f"  Loss: {train_loss:.4f} (class: {train_class_loss:.4f}, layout: {train_layout_loss:.4f})")
            print(f"  Accuracy: train {train_acc:.1f}%, test {test_acc:.1f}%")
            print(f"  Layout: spacing {stats['min_spacing']:.1f}mm, traces {stats['total_trace_length_mm']/1000:.1f}m")
            print()

    if verbose:
        print("=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print(f"  Best test accuracy: {best_test_acc:.2f}%")
        print()
        print(model.summary())

    return model, history


def save_model(model: MNISTPhysicalNetwork, filepath: str):
    """Save trained model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': model.hidden_size,
        'board_size': (model.board_width, model.board_height),
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> MNISTPhysicalNetwork:
    """Load trained model."""
    checkpoint = torch.load(filepath)
    model = MNISTPhysicalNetwork(
        hidden_size=checkpoint['hidden_size'],
        board_size=checkpoint['board_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train MNIST Physical Neural Network')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden neurons')
    parser.add_argument('--layout-weight', type=float, default=0.01, help='Layout loss weight')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for weights')
    parser.add_argument('--lr-pos', type=float, default=0.1, help='Learning rate for positions')

    args = parser.parse_args()

    model, history = train_mnist(
        epochs=args.epochs,
        hidden_size=args.hidden,
        layout_weight=args.layout_weight,
        batch_size=args.batch_size,
        lr_weights=args.lr,
        lr_positions=args.lr_pos
    )

    save_model(model, 'mnist_physical_network.pt')

    # Quick test
    print("\nQuick inference test:")
    device = next(model.parameters()).device
    test_input = torch.randn(1, 784).to(device)
    output = model(test_input)
    print(f"Sample prediction: digit {output.argmax().item()}")
