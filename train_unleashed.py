"""
Train MNIST Unleashed - 256 neurons, 200mm board, 4 layers.

Let intelligence find its natural geometry!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
from model_unleashed import MNISTUnleashed


def train_unleashed(epochs: int = 50,
                    hidden_size: int = 256,
                    board_size: tuple = (200, 200),
                    n_layers: int = 4,
                    layout_weight: float = 0.005,  # Lower = more freedom
                    lr_weights: float = 0.001,
                    lr_positions: float = 0.5,     # Faster position learning
                    lr_layers: float = 0.1,        # Layer assignment learning
                    batch_size: int = 256,         # Bigger batches for 3090
                    min_spacing: float = 0.15,
                    save_every: int = 1,
                    verbose: bool = True) -> tuple:
    """
    Train MNIST with unleashed physical constraints.

    Returns:
        (model, history)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("=" * 70)
        print("  MNIST UNLEASHED - Intelligence Finding Its Natural Geometry")
        print("=" * 70)
        print(f"  Device: {device}")
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Hidden neurons: {hidden_size}")
        print(f"  Board: {board_size[0]}mm × {board_size[1]}mm ({n_layers} layers)")
        print(f"  Min spacing: {min_spacing}mm (vs 3mm before = 20× tighter!)")
        print(f"  Layout weight: {layout_weight} (relaxed for freedom)")
        print("=" * 70)
        print()

    # Data
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
        print(f"Training: {len(train_dataset)} samples")
        print(f"Test: {len(test_dataset)} samples")
        print()

    # Create model
    model = MNISTUnleashed(
        hidden_size=hidden_size,
        board_size=board_size,
        n_layers=n_layers,
        min_spacing=min_spacing
    ).to(device)

    # Optimizer with separate LRs for weights, positions, and layers
    optimizer = torch.optim.AdamW([
        {'params': [model.W1, model.W2, model.b1, model.b2], 'lr': lr_weights, 'weight_decay': 1e-4},
        {'params': [model.hidden_positions], 'lr': lr_positions, 'weight_decay': 0},
        {'params': [model.hidden_layers], 'lr': lr_layers, 'weight_decay': 0}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # History
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'layout_loss': [],
        'classification_loss': [],
        'hidden_positions': [],
        'hidden_layers': [],
        'epochs': [],
        'weights_W1_sample': [],
        'weights_W2_sample': [],
        'spacing_min': [],
        'trace_length': [],
    }

    # Initial state
    history['hidden_positions'].append(model.hidden_positions.detach().cpu().numpy().copy())
    history['hidden_layers'].append(model.hidden_layers.detach().cpu().numpy().copy())
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

            output = model(data)
            class_loss = F.cross_entropy(output, target)
            layout_loss = model.layout_loss()
            loss = class_loss + layout_weight * layout_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_class_loss += class_loss.item()
            epoch_layout_loss += layout_loss.item()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        scheduler.step()

        # Epoch stats
        n_batches = len(train_loader)
        train_loss = epoch_loss / n_batches
        train_class_loss = epoch_class_loss / n_batches
        train_layout_loss = epoch_layout_loss / n_batches
        train_acc = 100.0 * correct / total

        # Test
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
        best_test_acc = max(best_test_acc, test_acc)

        elapsed = time.time() - start_time

        # Save history
        if epoch % save_every == 0 or epoch == epochs - 1:
            stats = model.get_layout_stats()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['classification_loss'].append(train_class_loss)
            history['layout_loss'].append(train_layout_loss)
            history['hidden_positions'].append(model.hidden_positions.detach().cpu().numpy().copy())
            history['hidden_layers'].append(model.hidden_layers.detach().cpu().numpy().copy())
            history['epochs'].append(epoch + 1)
            history['weights_W1_sample'].append(model.W1.detach().cpu().numpy()[:100, :].copy())
            history['weights_W2_sample'].append(model.W2.detach().cpu().numpy().copy())
            history['spacing_min'].append(stats['min_spacing'])
            history['trace_length'].append(stats['total_trace_length_m'])

        if verbose:
            stats = model.get_layout_stats()
            print(f"Epoch {epoch+1:3d}/{epochs} [{elapsed:.1f}s]")
            print(f"  Loss: {train_loss:.4f} (class: {train_class_loss:.4f}, layout: {train_layout_loss:.4f})")
            print(f"  Accuracy: train {train_acc:.1f}%, test {test_acc:.1f}%")
            print(f"  Layout: spacing {stats['min_spacing']:.2f}mm, traces {stats['total_trace_length_m']:.1f}m")
            print(f"  Layers: {stats['layer_distribution']} (mean: {stats['mean_layer']:.2f})")
            print()

    if verbose:
        print("=" * 70)
        print("  Training Complete!")
        print("=" * 70)
        print(f"  Best test accuracy: {best_test_acc:.2f}%")
        print()
        print(model.summary())

    return model, history


def save_model(model: MNISTUnleashed, filepath: str):
    """Save model."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': model.hidden_size,
        'board_size': (model.board_width, model.board_height),
        'n_layers': model.n_layers,
        'min_spacing': model.min_spacing,
    }, filepath)
    print(f"Model saved to {filepath}")


if __name__ == "__main__":
    model, history = train_unleashed(
        epochs=50,
        hidden_size=256,
        board_size=(200, 200),
        n_layers=4,
        layout_weight=0.005,
        min_spacing=0.15
    )

    save_model(model, 'mnist_unleashed.pt')
