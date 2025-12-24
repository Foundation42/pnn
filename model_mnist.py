"""
MNIST-scale Physical Neural Network.

Optimized for GPU with vectorized operations.
Input neurons arranged as 28x28 grid (like the actual image!).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MNISTPhysicalNetwork(nn.Module):
    """
    Physical Neural Network scaled for MNIST.

    Architecture: 784 (28x28 grid) → hidden → 10
    All positions in millimeters on a PCB.
    """

    # Copper physics constants
    RHO_COPPER = 1.68e-8  # Ω·m
    COPPER_THICKNESS = 35e-6  # 35μm (1oz copper)

    def __init__(self,
                 hidden_size: int = 64,
                 board_size: tuple = (100, 100),  # mm - bigger board for MNIST
                 input_grid_size: float = 30,     # mm - size of input pixel grid
                 device: str = None):
        """
        Args:
            hidden_size: Number of hidden neurons (their positions are trainable)
            board_size: PCB dimensions (width, height) in mm
            input_grid_size: Size of the 28x28 input grid region in mm
            device: 'cuda' or 'cpu'
        """
        super().__init__()

        self.input_size = 784  # 28x28
        self.hidden_size = hidden_size
        self.output_size = 10
        self.board_width, self.board_height = board_size
        self.input_grid_size = input_grid_size

        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # === TRAINABLE WEIGHTS ===
        # Xavier initialization
        self.W1 = nn.Parameter(torch.randn(self.input_size, hidden_size) * math.sqrt(2.0 / self.input_size))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.W2 = nn.Parameter(torch.randn(hidden_size, self.output_size) * math.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(self.output_size))

        # === FIXED INPUT POSITIONS (28x28 grid on left side) ===
        # Arrange inputs as actual image pixels!
        input_positions = []
        grid_start_x = 5  # mm from left edge
        grid_start_y = (self.board_height - input_grid_size) / 2  # centered vertically
        pixel_spacing = input_grid_size / 28

        for row in range(28):
            for col in range(28):
                x = grid_start_x + col * pixel_spacing
                y = grid_start_y + (27 - row) * pixel_spacing  # flip Y so 0,0 is top-left
                input_positions.append([x, y])

        self.register_buffer('input_positions', torch.tensor(input_positions, dtype=torch.float32))

        # === TRAINABLE HIDDEN POSITIONS ===
        # Initialize in the middle region of the board
        hidden_x = torch.rand(hidden_size) * 30 + 40  # x: 40-70mm
        hidden_y = torch.rand(hidden_size) * (self.board_height - 20) + 10  # y: 10-90mm
        self.hidden_positions = nn.Parameter(torch.stack([hidden_x, hidden_y], dim=1))

        # === FIXED OUTPUT POSITIONS (right edge, spaced vertically) ===
        output_positions = []
        output_x = self.board_width - 5  # mm from right edge
        output_spacing = (self.board_height - 20) / (self.output_size - 1)

        for i in range(self.output_size):
            y = 10 + i * output_spacing
            output_positions.append([output_x, y])

        self.register_buffer('output_positions', torch.tensor(output_positions, dtype=torch.float32))

        # Trace constraints
        self.min_trace_width = 0.09  # mm
        self.max_trace_width = 2.0   # mm (smaller for dense routing)
        self.min_spacing = 3.0       # mm between hidden neurons

    def to(self, device):
        """Move to device."""
        super().to(device)
        self.device = device
        return self

    def compute_distances(self, pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise Manhattan distances between two sets of positions.
        Vectorized for GPU efficiency.

        Args:
            pos1: (N, 2) positions
            pos2: (M, 2) positions

        Returns:
            (N, M) distance matrix
        """
        # Manhattan distance: |x1-x2| + |y1-y2|
        dx = torch.abs(pos1[:, 0:1] - pos2[:, 0:1].T)  # (N, M)
        dy = torch.abs(pos1[:, 1:2] - pos2[:, 1:2].T)  # (N, M)
        return dx + dy

    def weight_to_conductance(self, weight: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """
        Convert weight and trace length to effective conductance.
        Vectorized for GPU.

        Args:
            weight: Weight matrix
            length: Distance matrix (same shape)

        Returns:
            Conductance matrix
        """
        # Map weight magnitude to trace width
        w_normalized = (torch.clamp(torch.abs(weight), 0, 3) / 3)
        width = self.min_trace_width + w_normalized * (self.max_trace_width - self.min_trace_width)

        # Trace resistance: R = ρL / A
        length_m = length / 1000
        width_m = width / 1000
        area = width_m * self.COPPER_THICKNESS

        R = self.RHO_COPPER * length_m / (area + 1e-12)

        # Conductance with baseline
        conductance = 1.0 / (R + 0.1)

        return conductance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physical trace simulation.
        Fully vectorized for GPU acceleration.

        Args:
            x: Input tensor (batch_size, 784)

        Returns:
            Output logits (batch_size, 10)
        """
        batch_size = x.shape[0]

        # === Layer 1: Input → Hidden ===
        # Compute all input-to-hidden distances at once
        dist_ih = self.compute_distances(self.input_positions, self.hidden_positions)  # (784, hidden)

        # Physical conductance modulates weights
        conductance_ih = self.weight_to_conductance(self.W1, dist_ih)
        effective_W1 = self.W1 * conductance_ih

        # Matrix multiply (fully vectorized!)
        hidden_pre = torch.matmul(x, effective_W1) + self.b1  # (batch, hidden)
        hidden = torch.relu(hidden_pre)  # ReLU activation

        # === Layer 2: Hidden → Output ===
        dist_ho = self.compute_distances(self.hidden_positions, self.output_positions)  # (hidden, 10)

        conductance_ho = self.weight_to_conductance(self.W2, dist_ho)
        effective_W2 = self.W2 * conductance_ho

        output = torch.matmul(hidden, effective_W2) + self.b2  # (batch, 10)

        return output

    def layout_loss(self) -> torch.Tensor:
        """
        Physical layout constraints.
        """
        losses = []

        pos = self.hidden_positions

        # 1. Keep hidden neurons in valid region (not overlapping input grid or outputs)
        margin = 5.0
        input_region_end = 5 + self.input_grid_size + margin
        output_region_start = self.board_width - 10

        # Must be right of input region
        left_violation = torch.relu(input_region_end - pos[:, 0])
        # Must be left of output region
        right_violation = torch.relu(pos[:, 0] - output_region_start)
        # Must be on board (with margin)
        top_violation = torch.relu(pos[:, 1] - (self.board_height - margin))
        bottom_violation = torch.relu(margin - pos[:, 1])

        bounds_loss = (left_violation.sum() + right_violation.sum() +
                       top_violation.sum() + bottom_violation.sum()) * 10.0
        losses.append(bounds_loss)

        # 2. Minimum spacing between hidden neurons
        n = self.hidden_size
        if n > 1:
            # Pairwise Euclidean distances
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (n, n, 2)
            dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)  # (n, n)

            # Only upper triangle (avoid double counting)
            mask = torch.triu(torch.ones(n, n, device=pos.device), diagonal=1)
            pairwise_dist = dist * mask

            # Penalize if closer than min_spacing
            spacing_violations = torch.relu(self.min_spacing - pairwise_dist) * mask
            spacing_loss = spacing_violations.sum() * 5.0
            losses.append(spacing_loss)

        # 3. Total trace length (encourage compact layout)
        # Input to hidden total length
        dist_ih = self.compute_distances(self.input_positions, self.hidden_positions)
        # Weight by absolute weight (important connections should be short)
        weighted_dist_ih = dist_ih * torch.abs(self.W1)
        length_loss_ih = weighted_dist_ih.sum() / (self.input_size * self.hidden_size)

        # Hidden to output
        dist_ho = self.compute_distances(self.hidden_positions, self.output_positions)
        weighted_dist_ho = dist_ho * torch.abs(self.W2)
        length_loss_ho = weighted_dist_ho.sum() / (self.hidden_size * self.output_size)

        length_loss = (length_loss_ih + length_loss_ho) * 0.01
        losses.append(length_loss)

        return sum(losses)

    def get_layout_stats(self) -> dict:
        """Get statistics about current layout."""
        pos = self.hidden_positions.detach()

        # Pairwise distances
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)
        mask = torch.triu(torch.ones_like(dist), diagonal=1).bool()
        pairwise = dist[mask]

        # Total trace length
        dist_ih = self.compute_distances(self.input_positions, self.hidden_positions)
        dist_ho = self.compute_distances(self.hidden_positions, self.output_positions)
        total_length = dist_ih.sum() + dist_ho.sum()

        return {
            'min_spacing': pairwise.min().item() if len(pairwise) > 0 else 0,
            'mean_spacing': pairwise.mean().item() if len(pairwise) > 0 else 0,
            'total_trace_length_mm': total_length.item(),
            'hidden_x_range': (pos[:, 0].min().item(), pos[:, 0].max().item()),
            'hidden_y_range': (pos[:, 1].min().item(), pos[:, 1].max().item()),
        }

    def summary(self) -> str:
        """Print network summary."""
        stats = self.get_layout_stats()
        lines = [
            "=" * 50,
            "MNIST Physical Neural Network",
            "=" * 50,
            f"Architecture: {self.input_size} → {self.hidden_size} → {self.output_size}",
            f"Board size: {self.board_width}mm × {self.board_height}mm",
            f"Device: {self.device}",
            "",
            "Layout Statistics:",
            f"  Min neuron spacing: {stats['min_spacing']:.2f}mm",
            f"  Mean neuron spacing: {stats['mean_spacing']:.2f}mm",
            f"  Total trace length: {stats['total_trace_length_mm']:.0f}mm",
            f"  Hidden X range: {stats['hidden_x_range'][0]:.1f} - {stats['hidden_x_range'][1]:.1f}mm",
            f"  Hidden Y range: {stats['hidden_y_range'][0]:.1f} - {stats['hidden_y_range'][1]:.1f}mm",
            "=" * 50,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    print("Testing MNIST Physical Network...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MNISTPhysicalNetwork(hidden_size=64, board_size=(100, 100)).to(device)

    # Test forward pass
    x = torch.randn(32, 784).to(device)
    y = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Layout loss: {model.layout_loss().item():.4f}")
    print(f"\n{model.summary()}")
