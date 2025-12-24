"""
MNIST Unleashed - Scaled Physical Neural Network

256 neurons, 200mm board, relaxed constraints.
Let's see what geometry intelligence naturally wants!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MNISTUnleashed(nn.Module):
    """
    Physical Neural Network with room to breathe.

    - 256 hidden neurons (4x more capacity)
    - 200mm x 200mm board (4x area)
    - 0.15mm minimum spacing (20x tighter than before!)
    - 4-layer PCB support (neurons have Z coordinate)
    - Serpentine trace awareness
    """

    # Copper physics
    RHO_COPPER = 1.68e-8  # Ω·m
    COPPER_THICKNESS = 35e-6  # 35μm (1oz)

    def __init__(self,
                 hidden_size: int = 256,
                 board_size: tuple = (200, 200),
                 n_layers: int = 4,
                 input_grid_size: float = 50,
                 min_spacing: float = 0.15,
                 min_trace_width: float = 0.09,
                 max_trace_width: float = 5.0,
                 device: str = None):
        """
        Args:
            hidden_size: Number of hidden neurons (256 recommended)
            board_size: PCB dimensions (width, height) in mm
            n_layers: Number of copper layers (2, 4, or 6)
            input_grid_size: Size of 28x28 input region in mm
            min_spacing: Minimum neuron spacing in mm
            min_trace_width: JLC minimum trace width
            max_trace_width: Maximum practical trace width
        """
        super().__init__()

        self.input_size = 784
        self.hidden_size = hidden_size
        self.output_size = 10
        self.board_width, self.board_height = board_size
        self.n_layers = n_layers
        self.input_grid_size = input_grid_size
        self.min_spacing = min_spacing
        self.min_trace_width = min_trace_width
        self.max_trace_width = max_trace_width

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # === NETWORK WEIGHTS (Xavier init) ===
        self.W1 = nn.Parameter(torch.randn(self.input_size, hidden_size) * math.sqrt(2.0 / self.input_size))
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.W2 = nn.Parameter(torch.randn(hidden_size, self.output_size) * math.sqrt(2.0 / hidden_size))
        self.b2 = nn.Parameter(torch.zeros(self.output_size))

        # === INPUT POSITIONS (28x28 grid, larger spacing) ===
        input_positions = []
        grid_start_x = 10
        grid_start_y = (self.board_height - input_grid_size) / 2
        pixel_spacing = input_grid_size / 28

        for row in range(28):
            for col in range(28):
                x = grid_start_x + col * pixel_spacing
                y = grid_start_y + (27 - row) * pixel_spacing
                input_positions.append([x, y])

        self.register_buffer('input_positions', torch.tensor(input_positions, dtype=torch.float32))

        # === HIDDEN POSITIONS (trainable, 3D: x, y, layer) ===
        # Initialize spread across the middle region
        hidden_x = torch.rand(hidden_size) * 100 + 70  # x: 70-170mm
        hidden_y = torch.rand(hidden_size) * 160 + 20  # y: 20-180mm
        # Layer assignment (continuous, will be discretized for routing)
        hidden_layer = torch.rand(hidden_size) * (n_layers - 1)  # 0 to n_layers-1

        self.hidden_positions = nn.Parameter(torch.stack([hidden_x, hidden_y], dim=1))
        self.hidden_layers = nn.Parameter(hidden_layer)  # Which PCB layer

        # === OUTPUT POSITIONS (right edge) ===
        output_positions = []
        output_x = self.board_width - 10
        output_spacing = (self.board_height - 40) / (self.output_size - 1)

        for i in range(self.output_size):
            y = 20 + i * output_spacing
            output_positions.append([output_x, y])

        self.register_buffer('output_positions', torch.tensor(output_positions, dtype=torch.float32))

    def compute_distances_3d(self, pos1: torch.Tensor, pos2: torch.Tensor,
                              layers1: torch.Tensor = None, layers2: torch.Tensor = None) -> torch.Tensor:
        """
        Compute pairwise distances including layer transitions (vias).

        Via adds ~0.5mm equivalent routing length per layer crossed.
        """
        # 2D Manhattan distance
        dx = torch.abs(pos1[:, 0:1] - pos2[:, 0:1].T)
        dy = torch.abs(pos1[:, 1:2] - pos2[:, 1:2].T)
        dist_2d = dx + dy

        # Add via penalty if layers provided
        if layers1 is not None and layers2 is not None:
            layer_diff = torch.abs(layers1.unsqueeze(1) - layers2.unsqueeze(0))
            via_penalty = layer_diff * 0.5  # 0.5mm per layer transition
            return dist_2d + via_penalty

        return dist_2d

    def weight_to_conductance(self, weight: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """
        Physical conductance based on trace geometry.
        Longer/thinner traces = more resistance = lower conductance.
        """
        # Weight magnitude → trace width
        w_norm = torch.clamp(torch.abs(weight), 0, 3) / 3
        width = self.min_trace_width + w_norm * (self.max_trace_width - self.min_trace_width)

        # Resistance: R = ρL / A
        length_m = length / 1000
        width_m = width / 1000
        area = width_m * self.COPPER_THICKNESS

        R = self.RHO_COPPER * length_m / (area + 1e-12)

        # Conductance (higher = stronger signal)
        conductance = 1.0 / (R + 0.05)  # Lower baseline for more sensitivity

        return conductance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with physical simulation."""
        batch_size = x.shape[0]

        # Layer 1: Input → Hidden
        # Input is on layer 0, hidden neurons have their own layers
        input_layers = torch.zeros(self.input_size, device=x.device)
        dist_ih = self.compute_distances_3d(
            self.input_positions, self.hidden_positions,
            input_layers, self.hidden_layers
        )

        conductance_ih = self.weight_to_conductance(self.W1, dist_ih)
        effective_W1 = self.W1 * conductance_ih

        hidden_pre = torch.matmul(x, effective_W1) + self.b1
        hidden = F.relu(hidden_pre)

        # Layer 2: Hidden → Output
        # Output is on last layer
        output_layers = torch.ones(self.output_size, device=x.device) * (self.n_layers - 1)
        dist_ho = self.compute_distances_3d(
            self.hidden_positions, self.output_positions,
            self.hidden_layers, output_layers
        )

        conductance_ho = self.weight_to_conductance(self.W2, dist_ho)
        effective_W2 = self.W2 * conductance_ho

        output = torch.matmul(hidden, effective_W2) + self.b2

        return output

    def layout_loss(self) -> torch.Tensor:
        """
        Physical constraints - RELAXED for more freedom.
        """
        losses = []
        pos = self.hidden_positions
        layers = self.hidden_layers

        # 1. Stay within board bounds (with margin)
        margin = 5.0
        input_end = 10 + self.input_grid_size + 10  # After input region
        output_start = self.board_width - 20

        left_violation = torch.relu(input_end - pos[:, 0])
        right_violation = torch.relu(pos[:, 0] - output_start)
        top_violation = torch.relu(pos[:, 1] - (self.board_height - margin))
        bottom_violation = torch.relu(margin - pos[:, 1])

        bounds_loss = (left_violation.sum() + right_violation.sum() +
                       top_violation.sum() + bottom_violation.sum()) * 5.0
        losses.append(bounds_loss)

        # 2. Minimum spacing (relaxed to 0.15mm!)
        n = self.hidden_size
        if n > 1:
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)
            mask = torch.triu(torch.ones(n, n, device=pos.device), diagonal=1)

            # Also consider layer separation (different layers can be closer)
            layer_diff = torch.abs(layers.unsqueeze(0) - layers.unsqueeze(1))
            effective_min_spacing = self.min_spacing / (1 + layer_diff * 0.5)  # Layers help!

            spacing_violations = torch.relu(effective_min_spacing - dist) * mask
            spacing_loss = spacing_violations.sum() * 2.0  # Reduced penalty
            losses.append(spacing_loss)

        # 3. Layer bounds (0 to n_layers-1)
        layer_violation = torch.relu(-layers) + torch.relu(layers - (self.n_layers - 1))
        losses.append(layer_violation.sum() * 10.0)

        # 4. Weighted trace length (important connections should be shorter)
        dist_ih = self.compute_distances_3d(
            self.input_positions, self.hidden_positions,
            torch.zeros(self.input_size, device=pos.device), self.hidden_layers
        )
        weighted_ih = (dist_ih * torch.abs(self.W1)).sum() / (self.input_size * self.hidden_size)

        dist_ho = self.compute_distances_3d(
            self.hidden_positions, self.output_positions,
            self.hidden_layers, torch.ones(self.output_size, device=pos.device) * (self.n_layers - 1)
        )
        weighted_ho = (dist_ho * torch.abs(self.W2)).sum() / (self.hidden_size * self.output_size)

        # Very gentle trace length penalty - let structure emerge!
        length_loss = (weighted_ih + weighted_ho) * 0.002
        losses.append(length_loss)

        # 5. Encourage layer utilization (spread across layers, don't bunch up)
        layer_std = layers.std()
        layer_spread_bonus = -layer_std * 0.1  # Negative = reward spread
        losses.append(layer_spread_bonus)

        return sum(losses)

    def get_layout_stats(self) -> dict:
        """Get comprehensive layout statistics."""
        pos = self.hidden_positions.detach()
        layers = self.hidden_layers.detach()

        # Pairwise distances
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-8)
        mask = torch.triu(torch.ones_like(dist), diagonal=1).bool()
        pairwise = dist[mask]

        # Total trace length
        dist_ih = self.compute_distances_3d(
            self.input_positions, pos,
            torch.zeros(self.input_size, device=pos.device), layers
        )
        dist_ho = self.compute_distances_3d(
            pos, self.output_positions,
            layers, torch.ones(self.output_size, device=pos.device) * (self.n_layers - 1)
        )
        total_length = dist_ih.sum() + dist_ho.sum()

        # Layer distribution
        layer_counts = [(layers >= i - 0.5).sum().item() - (layers >= i + 0.5).sum().item()
                        for i in range(self.n_layers)]

        return {
            'min_spacing': pairwise.min().item() if len(pairwise) > 0 else 0,
            'mean_spacing': pairwise.mean().item() if len(pairwise) > 0 else 0,
            'total_trace_length_mm': total_length.item(),
            'total_trace_length_m': total_length.item() / 1000,
            'hidden_x_range': (pos[:, 0].min().item(), pos[:, 0].max().item()),
            'hidden_y_range': (pos[:, 1].min().item(), pos[:, 1].max().item()),
            'layer_distribution': layer_counts,
            'mean_layer': layers.mean().item(),
        }

    def get_neuron_clusters(self, n_clusters: int = 10) -> torch.Tensor:
        """
        Identify spatial clusters of neurons.
        Returns cluster assignment for each hidden neuron.
        """
        from sklearn.cluster import KMeans

        pos = self.hidden_positions.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pos)

        return torch.tensor(clusters), kmeans.cluster_centers_

    def summary(self) -> str:
        stats = self.get_layout_stats()
        lines = [
            "=" * 60,
            "MNIST UNLEASHED - Physical Neural Network",
            "=" * 60,
            f"Architecture: {self.input_size} → {self.hidden_size} → {self.output_size}",
            f"Board: {self.board_width}mm × {self.board_height}mm ({self.n_layers} layers)",
            f"Constraints: {self.min_spacing}mm spacing, {self.min_trace_width}-{self.max_trace_width}mm traces",
            f"Device: {self.device}",
            "",
            "Layout Statistics:",
            f"  Min neuron spacing: {stats['min_spacing']:.3f}mm",
            f"  Mean neuron spacing: {stats['mean_spacing']:.2f}mm",
            f"  Total trace length: {stats['total_trace_length_m']:.1f}m ({stats['total_trace_length_mm']/1000:.1f}km)",
            f"  Hidden X range: {stats['hidden_x_range'][0]:.1f} - {stats['hidden_x_range'][1]:.1f}mm",
            f"  Hidden Y range: {stats['hidden_y_range'][0]:.1f} - {stats['hidden_y_range'][1]:.1f}mm",
            f"  Layer distribution: {stats['layer_distribution']}",
            f"  Mean layer: {stats['mean_layer']:.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print("Testing MNIST Unleashed...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MNISTUnleashed(
        hidden_size=256,
        board_size=(200, 200),
        n_layers=4,
        min_spacing=0.15
    ).to(device)

    # Test forward pass
    x = torch.randn(32, 784).to(device)
    y = model(x)

    print(f"\nInput: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Layout loss: {model.layout_loss().item():.4f}")
    print(f"\n{model.summary()}")
