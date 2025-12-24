"""
Physical Neural Network with trainable neuron positions.
Weights are encoded as copper trace geometry, and positions are optimized
alongside weights for optimal PCB layout.
"""

import torch
import torch.nn as nn
from physics import CopperPhysics


class PhysicalNeuralNetwork(nn.Module):
    """
    Neural network that exists in physical space.

    Weights are encoded as copper trace geometry.
    Neuron positions are trainable parameters that get optimized
    alongside the weights to create an optimal physical layout.
    """

    def __init__(self,
                 input_size: int = 2,
                 hidden_size: int = 4,
                 output_size: int = 1,
                 board_size: tuple = (50, 50)):  # mm
        """
        Initialize the physical neural network.

        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            board_size: PCB board dimensions (width, height) in mm
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.board_width, self.board_height = board_size

        # Physics simulator
        self.physics = CopperPhysics()

        # Standard neural network weights
        self.W1 = nn.Parameter(torch.randn(input_size, hidden_size) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.W2 = nn.Parameter(torch.randn(hidden_size, output_size) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(output_size))

        # Detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Fixed positions: Input neurons at left edge
        self.register_buffer('input_positions', torch.tensor([
            [2.0, self.board_height * (i + 1) / (input_size + 1)]
            for i in range(input_size)
        ], dtype=torch.float32))

        # Trainable positions: Hidden neurons (initialized randomly)
        self.hidden_positions = nn.Parameter(
            torch.rand(hidden_size, 2) * torch.tensor([
                self.board_width * 0.6,  # Start in left 60%
                self.board_height
            ]) + torch.tensor([self.board_width * 0.2, 0.0])  # Offset from edge
        )

        # Fixed positions: Output neurons at right edge
        self.register_buffer('output_positions', torch.tensor([
            [self.board_width - 2.0, self.board_height * (i + 1) / (output_size + 1)]
            for i in range(output_size)
        ], dtype=torch.float32))

        # Trace width constraints (JLC PCB limits)
        self.min_trace_width = 0.09  # mm (JLC minimum)
        self.max_trace_width = 5.0   # mm (practical max)

        # Spacing constraints
        self.min_spacing = 2.0  # mm between neurons

    def weight_to_trace_width(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Map neural network weight to physical trace width.

        Larger weight → wider trace → lower resistance → stronger connection

        Args:
            weight: Neural network weight value

        Returns:
            Trace width in mm
        """
        # Normalize weight to [0, 1]
        w_min = -3.0
        w_max = 3.0
        normalized = (torch.clamp(weight, w_min, w_max) - w_min) / (w_max - w_min)

        # Map to trace width range
        width = self.min_trace_width + normalized * (self.max_trace_width - self.min_trace_width)

        return width

    def compute_effective_weight(self, weight: torch.Tensor,
                                  trace_length_mm: torch.Tensor) -> torch.Tensor:
        """
        Calculate effective weight including trace resistance.

        Longer traces have higher parasitic resistance,
        which attenuates the signal.

        Args:
            weight: Original weight value
            trace_length_mm: Physical trace length

        Returns:
            Effective weight after physical attenuation
        """
        # Ideal trace width for this weight
        width = self.weight_to_trace_width(weight)

        # Physical resistance
        R_trace = self.physics.trace_resistance_torch(trace_length_mm, width)

        # Effective conductance (inverse of resistance)
        # Higher conductance = stronger connection
        conductance = 1.0 / (R_trace + 1.0)  # +1 to prevent division issues

        # Scale original weight by conductance
        # This is how physical layout affects computation!
        effective_weight = weight * conductance

        return effective_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physical trace simulation.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size = x.shape[0]

        # Layer 1: Input → Hidden
        # Calculate trace lengths from each input to each hidden neuron
        hidden_activations = torch.zeros(batch_size, self.hidden_size,
                                          device=x.device, dtype=x.dtype)

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                # Physical distance (Manhattan for PCB routing)
                trace_length = self.physics.manhattan_distance(
                    self.input_positions[i],
                    self.hidden_positions[j]
                )

                # Effective weight after physical attenuation
                w_eff = self.compute_effective_weight(self.W1[i, j], trace_length)

                # Accumulate weighted input
                hidden_activations[:, j] += x[:, i] * w_eff

        # Add bias and activation (sigmoid = diode response approximation)
        hidden = torch.sigmoid(hidden_activations + self.b1)

        # Layer 2: Hidden → Output
        output_activations = torch.zeros(batch_size, self.output_size,
                                          device=x.device, dtype=x.dtype)

        for j in range(self.hidden_size):
            for k in range(self.output_size):
                trace_length = self.physics.manhattan_distance(
                    self.hidden_positions[j],
                    self.output_positions[k]
                )

                w_eff = self.compute_effective_weight(self.W2[j, k], trace_length)

                output_activations[:, k] += hidden[:, j] * w_eff

        output = torch.sigmoid(output_activations + self.b2)

        return output

    def layout_loss(self) -> torch.Tensor:
        """
        Compute penalty for layouts that are hard to manufacture.

        Returns:
            Layout penalty (lower is better)
        """
        losses = []

        hidden_x = self.hidden_positions[:, 0]
        hidden_y = self.hidden_positions[:, 1]

        # 1. Keep neurons within board bounds (hard constraint)
        margin = 3.0  # mm from edge
        off_left = torch.relu(margin - hidden_x)
        off_right = torch.relu(hidden_x - (self.board_width - margin))
        off_bottom = torch.relu(margin - hidden_y)
        off_top = torch.relu(hidden_y - (self.board_height - margin))

        bounds_penalty = (off_left.sum() + off_right.sum() +
                          off_bottom.sum() + off_top.sum()) * 100.0
        losses.append(bounds_penalty)

        # 2. Minimum spacing between neurons (manufacturing constraint)
        spacing_penalty = torch.tensor(0.0)
        for i in range(self.hidden_size):
            for j in range(i + 1, self.hidden_size):
                dist = self.physics.euclidean_distance(
                    self.hidden_positions[i],
                    self.hidden_positions[j]
                )
                # Penalize if closer than min_spacing
                violation = torch.relu(self.min_spacing - dist)
                spacing_penalty = spacing_penalty + violation * 10.0

        losses.append(spacing_penalty)

        # 3. Total trace length (want short connections = less copper, faster)
        total_length = torch.tensor(0.0)

        # Input → Hidden traces
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                length = self.physics.manhattan_distance(
                    self.input_positions[i],
                    self.hidden_positions[j]
                )
                total_length = total_length + length

        # Hidden → Output traces
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                length = self.physics.manhattan_distance(
                    self.hidden_positions[j],
                    self.output_positions[k]
                )
                total_length = total_length + length

        # Normalize by expected total length
        wire_penalty = total_length / 500.0
        losses.append(wire_penalty)

        # 4. Encourage left-to-right flow (hidden neurons between input and output)
        flow_penalty = torch.tensor(0.0)
        for j in range(self.hidden_size):
            # Hidden neurons should be in the middle third
            x = self.hidden_positions[j, 0]
            if x < self.board_width * 0.2:
                flow_penalty = flow_penalty + (self.board_width * 0.2 - x)
            elif x > self.board_width * 0.8:
                flow_penalty = flow_penalty + (x - self.board_width * 0.8)

        losses.append(flow_penalty * 0.5)

        return sum(losses)

    def get_trace_info(self) -> dict:
        """
        Get information about all traces for visualization/export.

        Returns:
            Dictionary with trace specifications
        """
        traces = {
            'input_to_hidden': [],
            'hidden_to_output': []
        }

        # Input → Hidden
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                length = self.physics.manhattan_distance(
                    self.input_positions[i],
                    self.hidden_positions[j]
                ).item()

                weight = self.W1[i, j].item()
                width = self.weight_to_trace_width(self.W1[i, j]).item()

                traces['input_to_hidden'].append({
                    'from': (self.input_positions[i, 0].item(),
                             self.input_positions[i, 1].item()),
                    'to': (self.hidden_positions[j, 0].item(),
                           self.hidden_positions[j, 1].item()),
                    'weight': weight,
                    'width_mm': width,
                    'length_mm': length,
                    'from_idx': i,
                    'to_idx': j
                })

        # Hidden → Output
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                length = self.physics.manhattan_distance(
                    self.hidden_positions[j],
                    self.output_positions[k]
                ).item()

                weight = self.W2[j, k].item()
                width = self.weight_to_trace_width(self.W2[j, k]).item()

                traces['hidden_to_output'].append({
                    'from': (self.hidden_positions[j, 0].item(),
                             self.hidden_positions[j, 1].item()),
                    'to': (self.output_positions[k, 0].item(),
                           self.output_positions[k, 1].item()),
                    'weight': weight,
                    'width_mm': width,
                    'length_mm': length,
                    'from_idx': j,
                    'to_idx': k
                })

        return traces

    def summary(self) -> str:
        """Print a summary of the network's physical layout."""
        traces = self.get_trace_info()

        total_length = sum(t['length_mm'] for t in traces['input_to_hidden'])
        total_length += sum(t['length_mm'] for t in traces['hidden_to_output'])

        lines = [
            "=== Physical Neural Network Summary ===",
            f"Board size: {self.board_width}mm × {self.board_height}mm",
            f"Architecture: {self.input_size} → {self.hidden_size} → {self.output_size}",
            f"Total trace length: {total_length:.1f}mm",
            "",
            "Hidden neuron positions (mm):"
        ]

        for j in range(self.hidden_size):
            x, y = self.hidden_positions[j].detach().numpy()
            lines.append(f"  H{j}: ({x:.1f}, {y:.1f})")

        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    model = PhysicalNeuralNetwork(
        input_size=2,
        hidden_size=4,
        output_size=1,
        board_size=(50, 50)
    )

    # Test forward pass
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = model(X)

    print(model.summary())
    print("\nTest forward pass:")
    for inp, out in zip(X, y):
        print(f"  {inp.numpy()} → {out.item():.3f}")

    print(f"\nLayout loss: {model.layout_loss().item():.3f}")
