# Let's Build The Material Compiler! 🦩⚡

Your list just keeps getting better:

**Eyeglasses with optical neural networks** = Real-time style transfer on your retina. Van Gogh mode while walking through Yorkshire!

**Holographic SLMs** = Oh god yes, programmable diffraction patterns. Could literally "print" holographic AI processors.

**Intelligent diagnostic stickers** = Stick on forehead, changes color based on biomarkers. Zero-power continuous health monitoring.

**Intelligent windows** = Adaptive tinting that learns your preferences. "Darken when I'm at my desk, clear when I'm reading."

But you're right—**PCB first**. Master one domain, then the framework naturally generalizes.

---

# The PCB Neural Network Trainer: Let's Code This

## File Structure for Tonight

```
physical-neural-network/
├── train_xor.py           # Train XOR with position optimization
├── physics.py             # Copper trace resistance calculations
├── visualize.py           # Watch it condense!
├── export_gerber.py       # Output → JLCPCB
└── requirements.txt       # torch, matplotlib, numpy
```

## Step 1: The Physics Engine

```python
# physics.py
"""
Physical simulation of copper PCB traces.
All the actual electronics calculations.
"""

import torch
import numpy as np

class CopperPhysics:
    """Calculate electrical properties of copper traces."""
    
    # Material constants
    RHO_COPPER = 1.68e-8  # Ω·m (resistivity at 20°C)
    TEMP_COEFF = 0.00393  # Temperature coefficient (1/°C)
    THERMAL_CONDUCTIVITY = 385  # W/(m·K)
    
    def __init__(self, 
                 copper_thickness_um=35,  # Standard 1oz copper
                 board_thickness_mm=1.6,   # Standard FR4
                 ambient_temp_c=25):
        
        self.copper_thickness = copper_thickness_um / 1e6  # Convert to meters
        self.board_thickness = board_thickness_mm / 1000
        self.ambient_temp = ambient_temp_c
    
    def trace_resistance(self, length_mm, width_mm, temperature_c=None):
        """
        Calculate resistance of a copper trace.
        
        R = ρ × L / A
        where A = width × thickness
        
        Returns: Resistance in Ω
        """
        if temperature_c is None:
            temperature_c = self.ambient_temp
        
        # Adjust resistivity for temperature
        temp_diff = temperature_c - 20  # Reference temp
        rho = self.RHO_COPPER * (1 + self.TEMP_COEFF * temp_diff)
        
        # Convert to meters
        length_m = length_mm / 1000
        width_m = width_mm / 1000
        
        # Cross-sectional area
        area = width_m * self.copper_thickness
        
        # Resistance
        R = rho * length_m / area
        
        return R
    
    def trace_resistance_torch(self, length_mm, width_mm):
        """
        Torch version for backprop.
        Simplified (no temperature for now).
        """
        rho = self.RHO_COPPER
        
        length_m = length_mm / 1000
        width_m = width_mm / 1000
        
        area = width_m * self.copper_thickness
        
        # Add small epsilon to prevent division by zero
        R = rho * length_m / (area + 1e-12)
        
        return R
    
    def manhattan_distance(self, pos1, pos2):
        """
        Calculate Manhattan (L1) distance between two points.
        PCB traces follow grid, so this is more realistic than Euclidean.
        """
        return torch.abs(pos1[0] - pos2[0]) + torch.abs(pos1[1] - pos2[1])
    
    def via_resistance(self, diameter_mm):
        """
        Resistance through a via connecting layers.
        """
        # Simplified model
        plating_thickness = 25e-6  # 25 μm plating
        circumference = np.pi * (diameter_mm / 1000)
        area = circumference * plating_thickness
        
        # Via length = board thickness
        R = self.RHO_COPPER * self.board_thickness / area
        
        return R
    
    def current_capacity(self, width_mm, temp_rise_c=10):
        """
        Maximum safe current for a trace.
        Based on IPC-2221 standards.
        
        Returns: Current in Amps
        """
        # IPC-2221 formula (simplified)
        # I = k × ΔT^0.44 × A^0.725
        
        k = 0.048  # Internal layer
        area_sq_mil = (width_mm * 1000 / 25.4) * (self.copper_thickness * 1e6 / 25.4)
        
        I = k * (temp_rise_c ** 0.44) * (area_sq_mil ** 0.725)
        
        return I
    
    def power_dissipation(self, current_a, resistance_ohm):
        """
        Power dissipated as heat: P = I²R
        """
        return current_a ** 2 * resistance_ohm


# Quick test
if __name__ == "__main__":
    physics = CopperPhysics()
    
    print("=== Copper Trace Physics ===\n")
    
    # Test various trace geometries
    test_cases = [
        (10, 0.1, "Thin trace"),
        (10, 0.5, "Medium trace"),
        (10, 2.0, "Fat trace"),
        (50, 0.1, "Long thin trace"),
    ]
    
    for length, width, description in test_cases:
        R = physics.trace_resistance(length, width)
        I_max = physics.current_capacity(width)
        print(f"{description}:")
        print(f"  {length}mm × {width}mm = {R:.3f}Ω")
        print(f"  Max current: {I_max:.3f}A")
        print()
```

## Step 2: The Neural Network with Physical Layout

```python
# train_xor.py
"""
Train XOR network where neuron positions are optimized
alongside weights for optimal PCB layout.
"""

import torch
import torch.nn as nn
import numpy as np
from physics import CopperPhysics

class PhysicalNeuralNetwork(nn.Module):
    """
    Neural network that exists in physical space.
    
    Weights are encoded as copper trace geometry.
    Neuron positions are trainable parameters.
    """
    
    def __init__(self, 
                 input_size=2, 
                 hidden_size=4, 
                 output_size=1,
                 board_size=(50, 50)):  # mm
        
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
        
        # NEW: Physical positions (also trainable!)
        # Input positions are fixed at left edge
        self.input_positions = torch.tensor([
            [0.0, self.board_height * (i + 1) / (input_size + 1)]
            for i in range(input_size)
        ], dtype=torch.float32)
        
        # Hidden neuron positions (trainable)
        self.hidden_positions = nn.Parameter(
            torch.rand(hidden_size, 2) * torch.tensor([self.board_width, self.board_height])
        )
        
        # Output positions are fixed at right edge
        self.output_positions = torch.tensor([
            [self.board_width, self.board_height * (i + 1) / (output_size + 1)]
            for i in range(output_size)
        ], dtype=torch.float32)
        
        # Trace width constraints (JLC limits)
        self.min_trace_width = 0.09  # mm (JLC minimum)
        self.max_trace_width = 5.0   # mm (practical max)
    
    def weight_to_trace_width(self, weight):
        """
        Map neural network weight to physical trace width.
        
        Larger weight → wider trace → lower resistance → stronger connection
        """
        # Normalize weight to [0, 1]
        w_min = -3.0
        w_max = 3.0
        normalized = (torch.clamp(weight, w_min, w_max) - w_min) / (w_max - w_min)
        
        # Map to trace width range
        width = self.min_trace_width + normalized * (self.max_trace_width - self.min_trace_width)
        
        return width
    
    def compute_effective_weight(self, weight, trace_length_mm):
        """
        Calculate effective weight including trace resistance.
        
        Longer traces have higher parasitic resistance,
        which attenuates the signal.
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
    
    def forward(self, x):
        """
        Forward pass with physical trace simulation.
        """
        batch_size = x.shape[0]
        
        # Layer 1: Input → Hidden
        # Calculate trace lengths from each input to each hidden neuron
        hidden_activations = torch.zeros(batch_size, self.hidden_size)
        
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                # Physical distance (Manhattan)
                trace_length = self.physics.manhattan_distance(
                    self.input_positions[i],
                    self.hidden_positions[j]
                )
                
                # Effective weight after physical attenuation
                w_eff = self.compute_effective_weight(self.W1[i, j], trace_length)
                
                # Accumulate (this is the weighted sum)
                hidden_activations[:, j] += x[:, i] * w_eff
        
        # Add bias and activation
        hidden = torch.sigmoid(hidden_activations + self.b1)
        
        # Layer 2: Hidden → Output (similar process)
        output_activations = torch.zeros(batch_size, self.output_size)
        
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
    
    def layout_loss(self):
        """
        Penalize bad layouts that are hard to manufacture.
        """
        losses = []
        
        # 1. Board utilization (want compact design)
        hidden_x = self.hidden_positions[:, 0]
        hidden_y = self.hidden_positions[:, 1]
        
        used_width = hidden_x.max() - hidden_x.min()
        used_height = hidden_y.max() - hidden_y.min()
        
        # Penalize if we're using more than 80% of board
        width_penalty = torch.relu(used_width - 0.8 * self.board_width)
        height_penalty = torch.relu(used_height - 0.8 * self.board_height)
        
        losses.append(width_penalty + height_penalty)
        
        # 2. Minimum spacing (neurons too close)
        for i in range(self.hidden_size):
            for j in range(i + 1, self.hidden_size):
                dist = torch.norm(self.hidden_positions[i] - self.hidden_positions[j])
                # Neurons should be at least 2mm apart
                spacing_violation = torch.relu(2.0 - dist)
                losses.append(spacing_violation * 10)  # Heavy penalty
        
        # 3. Keep neurons on board
        off_board_x = torch.relu(hidden_x - self.board_width) + torch.relu(-hidden_x)
        off_board_y = torch.relu(hidden_y - self.board_height) + torch.relu(-hidden_y)
        losses.append((off_board_x.sum() + off_board_y.sum()) * 100)
        
        # 4. Total trace length (want short connections)
        total_length = 0.0
        
        # Input → Hidden traces
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                length = self.physics.manhattan_distance(
                    self.input_positions[i],
                    self.hidden_positions[j]
                )
                total_length += length
        
        # Hidden → Output traces
        for j in range(self.hidden_size):
            for k in range(self.output_size):
                length = self.physics.manhattan_distance(
                    self.hidden_positions[j],
                    self.output_positions[k]
                )
                total_length += length
        
        # Penalize excessive total wire length
        losses.append(total_length / 1000.0)  # Normalized
        
        return sum(losses)


def train_xor(epochs=5000, layout_weight=0.1):
    """
    Train XOR network with physical layout optimization.
    """
    # XOR dataset
    X = torch.tensor([[0.0, 0.0],
                      [0.0, 1.0],
                      [1.0, 0.0],
                      [1.0, 1.0]], dtype=torch.float32)
    
    y = torch.tensor([[0.0],
                      [1.0],
                      [1.0],
                      [0.0]], dtype=torch.float32)
    
    # Create network
    model = PhysicalNeuralNetwork(
        input_size=2,
        hidden_size=4,
        output_size=1,
        board_size=(50, 50)
    )
    
    # Optimizer - different learning rates for weights vs positions
    optimizer = torch.optim.Adam([
        {'params': [model.W1, model.W2, model.b1, model.b2], 'lr': 0.01},
        {'params': [model.hidden_positions], 'lr': 0.5}  # Positions can move faster
    ])
    
    # Training history
    history = {
        'classification_loss': [],
        'layout_loss': [],
        'total_loss': [],
        'accuracy': [],
        'hidden_positions': []
    }
    
    print("Training XOR with physical layout optimization...\n")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(X)
        
        # Classification loss
        classification_loss = nn.BCELoss()(y_pred, y)
        
        # Layout loss
        layout_loss = model.layout_loss()
        
        # Combined loss
        total_loss = classification_loss + layout_weight * layout_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        with torch.no_grad():
            accuracy = ((y_pred > 0.5).float() == y).float().mean()
            
            history['classification_loss'].append(classification_loss.item())
            history['layout_loss'].append(layout_loss.item())
            history['total_loss'].append(total_loss.item())
            history['accuracy'].append(accuracy.item())
            history['hidden_positions'].append(model.hidden_positions.detach().clone())
        
        # Print progress
        if epoch % 500 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Classification: {classification_loss:.4f}")
            print(f"  Layout: {layout_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2%}")
            print()
    
    return model, history


if __name__ == "__main__":
    model, history = train_xor(epochs=5000)
    
    # Final test
    print("\n=== Final Results ===")
    X_test = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_pred = model(X_test)
    print("XOR Truth Table:")
    for i, (inp, out) in enumerate(zip(X_test, y_pred)):
        print(f"  {inp.numpy()} → {out.item():.3f} ({'1' if out > 0.5 else '0'})")
```

