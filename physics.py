"""
Physical simulation of copper PCB traces.
All the actual electronics calculations for Physical Neural Networks.
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
                 copper_thickness_um: float = 35,  # Standard 1oz copper
                 board_thickness_mm: float = 1.6,   # Standard FR4
                 ambient_temp_c: float = 25):

        self.copper_thickness = copper_thickness_um / 1e6  # Convert to meters
        self.board_thickness = board_thickness_mm / 1000
        self.ambient_temp = ambient_temp_c

    def trace_resistance(self, length_mm: float, width_mm: float,
                         temperature_c: float = None) -> float:
        """
        Calculate resistance of a copper trace.

        R = ρ × L / A
        where A = width × thickness

        Args:
            length_mm: Trace length in millimeters
            width_mm: Trace width in millimeters
            temperature_c: Operating temperature (defaults to ambient)

        Returns:
            Resistance in Ohms
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

    def trace_resistance_torch(self, length_mm: torch.Tensor,
                               width_mm: torch.Tensor) -> torch.Tensor:
        """
        Torch-compatible version for backpropagation.
        Simplified (no temperature for now).

        Args:
            length_mm: Trace length (can be tensor)
            width_mm: Trace width (can be tensor)

        Returns:
            Resistance tensor in Ohms
        """
        rho = self.RHO_COPPER

        length_m = length_mm / 1000
        width_m = width_mm / 1000

        area = width_m * self.copper_thickness

        # Add small epsilon to prevent division by zero
        R = rho * length_m / (area + 1e-12)

        return R

    def manhattan_distance(self, pos1: torch.Tensor,
                           pos2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Manhattan (L1) distance between two points.
        PCB traces follow grid, so this is more realistic than Euclidean.

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)

        Returns:
            Manhattan distance in same units as input
        """
        return torch.abs(pos1[0] - pos2[0]) + torch.abs(pos1[1] - pos2[1])

    def euclidean_distance(self, pos1: torch.Tensor,
                           pos2: torch.Tensor) -> torch.Tensor:
        """
        Calculate Euclidean distance between two points.
        Used for spacing constraints.
        """
        return torch.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + 1e-8)

    def via_resistance(self, diameter_mm: float) -> float:
        """
        Resistance through a via connecting layers.

        Args:
            diameter_mm: Via diameter in millimeters

        Returns:
            Via resistance in Ohms
        """
        # Simplified model
        plating_thickness = 25e-6  # 25 μm plating
        circumference = np.pi * (diameter_mm / 1000)
        area = circumference * plating_thickness

        # Via length = board thickness
        R = self.RHO_COPPER * self.board_thickness / area

        return R

    def current_capacity(self, width_mm: float, temp_rise_c: float = 10) -> float:
        """
        Maximum safe current for a trace.
        Based on IPC-2221 standards.

        Args:
            width_mm: Trace width in millimeters
            temp_rise_c: Allowable temperature rise

        Returns:
            Maximum current in Amps
        """
        # IPC-2221 formula (simplified)
        # I = k × ΔT^0.44 × A^0.725

        k = 0.048  # Internal layer
        area_sq_mil = (width_mm * 1000 / 25.4) * (self.copper_thickness * 1e6 / 25.4)

        I = k * (temp_rise_c ** 0.44) * (area_sq_mil ** 0.725)

        return I

    def power_dissipation(self, current_a: float, resistance_ohm: float) -> float:
        """
        Power dissipated as heat: P = I²R

        Args:
            current_a: Current in Amps
            resistance_ohm: Resistance in Ohms

        Returns:
            Power in Watts
        """
        return current_a ** 2 * resistance_ohm


# Quick test
if __name__ == "__main__":
    physics = CopperPhysics()

    print("=== Copper Trace Physics ===\n")

    # Test various trace geometries
    test_cases = [
        (10, 0.1, "Thin trace (0.1mm)"),
        (10, 0.5, "Medium trace (0.5mm)"),
        (10, 2.0, "Fat trace (2.0mm)"),
        (50, 0.1, "Long thin trace"),
    ]

    for length, width, description in test_cases:
        R = physics.trace_resistance(length, width)
        I_max = physics.current_capacity(width)
        print(f"{description}:")
        print(f"  {length}mm × {width}mm = {R:.4f}Ω")
        print(f"  Max current: {I_max:.3f}A")
        print()

    # Test torch version
    print("=== Torch Compatibility Test ===")
    length = torch.tensor(10.0, requires_grad=True)
    width = torch.tensor(1.0, requires_grad=True)
    R = physics.trace_resistance_torch(length, width)
    R.backward()
    print(f"R = {R.item():.6f}Ω")
    print(f"dR/d(length) = {length.grad:.6f}")
    print(f"dR/d(width) = {width.grad:.6f}")
