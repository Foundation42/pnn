"""
Crystal Compiler: Compile frozen neural crystals to native C/CUDA code.

The key insight: frozen neurons have FIXED weights that never change.
We can emit them as compile-time constants and generate optimized code
that only dynamically computes the active frontier.

Architecture:
1. Load trained crystal model (.pt file)
2. Extract frozen vs active neurons
3. Generate C code with:
   - Frozen weights as static const arrays
   - Optimized forward pass
   - Optional CUDA kernels for GPU inference
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class CompiledNeuron:
    """A single neuron ready for code generation."""
    index: int
    position: np.ndarray
    input_weights: np.ndarray
    bias: float
    output_weights: np.ndarray  # This neuron's contribution to each output
    is_frozen: bool


@dataclass
class CompiledCrystal:
    """Complete compiled crystal structure."""
    input_dim: int
    output_dim: int
    num_neurons: int
    neurons: List[CompiledNeuron]
    frozen_indices: List[int]
    active_indices: List[int]
    interaction_scale: float
    positions: np.ndarray

    @property
    def num_frozen(self) -> int:
        return len(self.frozen_indices)

    @property
    def num_active(self) -> int:
        return len(self.active_indices)


class CrystalExtractor:
    """Extract compiled structure from a trained PyTorch model."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')

    def extract(self) -> CompiledCrystal:
        """Extract the crystal structure from checkpoint."""
        state_dict = self.checkpoint.get('model_state_dict', self.checkpoint)

        # Extract weights
        input_weights = state_dict['input_weights'].numpy()
        biases = state_dict['biases'].numpy()
        output_weights = state_dict['output_weights'].numpy()
        positions = state_dict['positions'].numpy()

        # Get frozen mask
        frozen_mask = self.checkpoint.get('frozen_mask',
                                          torch.zeros(input_weights.shape[0], dtype=torch.bool))
        if isinstance(frozen_mask, torch.Tensor):
            frozen_mask = frozen_mask.numpy()

        # Extract interaction scale
        interaction_scale = state_dict.get('interaction_scale', torch.tensor(1.0))
        if isinstance(interaction_scale, torch.Tensor):
            interaction_scale = interaction_scale.item()

        num_neurons = input_weights.shape[0]
        input_dim = input_weights.shape[1]
        output_dim = output_weights.shape[0]

        # Build neuron list
        neurons = []
        frozen_indices = []
        active_indices = []

        for i in range(num_neurons):
            neuron = CompiledNeuron(
                index=i,
                position=positions[i],
                input_weights=input_weights[i],
                bias=biases[i],
                output_weights=output_weights[:, i],
                is_frozen=bool(frozen_mask[i])
            )
            neurons.append(neuron)

            if neuron.is_frozen:
                frozen_indices.append(i)
            else:
                active_indices.append(i)

        return CompiledCrystal(
            input_dim=input_dim,
            output_dim=output_dim,
            num_neurons=num_neurons,
            neurons=neurons,
            frozen_indices=frozen_indices,
            active_indices=active_indices,
            interaction_scale=interaction_scale,
            positions=positions,
        )


class CCodeGenerator:
    """Generate C code from compiled crystal."""

    def __init__(self, crystal: CompiledCrystal):
        self.crystal = crystal

    def generate(self, include_interactions: bool = True) -> str:
        """Generate complete C source file."""
        lines = []

        # Header
        lines.append(self._generate_header())
        lines.append("")

        # Constants
        lines.append(self._generate_constants())
        lines.append("")

        # Frozen weights (compile-time constants)
        lines.append(self._generate_frozen_weights())
        lines.append("")

        # Active weights (still constants, but semantically different)
        lines.append(self._generate_active_weights())
        lines.append("")

        # Interaction matrix (precomputed distances)
        if include_interactions:
            lines.append(self._generate_interaction_matrix())
            lines.append("")

        # Forward pass function
        lines.append(self._generate_forward_pass(include_interactions))
        lines.append("")

        # Argmax helper
        lines.append(self._generate_argmax())
        lines.append("")

        # Main inference function
        lines.append(self._generate_inference())

        return "\n".join(lines)

    def _generate_header(self) -> str:
        return '''/*
 * Crystal Neural Network - Compiled from PyTorch
 *
 * This code was automatically generated from a trained crystal model.
 * Frozen neurons: {frozen} ({frozen_pct:.1f}%)
 * Active neurons: {active} ({active_pct:.1f}%)
 *
 * Generated by Crystal Compiler
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

// Activation function
static inline float tanh_activation(float x) {{
    return tanhf(x);
}}
'''.format(
            frozen=self.crystal.num_frozen,
            frozen_pct=100 * self.crystal.num_frozen / self.crystal.num_neurons,
            active=self.crystal.num_active,
            active_pct=100 * self.crystal.num_active / self.crystal.num_neurons,
        )

    def _generate_constants(self) -> str:
        return f'''// Network dimensions
#define INPUT_DIM {self.crystal.input_dim}
#define OUTPUT_DIM {self.crystal.output_dim}
#define NUM_NEURONS {self.crystal.num_neurons}
#define NUM_FROZEN {self.crystal.num_frozen}
#define NUM_ACTIVE {self.crystal.num_active}
#define INTERACTION_SCALE {self.crystal.interaction_scale}f
'''

    def _generate_frozen_weights(self) -> str:
        lines = ["// Frozen neuron weights (compile-time constants)"]

        if self.crystal.num_frozen == 0:
            lines.append("// No frozen neurons")
            return "\n".join(lines)

        # Input weights for frozen neurons
        lines.append(f"static const float FROZEN_INPUT_WEIGHTS[{self.crystal.num_frozen}][INPUT_DIM] = {{")
        for idx in self.crystal.frozen_indices:
            neuron = self.crystal.neurons[idx]
            weights_str = ", ".join(f"{w:.8f}f" for w in neuron.input_weights)
            lines.append(f"    {{ {weights_str} }},")
        lines.append("};")
        lines.append("")

        # Biases for frozen neurons
        lines.append(f"static const float FROZEN_BIASES[{self.crystal.num_frozen}] = {{")
        biases = [self.crystal.neurons[idx].bias for idx in self.crystal.frozen_indices]
        lines.append(f"    {', '.join(f'{b:.8f}f' for b in biases)}")
        lines.append("};")
        lines.append("")

        # Output weights for frozen neurons
        lines.append(f"static const float FROZEN_OUTPUT_WEIGHTS[OUTPUT_DIM][{self.crystal.num_frozen}] = {{")
        for out_idx in range(self.crystal.output_dim):
            weights = [self.crystal.neurons[idx].output_weights[out_idx]
                      for idx in self.crystal.frozen_indices]
            weights_str = ", ".join(f"{w:.8f}f" for w in weights)
            lines.append(f"    {{ {weights_str} }},")
        lines.append("};")

        # Frozen neuron indices (for interaction computation)
        lines.append("")
        lines.append(f"static const int FROZEN_INDICES[{self.crystal.num_frozen}] = {{")
        lines.append(f"    {', '.join(str(i) for i in self.crystal.frozen_indices)}")
        lines.append("};")

        return "\n".join(lines)

    def _generate_active_weights(self) -> str:
        lines = ["// Active neuron weights"]

        if self.crystal.num_active == 0:
            lines.append("// No active neurons (fully crystallized!)")
            return "\n".join(lines)

        # Input weights for active neurons
        lines.append(f"static const float ACTIVE_INPUT_WEIGHTS[{self.crystal.num_active}][INPUT_DIM] = {{")
        for idx in self.crystal.active_indices:
            neuron = self.crystal.neurons[idx]
            weights_str = ", ".join(f"{w:.8f}f" for w in neuron.input_weights)
            lines.append(f"    {{ {weights_str} }},")
        lines.append("};")
        lines.append("")

        # Biases for active neurons
        lines.append(f"static const float ACTIVE_BIASES[{self.crystal.num_active}] = {{")
        biases = [self.crystal.neurons[idx].bias for idx in self.crystal.active_indices]
        lines.append(f"    {', '.join(f'{b:.8f}f' for b in biases)}")
        lines.append("};")
        lines.append("")

        # Output weights for active neurons
        lines.append(f"static const float ACTIVE_OUTPUT_WEIGHTS[OUTPUT_DIM][{self.crystal.num_active}] = {{")
        for out_idx in range(self.crystal.output_dim):
            weights = [self.crystal.neurons[idx].output_weights[out_idx]
                      for idx in self.crystal.active_indices]
            weights_str = ", ".join(f"{w:.8f}f" for w in weights)
            lines.append(f"    {{ {weights_str} }},")
        lines.append("};")

        # Active neuron indices
        lines.append("")
        lines.append(f"static const int ACTIVE_INDICES[{self.crystal.num_active}] = {{")
        lines.append(f"    {', '.join(str(i) for i in self.crystal.active_indices)}")
        lines.append("};")

        return "\n".join(lines)

    def _generate_interaction_matrix(self) -> str:
        lines = ["// Precomputed interaction matrix (1/distance between neurons)"]

        # Compute interaction strengths
        positions = self.crystal.positions
        n = self.crystal.num_neurons

        lines.append(f"static const float INTERACTIONS[{n}][{n}] = {{")
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append("0.0f")
                else:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    interaction = self.crystal.interaction_scale / (dist + 1e-6)
                    row.append(f"{interaction:.6f}f")
            lines.append(f"    {{ {', '.join(row)} }},")
        lines.append("};")

        return "\n".join(lines)

    def _generate_forward_pass(self, include_interactions: bool) -> str:
        lines = ['''// Forward pass through crystal network
void crystal_forward(const float* input, float* output) {
    float activations[NUM_NEURONS];
    float modulated[NUM_NEURONS];

    // Compute frozen neuron activations
    for (int i = 0; i < NUM_FROZEN; i++) {
        float sum = FROZEN_BIASES[i];
        for (int j = 0; j < INPUT_DIM; j++) {
            sum += FROZEN_INPUT_WEIGHTS[i][j] * input[j];
        }
        activations[FROZEN_INDICES[i]] = tanh_activation(sum);
    }

    // Compute active neuron activations
    for (int i = 0; i < NUM_ACTIVE; i++) {
        float sum = ACTIVE_BIASES[i];
        for (int j = 0; j < INPUT_DIM; j++) {
            sum += ACTIVE_INPUT_WEIGHTS[i][j] * input[j];
        }
        activations[ACTIVE_INDICES[i]] = tanh_activation(sum);
    }''']

        if include_interactions:
            lines.append('''
    // Apply neuron interactions (modulation)
    for (int i = 0; i < NUM_NEURONS; i++) {
        float interaction_sum = 0.0f;
        for (int j = 0; j < NUM_NEURONS; j++) {
            interaction_sum += activations[j] * INTERACTIONS[i][j];
        }
        modulated[i] = tanh_activation(activations[i] + 0.1f * interaction_sum);
    }''')
        else:
            lines.append('''
    // No interactions - use activations directly
    memcpy(modulated, activations, sizeof(activations));''')

        lines.append('''
    // Compute output from all neurons
    for (int o = 0; o < OUTPUT_DIM; o++) {
        output[o] = 0.0f;
        for (int i = 0; i < NUM_FROZEN; i++) {
            output[o] += modulated[FROZEN_INDICES[i]] * FROZEN_OUTPUT_WEIGHTS[o][i];
        }
        for (int i = 0; i < NUM_ACTIVE; i++) {
            output[o] += modulated[ACTIVE_INDICES[i]] * ACTIVE_OUTPUT_WEIGHTS[o][i];
        }
    }
}''')

        return "\n".join(lines)

    def _generate_argmax(self) -> str:
        return '''// Argmax helper for classification
int argmax(const float* arr, int n) {
    int best_idx = 0;
    float best_val = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best_idx = i;
        }
    }
    return best_idx;
}'''

    def _generate_inference(self) -> str:
        return '''// Main inference function - returns predicted class
int crystal_predict(const float* input) {
    float output[OUTPUT_DIM];
    crystal_forward(input, output);
    return argmax(output, OUTPUT_DIM);
}'''


class CUDACodeGenerator:
    """Generate CUDA kernel code from compiled crystal."""

    def __init__(self, crystal: CompiledCrystal):
        self.crystal = crystal

    def generate(self) -> str:
        """Generate CUDA kernel for batch inference."""
        lines = []

        # Header
        lines.append(self._generate_header())

        # Device constants
        lines.append(self._generate_device_constants())

        # Kernel
        lines.append(self._generate_kernel())

        # Host wrapper
        lines.append(self._generate_host_wrapper())

        return "\n".join(lines)

    def _generate_header(self) -> str:
        return f'''/*
 * Crystal Neural Network - CUDA Kernel
 * Compiled from PyTorch crystal model
 *
 * Frozen: {self.crystal.num_frozen} neurons
 * Active: {self.crystal.num_active} neurons
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define INPUT_DIM {self.crystal.input_dim}
#define OUTPUT_DIM {self.crystal.output_dim}
#define NUM_NEURONS {self.crystal.num_neurons}
'''

    def _generate_device_constants(self) -> str:
        # For CUDA, we'd put weights in constant memory
        return '''// Weights stored in constant memory for fast access
__constant__ float d_input_weights[NUM_NEURONS * INPUT_DIM];
__constant__ float d_biases[NUM_NEURONS];
__constant__ float d_output_weights[OUTPUT_DIM * NUM_NEURONS];
'''

    def _generate_kernel(self) -> str:
        return '''// CUDA kernel for batch inference
__global__ void crystal_forward_kernel(
    const float* __restrict__ inputs,  // [batch_size, INPUT_DIM]
    float* __restrict__ outputs,        // [batch_size, OUTPUT_DIM]
    int batch_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    const float* input = inputs + batch_idx * INPUT_DIM;
    float* output = outputs + batch_idx * OUTPUT_DIM;

    // Compute neuron activations
    float activations[NUM_NEURONS];
    for (int n = 0; n < NUM_NEURONS; n++) {
        float sum = d_biases[n];
        for (int i = 0; i < INPUT_DIM; i++) {
            sum += d_input_weights[n * INPUT_DIM + i] * input[i];
        }
        activations[n] = tanhf(sum);
    }

    // Compute outputs
    for (int o = 0; o < OUTPUT_DIM; o++) {
        float sum = 0.0f;
        for (int n = 0; n < NUM_NEURONS; n++) {
            sum += activations[n] * d_output_weights[o * NUM_NEURONS + n];
        }
        output[o] = sum;
    }
}
'''

    def _generate_host_wrapper(self) -> str:
        return '''// Host function to run inference
void crystal_inference_cuda(
    const float* h_inputs,
    float* h_outputs,
    int batch_size
) {
    float *d_inputs, *d_outputs;

    cudaMalloc(&d_inputs, batch_size * INPUT_DIM * sizeof(float));
    cudaMalloc(&d_outputs, batch_size * OUTPUT_DIM * sizeof(float));

    cudaMemcpy(d_inputs, h_inputs, batch_size * INPUT_DIM * sizeof(float),
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    crystal_forward_kernel<<<blocks, threads>>>(d_inputs, d_outputs, batch_size);

    cudaMemcpy(h_outputs, d_outputs, batch_size * OUTPUT_DIM * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_inputs);
    cudaFree(d_outputs);
}
'''


def compile_crystal(checkpoint_path: str, output_dir: str,
                    generate_cuda: bool = True) -> Dict[str, str]:
    """
    Main compilation function.

    Args:
        checkpoint_path: Path to trained crystal model (.pt)
        output_dir: Directory to write generated code
        generate_cuda: Whether to also generate CUDA code

    Returns:
        Dict with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract crystal structure
    print(f"Loading crystal from {checkpoint_path}...")
    extractor = CrystalExtractor(checkpoint_path)
    crystal = extractor.extract()

    print(f"  Neurons: {crystal.num_neurons}")
    print(f"  Frozen: {crystal.num_frozen} ({100*crystal.num_frozen/crystal.num_neurons:.1f}%)")
    print(f"  Active: {crystal.num_active} ({100*crystal.num_active/crystal.num_neurons:.1f}%)")
    print(f"  Input dim: {crystal.input_dim}")
    print(f"  Output dim: {crystal.output_dim}")

    generated_files = {}

    # Generate C code
    print("\nGenerating C code...")
    c_generator = CCodeGenerator(crystal)
    c_code = c_generator.generate()

    c_path = output_dir / "crystal_net.c"
    c_path.write_text(c_code)
    generated_files['c'] = str(c_path)
    print(f"  Written: {c_path}")

    # Generate header
    header = f'''#ifndef CRYSTAL_NET_H
#define CRYSTAL_NET_H

#define INPUT_DIM {crystal.input_dim}
#define OUTPUT_DIM {crystal.output_dim}
#define NUM_NEURONS {crystal.num_neurons}

void crystal_forward(const float* input, float* output);
int crystal_predict(const float* input);

#endif
'''
    header_path = output_dir / "crystal_net.h"
    header_path.write_text(header)
    generated_files['header'] = str(header_path)
    print(f"  Written: {header_path}")

    # Generate CUDA code
    if generate_cuda:
        print("\nGenerating CUDA code...")
        cuda_generator = CUDACodeGenerator(crystal)
        cuda_code = cuda_generator.generate()

        cuda_path = output_dir / "crystal_net.cu"
        cuda_path.write_text(cuda_code)
        generated_files['cuda'] = str(cuda_path)
        print(f"  Written: {cuda_path}")

    # Generate metadata
    metadata = {
        'num_neurons': crystal.num_neurons,
        'num_frozen': crystal.num_frozen,
        'num_active': crystal.num_active,
        'input_dim': crystal.input_dim,
        'output_dim': crystal.output_dim,
        'frozen_ratio': crystal.num_frozen / crystal.num_neurons,
    }

    meta_path = output_dir / "crystal_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    generated_files['metadata'] = str(meta_path)
    print(f"  Written: {meta_path}")

    print("\n✅ Compilation complete!")
    return generated_files


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python compiler.py <checkpoint.pt> [output_dir]")
        print("\nExample:")
        print("  python compiler.py runs/growing_crystal_*/model_final.pt ./compiled")
        sys.exit(1)

    checkpoint = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./compiled"

    compile_crystal(checkpoint, output_dir)
