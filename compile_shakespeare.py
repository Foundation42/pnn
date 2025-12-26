"""
Compile Crystal Shakespeare to C code
Extracts frozen neurons as compile-time constants
"""

import torch
import numpy as np
from pathlib import Path


def compile_to_c(checkpoint_path: str, output_dir: str):
    """Compile Shakespeare crystal to C code."""

    print("=" * 60)
    print("Crystal Shakespeare Compiler")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    config = checkpoint['config']
    model = checkpoint['model']

    print(f"\nModel config:")
    print(f"  Vocab size: {config['vocab_size']:,}")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Neurons: {config['num_neurons']}")

    # Extract key tensors
    positions = model['attention.positions'].numpy()
    values = model['attention.values'].numpy()
    temperatures = model['attention.temperature'].numpy()
    frozen = model['attention.frozen'].numpy()

    num_neurons = positions.shape[0]
    embed_dim = positions.shape[1]
    num_frozen = frozen.sum()
    num_active = num_neurons - num_frozen

    print(f"\nCrystal structure:")
    print(f"  Total neurons: {num_neurons}")
    print(f"  Frozen: {num_frozen} ({100*num_frozen/num_neurons:.1f}%)")
    print(f"  Active: {num_active}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate C header
    header = f"""/* Crystal Shakespeare - Compiled Geometric Attention
 * Auto-generated from trained PyTorch model
 *
 * Neurons: {num_neurons} total, {num_frozen} frozen, {num_active} active
 * Embed dim: {embed_dim}
 * Speedup: {num_neurons / max(num_active, 1):.1f}x (frozen weights are compile-time constants)
 */

#ifndef CRYSTAL_SHAKESPEARE_H
#define CRYSTAL_SHAKESPEARE_H

#include <math.h>

#define CRYSTAL_NUM_NEURONS {num_neurons}
#define CRYSTAL_NUM_FROZEN {num_frozen}
#define CRYSTAL_NUM_ACTIVE {num_active}
#define CRYSTAL_EMBED_DIM {embed_dim}

/* Frozen neuron indices (these use precomputed weights) */
static const int frozen_indices[CRYSTAL_NUM_FROZEN] = {{
    {', '.join(str(i) for i in np.where(frozen)[0])}
}};

/* Active neuron indices (these need dynamic computation) */
static const int active_indices[CRYSTAL_NUM_ACTIVE] = {{
    {', '.join(str(i) for i in np.where(~frozen)[0])}
}};

#endif /* CRYSTAL_SHAKESPEARE_H */
"""

    # Generate C source with frozen weights
    source = f"""/* Crystal Shakespeare - Frozen Weights
 * {num_frozen} frozen neurons as compile-time constants
 */

#include "crystal_shakespeare.h"
#include <stdio.h>
#include <string.h>

/* Neuron positions in embedding space */
static const float positions[CRYSTAL_NUM_NEURONS][CRYSTAL_EMBED_DIM] = {{
"""

    for i in range(num_neurons):
        pos_str = ', '.join(f'{v:.6f}f' for v in positions[i])
        source += f"    {{ {pos_str} }}"
        source += ',\n' if i < num_neurons - 1 else '\n'
    source += "};\n\n"

    source += f"""/* Neuron values */
static const float values[CRYSTAL_NUM_NEURONS][CRYSTAL_EMBED_DIM] = {{
"""

    for i in range(num_neurons):
        val_str = ', '.join(f'{v:.6f}f' for v in values[i])
        source += f"    {{ {val_str} }}"
        source += ',\n' if i < num_neurons - 1 else '\n'
    source += "};\n\n"

    source += f"""/* RBF temperatures */
static const float temperatures[CRYSTAL_NUM_NEURONS] = {{
    {', '.join(f'{t:.6f}f' for t in temperatures)}
}};

/* Frozen mask (1 = frozen, 0 = active) */
static const int frozen_mask[CRYSTAL_NUM_NEURONS] = {{
    {', '.join(str(int(f)) for f in frozen)}
}};

/* Compute RBF distance between point and neuron */
static inline float rbf_distance(const float* point, int neuron_idx) {{
    float dist_sq = 0.0f;
    for (int d = 0; d < CRYSTAL_EMBED_DIM; d++) {{
        float diff = point[d] - positions[neuron_idx][d];
        dist_sq += diff * diff;
    }}
    return sqrtf(dist_sq);
}}

/* Compute geometric attention for a single token embedding */
void crystal_attention(const float* input, float* output) {{
    float weights[CRYSTAL_NUM_NEURONS];
    float weight_sum = 0.0f;

    /* Compute RBF weights to all neurons */
    for (int n = 0; n < CRYSTAL_NUM_NEURONS; n++) {{
        float dist = rbf_distance(input, n);
        float temp = fabsf(temperatures[n]) + 0.1f;
        weights[n] = expf(-dist / temp);
        weight_sum += weights[n];
    }}

    /* Normalize weights */
    for (int n = 0; n < CRYSTAL_NUM_NEURONS; n++) {{
        weights[n] /= (weight_sum + 1e-8f);
    }}

    /* Compute weighted sum of values */
    memset(output, 0, CRYSTAL_EMBED_DIM * sizeof(float));
    for (int n = 0; n < CRYSTAL_NUM_NEURONS; n++) {{
        for (int d = 0; d < CRYSTAL_EMBED_DIM; d++) {{
            output[d] += weights[n] * values[n][d];
        }}
    }}
}}

/* Optimized version using only frozen neurons (for inference speedup) */
void crystal_attention_frozen_only(const float* input, float* output) {{
    float weights[CRYSTAL_NUM_FROZEN];
    float weight_sum = 0.0f;

    /* Compute RBF weights to frozen neurons only */
    for (int i = 0; i < CRYSTAL_NUM_FROZEN; i++) {{
        int n = frozen_indices[i];
        float dist = rbf_distance(input, n);
        float temp = fabsf(temperatures[n]) + 0.1f;
        weights[i] = expf(-dist / temp);
        weight_sum += weights[i];
    }}

    /* Normalize */
    for (int i = 0; i < CRYSTAL_NUM_FROZEN; i++) {{
        weights[i] /= (weight_sum + 1e-8f);
    }}

    /* Compute weighted sum */
    memset(output, 0, CRYSTAL_EMBED_DIM * sizeof(float));
    for (int i = 0; i < CRYSTAL_NUM_FROZEN; i++) {{
        int n = frozen_indices[i];
        for (int d = 0; d < CRYSTAL_EMBED_DIM; d++) {{
            output[d] += weights[i] * values[n][d];
        }}
    }}
}}

/* Print crystal info */
void crystal_info() {{
    printf("Crystal Shakespeare - Compiled Geometric Attention\\n");
    printf("  Total neurons: %d\\n", CRYSTAL_NUM_NEURONS);
    printf("  Frozen: %d (%.1f%%)\\n", CRYSTAL_NUM_FROZEN,
           100.0f * CRYSTAL_NUM_FROZEN / CRYSTAL_NUM_NEURONS);
    printf("  Active: %d\\n", CRYSTAL_NUM_ACTIVE);
    printf("  Embed dim: %d\\n", CRYSTAL_EMBED_DIM);
    printf("  Speedup: %.1fx\\n",
           (float)CRYSTAL_NUM_NEURONS / CRYSTAL_NUM_ACTIVE);
}}

#ifdef CRYSTAL_MAIN
int main() {{
    crystal_info();

    /* Test with random input */
    float input[CRYSTAL_EMBED_DIM];
    float output[CRYSTAL_EMBED_DIM];

    for (int d = 0; d < CRYSTAL_EMBED_DIM; d++) {{
        input[d] = (float)d / CRYSTAL_EMBED_DIM;
    }}

    crystal_attention(input, output);

    printf("\\nTest forward pass:\\n");
    printf("  Input[0:4]: %.4f, %.4f, %.4f, %.4f\\n",
           input[0], input[1], input[2], input[3]);
    printf("  Output[0:4]: %.4f, %.4f, %.4f, %.4f\\n",
           output[0], output[1], output[2], output[3]);

    return 0;
}}
#endif
"""

    # Write files
    header_path = Path(output_dir) / 'crystal_shakespeare.h'
    source_path = Path(output_dir) / 'crystal_shakespeare.c'

    with open(header_path, 'w') as f:
        f.write(header)
    print(f"\nWrote: {header_path}")

    with open(source_path, 'w') as f:
        f.write(source)
    print(f"Wrote: {source_path}")

    # Calculate sizes
    header_size = header_path.stat().st_size
    source_size = source_path.stat().st_size

    print(f"\nFile sizes:")
    print(f"  Header: {header_size:,} bytes")
    print(f"  Source: {source_size:,} bytes ({source_size/1024/1024:.1f} MB)")

    print(f"\nTo compile:")
    print(f"  gcc -O3 -DCRYSTAL_MAIN -lm {source_path} -o crystal_shakespeare")

    return header_path, source_path


if __name__ == "__main__":
    compile_to_c(
        'runs/crystal_shakespeare_20251226_215547/best_model.pt',
        'runs/crystal_shakespeare_compiled'
    )
