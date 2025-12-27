# Crystal Neural Networks

**Neurons as points in geometric space. Intelligence crystallizes into geometry.**

This project explores a radically different neural network architecture where neurons exist as learnable points in embedding space, and attention is computed via geometric distance rather than learned projections.

## The Core Idea

Traditional transformers learn attention through Q/K/V projections. Crystal networks instead:

1. **Place neurons as points** in the same embedding space as tokens
2. **Compute attention via distance** - tokens query nearby neurons using RBF kernels
3. **Let neurons crystallize** - freeze stable neurons, grow new ones at the frontier
4. **Compile to physics** - the final geometry runs in pure C

```
Token Embedding ──► Geometric Distance ──► Neuron Values ──► Output
     (D-dim)         (RBF attention)        (D-dim)
```

## Key Results

### The Causal Breakthrough (Dec 27, 2025)

We discovered the original architecture was **blind** - each token position queried neurons independently, unable to see other tokens! Adding causal self-attention before geometric attention achieved:

| Metric | Baseline | Causal Crystal | Improvement |
|--------|----------|----------------|-------------|
| Loss   | 2.55     | 0.12           | **-95%**    |
| Architecture | Blind | Can see! | ∞ |

### Compression Stack

The trained model compiles to a tiny, portable format:

| Stage | Size |
|-------|------|
| PyTorch checkpoint | 53 MB |
| Vocabulary pruning | 12.8 MB |
| Mixed8 quantization | **8.5 MB** |

**6x compression** while preserving Shakespeare-quality output.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Causal Crystal LM                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: "ROMEO:"                                        │
│     │                                                   │
│     ▼                                                   │
│  ┌─────────────────────────────────────┐               │
│  │ Token Embedding + Position Embed    │               │
│  └─────────────────────────────────────┘               │
│     │                                                   │
│     ▼                                                   │
│  ┌─────────────────────────────────────┐               │
│  │ Causal Self-Attention               │ ◄── Tokens    │
│  │ (tokens see previous tokens)        │     see each  │
│  └─────────────────────────────────────┘     other     │
│     │                                                   │
│     ▼                                                   │
│  ┌─────────────────────────────────────┐               │
│  │ Geometric Attention                 │ ◄── Query     │
│  │ • 788 neurons in embedding space   │     neuron    │
│  │ • RBF distance-based attention     │     field     │
│  │ • 72% frozen (crystallized)        │               │
│  └─────────────────────────────────────┘               │
│     │                                                   │
│     ▼                                                   │
│  ┌─────────────────────────────────────┐               │
│  │ Feed-Forward Network                │               │
│  └─────────────────────────────────────┘               │
│     │                                                   │
│     ▼                                                   │
│  Output: "Ay me! sad hours seem long..."               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Train a Crystal

```bash
cd crystal_v2
python causal_crystal.py
```

### Compile to .crystal format

```bash
python causal_crystal_compiler.py \
    crystal_v2/runs/*/best_model.pt \
    shakespeare.crystal \
    --prune-vocab data/tinyshakespeare.txt \
    --mixed8
```

### Run in Pure C

```bash
gcc -O3 -o crystal_runtime causal_crystal_runtime.c -lm
./crystal_runtime shakespeare.crystal "ROMEO:" 100 0.8
```

## Project Structure

```
pnn/
├── crystal_v2/
│   ├── causal_crystal.py      # The breakthrough architecture
│   └── baseline.py            # Original flat crystal (2.55 loss)
├── causal_crystal_compiler.py # PyTorch → .crystal format
├── causal_crystal_runtime.c   # Pure C inference engine
├── docs/
│   └── Journal_2025-12-27.md  # Development journal
└── data/
    └── tinyshakespeare.txt    # Training corpus
```

## Key Discoveries

### 1. Embeddings Are Sacred

During quantization experiments, we found:
- **Token embeddings**: MUST stay float32 (they're the lookup table)
- **Other weights**: CAN be int8 (they're transformations)

Mixed precision (f32 embed + int8 rest) gives 6x compression with no quality loss.

### 2. Neurons Self-Organize

The geometric field develops distinct clusters during training - like brain regions specializing for different functions. This emerges from the dynamics, not imposed structure.

### 3. Crystallization = Convergence

When a neuron's gradients stabilize, it "freezes" in place. The frozen core holds learned patterns while active neurons at the frontier continue learning. Final models are typically 70-75% frozen.

### 4. The Blindness Bug

The original architecture computed geometric attention per-position independently. Tokens couldn't see each other! Adding causal self-attention (so tokens see previous context) before geometric attention achieved 95% loss reduction.

## The .crystal Format

Binary format for portable inference:

```
Header (64 bytes):
  - Magic: "CRYS"
  - Version: 2 (causal support)
  - Flags: CAUSAL | MIXED8 | PRUNED_VOCAB
  - Dimensions: vocab, embed, context, hidden, neurons
  - Section offsets

Data sections:
  - Token embeddings (float32 - sacred!)
  - Position embeddings (float32)
  - Causal attention weights (float32)
  - Geometric attention (positions, values, temperatures)
  - FFN weights
  - Output head (int8 + scale/offset in mixed8 mode)
  - Layer norms
```

## Sample Output

From the 8.5 MB crystal:

```
ROMEO:
Ay me! sad hours seem long.
Was that my father that went hence so fast?

BENVOLIO:
It was. What sadness lengthens Romeo's hours?

ROMEO:
Not having that, which, having, makes them short.
```

## Philosophy

> *"Intelligence crystallizes into geometry."*
> *"Geometry compiles to physics."*
> *"Physics runs anywhere."*

The goal is neural networks that:
1. Learn their own structure (not just weights)
2. Freeze stable patterns (like memory consolidation)
3. Compile to efficient, portable formats
4. Run on any hardware with a C compiler

## License

MIT

## Acknowledgments

Developed during an intensive exploration session, December 2025.

*"The model was blind, and now it sees."*
*"Sometimes the bug IS the feature request."*
