#!/usr/bin/env python3
"""
Crystal Compiler - Convert PyTorch models to .crystal format

Compresses crystallized neural networks using:
1. int8 quantization (4x reduction)
2. Pattern deduplication via hash table (10-50x on frozen weights)
3. Vocabulary pruning (remove unused tokens)

Usage:
    python crystal_compiler.py model.pt output.crystal [--prune-vocab text.txt]
"""

import torch
import numpy as np
import struct
import sys
import os
from collections import defaultdict
import tiktoken

# Crystal format constants
CRYSTAL_MAGIC = 0x53595243  # "CRYS"
CRYSTAL_VERSION = 1

# Flags
FLAG_HAS_VOCAB = 0x0001
FLAG_DELTA_ENCODED = 0x0002
FLAG_PRUNED_VOCAB = 0x0004
FLAG_FLOAT32 = 0x0008  # Weights stored as float32 (no quantization)
FLAG_INT16 = 0x0010    # Weights stored as int16 (2x size of int8)
FLAG_MIXED = 0x0020    # Mixed precision: f32 embeddings + int16 rest
FLAG_MIXED8 = 0x0040   # Mixed precision: f32 embeddings + int8 rest


def quantize_array(arr, bits=8):
    """Quantize float array to int8 with scale/offset."""
    arr = arr.astype(np.float32)

    # Find min/max
    vmin, vmax = arr.min(), arr.max()

    # Handle edge case of constant array
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.int8), 1.0, vmin

    # Scale to [-127, 127]
    scale = (vmax - vmin) / 254.0
    offset = vmin + 127 * scale

    quantized = np.clip(np.round((arr - offset) / scale), -127, 127).astype(np.int8)

    return quantized, scale, offset


def quantize_array_int16(arr):
    """Quantize float array to int16 with scale/offset."""
    arr = arr.astype(np.float32)

    # Find min/max
    vmin, vmax = arr.min(), arr.max()

    # Handle edge case of constant array
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.int16), 1.0, vmin

    # Scale to [-32767, 32767] (leave room for -32768)
    scale = (vmax - vmin) / 65534.0
    offset = vmin + 32767 * scale

    quantized = np.clip(np.round((arr - offset) / scale), -32767, 32767).astype(np.int16)

    return quantized, scale, offset


def find_unique_patterns(weights, tolerance=0):
    """
    Find unique weight patterns using a hash table approach.
    Returns: (unique_patterns, pattern_indices)
    """
    patterns = {}
    indices = []
    unique = []

    for i, w in enumerate(weights):
        # Convert to bytes for hashing
        key = w.tobytes()

        if key not in patterns:
            patterns[key] = len(unique)
            unique.append(w.copy())

        indices.append(patterns[key])

    return np.array(unique), np.array(indices, dtype=np.uint16)


def prune_vocabulary(tokenizer, text, min_count=1):
    """
    Find which tokens are actually used in the text.
    Returns: (used_token_ids, id_mapping)
    """
    tokens = tokenizer.encode(text)
    token_counts = defaultdict(int)

    for t in tokens:
        token_counts[t] += 1

    # Filter by min_count
    used = sorted([t for t, c in token_counts.items() if c >= min_count])

    # Create mapping: old_id -> new_id
    old_to_new = {old: new for new, old in enumerate(used)}

    return used, old_to_new


class CrystalCompiler:
    def __init__(self, model_path, output_path, prune_vocab=False, text_path=None,
                 no_quantize=False, use_int16=False, use_mixed=False, use_mixed8=False):
        self.model_path = model_path
        self.output_path = output_path
        self.prune_vocab = prune_vocab
        self.text_path = text_path
        self.no_quantize = no_quantize
        self.use_int16 = use_int16
        self.use_mixed = use_mixed    # f32 embeddings + int16 head/neurons
        self.use_mixed8 = use_mixed8  # f32 embeddings + int8 head/neurons

        # Load tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def load_model(self):
        """Load the PyTorch model checkpoint."""
        print(f"Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        if 'model' in checkpoint:
            self.state_dict = checkpoint['model']
            self.config = checkpoint.get('config', {})
        else:
            self.state_dict = checkpoint
            self.config = {}

        # Extract dimensions from state dict
        for key in self.state_dict:
            if 'token_embed' in key:
                shape = self.state_dict[key].shape
                self.vocab_size = shape[0]
                self.embed_dim = shape[1]
                break

        for key in self.state_dict:
            if 'attention.positions' in key:
                self.num_neurons = self.state_dict[key].shape[0]
                break

        for key in self.state_dict:
            if 'attention.frozen' in key:
                self.frozen_mask = self.state_dict[key].numpy()
                self.num_frozen = int(self.frozen_mask.sum())
                self.num_active = self.num_neurons - self.num_frozen
                break

        for key in self.state_dict:
            if 'pos_embed' in key:
                self.context_len = self.state_dict[key].shape[0]
                break

        self.hidden_dim = self.embed_dim * 2
        for key in self.state_dict:
            if 'ffn.0.weight' in key:
                self.hidden_dim = self.state_dict[key].shape[0]
                break

        print(f"  Vocab: {self.vocab_size}, Embed: {self.embed_dim}, Context: {self.context_len}")
        print(f"  Neurons: {self.num_neurons} ({self.num_frozen} frozen, {self.num_active} active)")

    def extract_weights(self):
        """Extract and organize weights from state dict."""
        self.weights = {}
        for key, tensor in self.state_dict.items():
            self.weights[key] = tensor.numpy()

    def quantize_all(self):
        """Quantize all weights to int8/int16 (or keep float32 if no_quantize)."""
        if self.no_quantize:
            print("Keeping weights as float32 (no quantization)...")
        elif self.use_mixed:
            print("Mixed precision: f32 embeddings + int16 rest...")
        elif self.use_mixed8:
            print("Mixed precision: f32 embeddings + int8 rest...")
        elif self.use_int16:
            print("Quantizing weights to int16...")
        else:
            print("Quantizing weights to int8...")

        self.quantized = {}
        self.scales = {}
        self.offsets_map = {}

        # Keys that should stay float32 in mixed modes
        float32_keys = {'token_embed.weight', 'pos_embed.weight'} if (self.use_mixed or self.use_mixed8) else set()

        for key, arr in self.weights.items():
            if arr.dtype == np.bool_:
                self.quantized[key] = arr
                continue

            if self.no_quantize or key in float32_keys:
                # Keep as float32
                self.quantized[key] = arr.astype(np.float32)
                self.scales[key] = 1.0
                self.offsets_map[key] = 0.0
                print(f"  {key}: {arr.nbytes:,} bytes (float32)")
            elif self.use_int16 or self.use_mixed:
                # Quantize to int16
                q, s, o = quantize_array_int16(arr)
                self.quantized[key] = q
                self.scales[key] = s
                self.offsets_map[key] = o

                orig_size = arr.nbytes
                quant_size = q.nbytes
                print(f"  {key}: {orig_size:,} → {quant_size:,} bytes ({orig_size/quant_size:.1f}x)")
            else:
                # Quantize to int8 (default or mixed8)
                q, s, o = quantize_array(arr)
                self.quantized[key] = q
                self.scales[key] = s
                self.offsets_map[key] = o

                orig_size = arr.nbytes
                quant_size = q.nbytes
                print(f"  {key}: {orig_size:,} → {quant_size:,} bytes ({orig_size/quant_size:.1f}x)")

    def deduplicate_frozen(self):
        """Find unique patterns among frozen neuron weights."""
        print("Deduplicating frozen weight patterns...")

        positions = self.quantized['attention.positions']
        values = self.quantized['attention.values']

        frozen_positions = positions[self.frozen_mask]
        frozen_values = values[self.frozen_mask]

        # Concatenate position + value for each neuron
        frozen_combined = np.concatenate([frozen_positions, frozen_values], axis=1)

        unique_patterns, pattern_indices = find_unique_patterns(frozen_combined)

        self.unique_patterns = unique_patterns
        self.pattern_indices = pattern_indices
        self.num_patterns = len(unique_patterns)
        self.pattern_dim = frozen_combined.shape[1]

        orig_size = frozen_combined.nbytes
        pattern_size = unique_patterns.nbytes
        index_size = pattern_indices.nbytes
        new_size = pattern_size + index_size

        print(f"  Found {self.num_patterns} unique patterns from {self.num_frozen} frozen neurons")
        print(f"  Original: {orig_size:,} bytes")
        print(f"  Compressed: {new_size:,} bytes ({orig_size/new_size:.1f}x)")

    def build_vocabulary(self):
        """Build vocabulary section."""
        if self.prune_vocab and self.text_path:
            print("Pruning vocabulary...")
            with open(self.text_path, 'r') as f:
                text = f.read()

            used_tokens, old_to_new = prune_vocabulary(self.tokenizer, text)
            self.used_tokens = used_tokens
            self.token_mapping = old_to_new
            self.pruned_vocab_size = len(used_tokens)

            print(f"  Reduced vocab: {self.vocab_size} → {self.pruned_vocab_size} ({100*self.pruned_vocab_size/self.vocab_size:.1f}%)")
        else:
            self.used_tokens = list(range(self.vocab_size))
            self.token_mapping = {i: i for i in range(self.vocab_size)}
            self.pruned_vocab_size = self.vocab_size

    def write_crystal(self):
        """Write the .crystal file."""
        print(f"Writing {self.output_path}...")

        # Prepare all data sections first
        D = self.embed_dim
        T = self.context_len
        H = self.hidden_dim
        V = self.pruned_vocab_size
        N = self.num_neurons

        # Quant params (store scales for each major tensor type)
        embed_scale = self.scales.get('token_embed.weight', 1.0)
        embed_offset = self.offsets_map.get('token_embed.weight', 0.0)
        weight_scale = self.scales.get('attention.positions', 1.0)
        weight_offset = self.offsets_map.get('attention.positions', 0.0)
        head_scale = self.scales.get('head.weight', 1.0)
        head_offset = self.offsets_map.get('head.weight', 0.0)
        head_bias_scale = self.scales.get('head.bias', 1.0)
        head_bias_offset = self.offsets_map.get('head.bias', 0.0)

        quant_data = struct.pack('<ffffffff',
            embed_scale, embed_offset,
            weight_scale, weight_offset,
            head_scale, head_offset,
            head_bias_scale, head_bias_offset)

        # Pattern table
        pattern_count = struct.pack('<I', self.num_patterns)
        pattern_data = self.unique_patterns.tobytes()

        # Mapping table
        mapping_data = self.pattern_indices.tobytes()

        # Active weights
        active_positions = self.quantized['attention.positions'][~self.frozen_mask]
        active_values = self.quantized['attention.values'][~self.frozen_mask]
        active_combined = np.concatenate([active_positions, active_values], axis=1)
        active_count = struct.pack('<I', self.num_active)
        active_data = active_combined.tobytes()

        # Token embeddings (pruned)
        token_embed = self.quantized['token_embed.weight']
        if self.prune_vocab:
            token_embed = token_embed[self.used_tokens]
        embed_data = token_embed.tobytes()

        # Position embeddings
        pos_embed = self.quantized['pos_embed.weight']
        pos_data = pos_embed.tobytes()

        # Temperatures
        temps = self.quantized['attention.temperature']
        temp_data = temps.tobytes()

        # FFN weights - keep as float32 for accuracy (small)
        ffn_w1 = self.weights['ffn.0.weight'].astype(np.float32)
        ffn_b1 = self.weights['ffn.0.bias'].astype(np.float32)
        ffn_w2 = self.weights['ffn.2.weight'].astype(np.float32)
        ffn_b2 = self.weights['ffn.2.bias'].astype(np.float32)
        ffn_data = ffn_w1.tobytes() + ffn_b1.tobytes() + ffn_w2.tobytes() + ffn_b2.tobytes()

        # Output head
        head_w = self.quantized['head.weight']
        head_b = self.quantized['head.bias']
        if self.prune_vocab:
            head_w = head_w[self.used_tokens]
            head_b = head_b[self.used_tokens]
        head_data = head_w.tobytes() + head_b.tobytes()

        # Layer norms (float32)
        norm1_w = self.weights['norm1.weight']
        norm1_b = self.weights['norm1.bias']
        norm2_w = self.weights['norm2.weight']
        norm2_b = self.weights['norm2.bias']
        norm_data = norm1_w.tobytes() + norm1_b.tobytes() + norm2_w.tobytes() + norm2_b.tobytes()

        # Calculate offsets
        header_size = 64
        quant_offset = header_size
        pattern_offset = quant_offset + len(quant_data)
        mapping_offset = pattern_offset + len(pattern_count) + len(pattern_data)
        active_offset = mapping_offset + len(mapping_data)
        embed_offset_val = active_offset + len(active_count) + len(active_data)
        pos_offset = embed_offset_val + len(embed_data)
        temp_offset = pos_offset + len(pos_data)
        ffn_offset = temp_offset + len(temp_data)
        head_offset = ffn_offset + len(ffn_data)
        norm_offset = head_offset + len(head_data)
        total_size = norm_offset + len(norm_data)

        # Build header (64 bytes exactly)
        flags = 0
        if self.prune_vocab:
            flags |= FLAG_PRUNED_VOCAB
        if self.no_quantize:
            flags |= FLAG_FLOAT32
        if self.use_int16:
            flags |= FLAG_INT16
        if self.use_mixed:
            flags |= FLAG_MIXED | FLAG_INT16  # f32 embeddings + int16 rest
        if self.use_mixed8:
            flags |= FLAG_MIXED8  # f32 embeddings + int8 rest

        header = struct.pack('<I', CRYSTAL_MAGIC)          # 4: magic
        header += struct.pack('<H', CRYSTAL_VERSION)       # 2: version
        header += struct.pack('<H', flags)                 # 2: flags
        header += struct.pack('<I', V)                     # 4: vocab_size
        header += struct.pack('<H', D)                     # 2: embed_dim
        header += struct.pack('<H', T)                     # 2: context_len
        header += struct.pack('<H', H)                     # 2: hidden_dim
        header += struct.pack('<H', N)                     # 2: num_neurons
        header += struct.pack('<H', self.num_frozen)       # 2: num_frozen
        header += struct.pack('<H', self.num_active)       # 2: num_active
        header += struct.pack('<H', self.num_patterns)     # 2: num_patterns
        header += struct.pack('<H', self.pattern_dim)      # 2: pattern_dim
        header += struct.pack('<H', 0)                     # 2: reserved1
        header += struct.pack('<H', 0)                     # 2: reserved2
        # Offsets (8 x 4 = 32 bytes)
        header += struct.pack('<I', quant_offset)
        header += struct.pack('<I', pattern_offset)
        header += struct.pack('<I', mapping_offset)
        header += struct.pack('<I', active_offset)
        header += struct.pack('<I', embed_offset_val)
        header += struct.pack('<I', pos_offset)
        header += struct.pack('<I', ffn_offset)
        header += struct.pack('<I', head_offset)

        # Should be exactly 64 bytes now
        assert len(header) == 64, f"Header is {len(header)} bytes, expected 64"

        # Write file
        with open(self.output_path, 'wb') as f:
            f.write(header)
            f.write(quant_data)
            f.write(pattern_count)
            f.write(pattern_data)
            f.write(mapping_data)
            f.write(active_count)
            f.write(active_data)
            f.write(embed_data)
            f.write(pos_data)
            f.write(temp_data)
            f.write(ffn_data)
            f.write(head_data)
            f.write(norm_data)

            actual_size = f.tell()

        print(f"\nFinal .crystal size: {actual_size:,} bytes ({actual_size/1024/1024:.2f} MB)")

        orig_size = os.path.getsize(self.model_path)
        print(f"Original model: {orig_size:,} bytes ({orig_size/1024/1024:.2f} MB)")
        print(f"Compression ratio: {orig_size/actual_size:.1f}x")

        # Debug: print offsets
        print(f"\nOffsets: quant={quant_offset}, pattern={pattern_offset}, mapping={mapping_offset}")
        print(f"         active={active_offset}, embed={embed_offset_val}, pos={pos_offset}")
        print(f"         temp={temp_offset}, ffn={ffn_offset}, head={head_offset}, norm={norm_offset}")

    def compile(self):
        """Run the full compilation pipeline."""
        print("=" * 60)
        print("Crystal Compiler v1.0")
        print("=" * 60)

        self.load_model()
        self.extract_weights()
        self.quantize_all()
        self.deduplicate_frozen()
        self.build_vocabulary()
        self.write_crystal()

        print("\n" + "=" * 60)
        print("Compilation complete!")
        print("=" * 60)


def main():
    if len(sys.argv) < 3:
        print("Usage: python crystal_compiler.py model.pt output.crystal [--prune-vocab text.txt] [--no-quantize] [--int16] [--mixed] [--mixed8]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    prune_vocab = False
    text_path = None
    no_quantize = '--no-quantize' in sys.argv
    use_int16 = '--int16' in sys.argv
    use_mixed = '--mixed' in sys.argv
    use_mixed8 = '--mixed8' in sys.argv

    if '--prune-vocab' in sys.argv:
        prune_vocab = True
        idx = sys.argv.index('--prune-vocab')
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith('--'):
            text_path = sys.argv[idx + 1]

    compiler = CrystalCompiler(model_path, output_path, prune_vocab, text_path, no_quantize, use_int16, use_mixed, use_mixed8)
    compiler.compile()


if __name__ == '__main__':
    main()
