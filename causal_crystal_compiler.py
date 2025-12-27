#!/usr/bin/env python3
"""
Causal Crystal Compiler - Convert PyTorch causal crystal models to .crystal format

Supports the new architecture with:
- Causal self-attention (Q, K, V projections)
- Geometric attention (neurons in embedding space)
- Three layer norms

Usage:
    python causal_crystal_compiler.py model.pt output.crystal [--prune-vocab text.txt]
"""

import torch
import numpy as np
import struct
import sys
import os
from collections import defaultdict
import tiktoken

# Crystal format constants - Version 2 for causal
CRYSTAL_MAGIC = 0x53595243  # "CRYS"
CRYSTAL_VERSION = 2  # Version 2 = causal support

# Flags
FLAG_PRUNED_VOCAB = 0x0004
FLAG_FLOAT32 = 0x0008
FLAG_MIXED8 = 0x0040
FLAG_CAUSAL = 0x0080  # New flag for causal attention


def quantize_array(arr, bits=8):
    """Quantize float array to int8 with scale/offset."""
    arr = arr.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(arr, dtype=np.int8), 1.0, vmin
    scale = (vmax - vmin) / 254.0
    offset = vmin + 127 * scale
    quantized = np.clip(np.round((arr - offset) / scale), -127, 127).astype(np.int8)
    return quantized, scale, offset


def prune_vocabulary(tokenizer, text, min_count=1):
    """Find which tokens are actually used in the text."""
    tokens = tokenizer.encode(text)
    token_counts = defaultdict(int)
    for t in tokens:
        token_counts[t] += 1
    used = sorted([t for t, c in token_counts.items() if c >= min_count])
    old_to_new = {old: new for new, old in enumerate(used)}
    return used, old_to_new


class CausalCrystalCompiler:
    def __init__(self, model_path, output_path, prune_vocab=False, text_path=None,
                 no_quantize=False, use_mixed8=False):
        self.model_path = model_path
        self.output_path = output_path
        self.prune_vocab = prune_vocab
        self.text_path = text_path
        self.no_quantize = no_quantize
        self.use_mixed8 = use_mixed8
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

        # Extract dimensions
        for key in self.state_dict:
            if 'token_embed' in key:
                shape = self.state_dict[key].shape
                self.vocab_size = shape[0]
                self.embed_dim = shape[1]
                break

        # Support both 'attention' and 'geo_attn' naming
        self.attn_prefix = 'geo_attn' if 'geo_attn.positions' in self.state_dict else 'attention'

        for key in self.state_dict:
            if f'{self.attn_prefix}.positions' in key:
                self.num_neurons = self.state_dict[key].shape[0]
                break

        for key in self.state_dict:
            if f'{self.attn_prefix}.frozen' in key:
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

        # Check for causal attention
        self.has_causal = 'causal_attn.q_proj.weight' in self.state_dict
        if self.has_causal:
            self.num_heads = self.config.get('num_heads', 4)
            # Infer from weight shape if not in config
            q_weight = self.state_dict['causal_attn.q_proj.weight']
            self.num_heads = self.embed_dim // (q_weight.shape[0] // self.embed_dim) if self.embed_dim else 4
            # Actually just use 4 as default, the shape is [embed_dim, embed_dim]
            self.num_heads = 4
            print(f"  Causal attention: {self.num_heads} heads")

        print(f"  Vocab: {self.vocab_size}, Embed: {self.embed_dim}, Context: {self.context_len}")
        print(f"  Neurons: {self.num_neurons} ({self.num_frozen} frozen, {self.num_active} active)")
        print(f"  Attention prefix: {self.attn_prefix}")

    def extract_weights(self):
        """Extract and organize weights from state dict."""
        self.weights = {}
        for key, tensor in self.state_dict.items():
            self.weights[key] = tensor.numpy()

    def quantize_all(self):
        """Quantize weights."""
        if self.no_quantize:
            print("Keeping weights as float32...")
        elif self.use_mixed8:
            print("Mixed precision: f32 embeddings + int8 rest...")
        else:
            print("Quantizing weights to int8...")

        self.quantized = {}
        self.scales = {}
        self.offsets_map = {}

        float32_keys = {'token_embed.weight', 'pos_embed.weight'} if self.use_mixed8 else set()

        for key, arr in self.weights.items():
            if arr.dtype == np.bool_:
                self.quantized[key] = arr
                continue

            if self.no_quantize or key in float32_keys:
                self.quantized[key] = arr.astype(np.float32)
                self.scales[key] = 1.0
                self.offsets_map[key] = 0.0
            else:
                q, s, o = quantize_array(arr)
                self.quantized[key] = q
                self.scales[key] = s
                self.offsets_map[key] = o

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
            print(f"  Reduced vocab: {self.vocab_size} → {self.pruned_vocab_size}")
        else:
            self.used_tokens = list(range(self.vocab_size))
            self.token_mapping = {i: i for i in range(self.vocab_size)}
            self.pruned_vocab_size = self.vocab_size

    def write_crystal(self):
        """Write the .crystal file with causal attention support."""
        print(f"Writing {self.output_path}...")

        D = self.embed_dim
        T = self.context_len
        H = self.hidden_dim
        V = self.pruned_vocab_size
        N = self.num_neurons

        # === Prepare data sections ===

        # Token embeddings
        token_embed = self.quantized['token_embed.weight']
        if self.prune_vocab:
            token_embed = token_embed[self.used_tokens]
        embed_data = token_embed.tobytes()

        # Position embeddings
        pos_embed = self.quantized['pos_embed.weight']
        pos_data = pos_embed.tobytes()

        # Causal attention weights (keep as float32 for accuracy)
        causal_data = b''
        if self.has_causal:
            causal_data += self.weights['causal_attn.q_proj.weight'].astype(np.float32).tobytes()
            causal_data += self.weights['causal_attn.q_proj.bias'].astype(np.float32).tobytes()
            causal_data += self.weights['causal_attn.k_proj.weight'].astype(np.float32).tobytes()
            causal_data += self.weights['causal_attn.k_proj.bias'].astype(np.float32).tobytes()
            causal_data += self.weights['causal_attn.v_proj.weight'].astype(np.float32).tobytes()
            causal_data += self.weights['causal_attn.v_proj.bias'].astype(np.float32).tobytes()
            causal_data += self.weights['causal_attn.out_proj.weight'].astype(np.float32).tobytes()
            causal_data += self.weights['causal_attn.out_proj.bias'].astype(np.float32).tobytes()
            print(f"  Causal attention: {len(causal_data):,} bytes")

        # Geometric attention - positions, values, temperatures
        positions = self.weights[f'{self.attn_prefix}.positions'].astype(np.float32)
        values = self.weights[f'{self.attn_prefix}.values'].astype(np.float32)
        temps = self.weights[f'{self.attn_prefix}.temperature'].astype(np.float32)
        geo_data = positions.tobytes() + values.tobytes() + temps.tobytes()

        # FFN weights (float32)
        ffn_w1 = self.weights['ffn.0.weight'].astype(np.float32)
        ffn_b1 = self.weights['ffn.0.bias'].astype(np.float32)
        ffn_w2 = self.weights['ffn.2.weight'].astype(np.float32)
        ffn_b2 = self.weights['ffn.2.bias'].astype(np.float32)
        ffn_data = ffn_w1.tobytes() + ffn_b1.tobytes() + ffn_w2.tobytes() + ffn_b2.tobytes()

        # Output head - quantize for mixed8 (this is the biggest tensor: V×D)
        if self.use_mixed8:
            # Quantize head weights to int8
            head_w = self.quantized['head.weight']
            head_b = self.quantized['head.bias']
            if self.prune_vocab:
                head_w = head_w[self.used_tokens]
                head_b = head_b[self.used_tokens]
            # Pack: scale, offset, then int8 data
            head_data = struct.pack('<ff', self.scales['head.weight'], self.offsets_map['head.weight'])
            head_data += struct.pack('<ff', self.scales['head.bias'], self.offsets_map['head.bias'])
            head_data += head_w.tobytes() + head_b.tobytes()
        else:
            head_w = self.weights['head.weight'].astype(np.float32)
            head_b = self.weights['head.bias'].astype(np.float32)
            if self.prune_vocab:
                head_w = head_w[self.used_tokens]
                head_b = head_b[self.used_tokens]
            head_data = head_w.tobytes() + head_b.tobytes()

        # Layer norms - handle both old (norm1/norm2) and new (norm_causal/norm_geo/norm_ffn) naming
        if 'norm_causal.weight' in self.weights:
            norm_causal_w = self.weights['norm_causal.weight'].astype(np.float32)
            norm_causal_b = self.weights['norm_causal.bias'].astype(np.float32)
            norm_geo_w = self.weights['norm_geo.weight'].astype(np.float32)
            norm_geo_b = self.weights['norm_geo.bias'].astype(np.float32)
            norm_ffn_w = self.weights['norm_ffn.weight'].astype(np.float32)
            norm_ffn_b = self.weights['norm_ffn.bias'].astype(np.float32)
            norm_data = (norm_causal_w.tobytes() + norm_causal_b.tobytes() +
                        norm_geo_w.tobytes() + norm_geo_b.tobytes() +
                        norm_ffn_w.tobytes() + norm_ffn_b.tobytes())
        else:
            # Old naming - only 2 norms
            norm1_w = self.weights['norm1.weight'].astype(np.float32)
            norm1_b = self.weights['norm1.bias'].astype(np.float32)
            norm2_w = self.weights['norm2.weight'].astype(np.float32)
            norm2_b = self.weights['norm2.bias'].astype(np.float32)
            norm_data = norm1_w.tobytes() + norm1_b.tobytes() + norm2_w.tobytes() + norm2_b.tobytes()

        # === Calculate offsets ===
        header_size = 64  # Header size
        embed_offset = header_size
        pos_offset = embed_offset + len(embed_data)
        causal_offset = pos_offset + len(pos_data)
        geo_offset = causal_offset + len(causal_data)
        ffn_offset = geo_offset + len(geo_data)
        head_offset = ffn_offset + len(ffn_data)
        norm_offset = head_offset + len(head_data)
        total_size = norm_offset + len(norm_data)

        # === Build header (72 bytes for v2) ===
        flags = 0
        if self.no_quantize:
            flags |= FLAG_FLOAT32
        if self.use_mixed8:
            flags |= FLAG_MIXED8
        if self.prune_vocab:
            flags |= FLAG_PRUNED_VOCAB
        if self.has_causal:
            flags |= FLAG_CAUSAL

        header = struct.pack('<I', CRYSTAL_MAGIC)           # 4: magic
        header += struct.pack('<H', CRYSTAL_VERSION)        # 2: version (2)
        header += struct.pack('<H', flags)                  # 2: flags
        header += struct.pack('<I', V)                      # 4: vocab_size
        header += struct.pack('<H', D)                      # 2: embed_dim
        header += struct.pack('<H', T)                      # 2: context_len
        header += struct.pack('<H', H)                      # 2: hidden_dim
        header += struct.pack('<H', N)                      # 2: num_neurons
        header += struct.pack('<H', self.num_frozen)        # 2: num_frozen
        header += struct.pack('<H', self.num_active)        # 2: num_active
        header += struct.pack('<H', self.num_heads if self.has_causal else 0)  # 2: num_heads
        header += struct.pack('<H', 3 if self.has_causal else 2)  # 2: num_norms
        header += struct.pack('<HH', 0, 0)  # 4: padding to align to 32 bytes
        # Offsets (8 x 4 = 32 bytes)  -- Total: 32 + 32 = 64
        header += struct.pack('<I', embed_offset)
        header += struct.pack('<I', pos_offset)
        header += struct.pack('<I', causal_offset)
        header += struct.pack('<I', geo_offset)
        header += struct.pack('<I', ffn_offset)
        header += struct.pack('<I', head_offset)
        header += struct.pack('<I', norm_offset)
        header += struct.pack('<I', total_size)  # Total file size

        assert len(header) == 64, f"Header is {len(header)} bytes, expected 64"

        # === Write file ===
        with open(self.output_path, 'wb') as f:
            f.write(header)
            f.write(embed_data)
            f.write(pos_data)
            f.write(causal_data)
            f.write(geo_data)
            f.write(ffn_data)
            f.write(head_data)
            f.write(norm_data)
            actual_size = f.tell()

        print(f"\nFinal .crystal size: {actual_size:,} bytes ({actual_size/1024/1024:.2f} MB)")
        orig_size = os.path.getsize(self.model_path)
        print(f"Original model: {orig_size:,} bytes ({orig_size/1024/1024:.2f} MB)")
        print(f"Compression ratio: {orig_size/actual_size:.1f}x")

    def compile(self):
        """Run the full compilation pipeline."""
        print("=" * 60)
        print("Causal Crystal Compiler v2.0")
        print("=" * 60)

        self.load_model()
        self.extract_weights()
        self.quantize_all()
        self.build_vocabulary()
        self.write_crystal()

        print("\n" + "=" * 60)
        print("Compilation complete!")
        print("=" * 60)


def main():
    if len(sys.argv) < 3:
        print("Usage: python causal_crystal_compiler.py model.pt output.crystal [--prune-vocab text.txt] [--mixed8]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]
    prune_vocab = '--prune-vocab' in sys.argv
    text_path = None
    if prune_vocab:
        idx = sys.argv.index('--prune-vocab')
        if idx + 1 < len(sys.argv):
            text_path = sys.argv[idx + 1]

    use_mixed8 = '--mixed8' in sys.argv
    no_quantize = '--no-quantize' in sys.argv or (not use_mixed8)  # Default to float32 unless mixed8

    compiler = CausalCrystalCompiler(
        model_path, output_path,
        prune_vocab=prune_vocab,
        text_path=text_path,
        no_quantize=no_quantize,
        use_mixed8=use_mixed8
    )
    compiler.compile()


if __name__ == "__main__":
    main()
