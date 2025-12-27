#!/usr/bin/env python3
"""
Crystal Shakespeare Generator
Runs the crystal runtime with .crystal model and decodes output to text.

Usage: python crystal_demo.py
       python crystal_demo.py shakespeare.crystal 200 0.8
"""

import os
import subprocess
import sys
import tiktoken
from collections import defaultdict

def load_vocab_mapping(text_path='data/tinyshakespeare.txt'):
    """Create mapping from pruned vocab IDs to original GPT-2 tokens."""
    tokenizer = tiktoken.get_encoding("gpt2")

    with open(text_path, 'r') as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    token_counts = defaultdict(int)
    for t in tokens:
        token_counts[t] += 1

    used = sorted([t for t, c in token_counts.items() if c >= 1])
    new_to_old = {new: old for new, old in enumerate(used)}

    return tokenizer, new_to_old

def generate_shakespeare(model_path='shakespeare_788_f32.crystal', num_tokens=100, temperature=0.8):
    """Run the crystal runtime and decode output."""

    # Load vocab mapping
    tokenizer, new_to_old = load_vocab_mapping()

    # Run the crystal runtime
    result = subprocess.run(
        ['./crystal_runtime', model_path, str(num_tokens), str(temperature)],
        capture_output=True,
        text=True
    )

    # Parse output - find the token IDs line
    lines = result.stdout.strip().split('\n')
    token_line = None
    for line in lines:
        parts = line.strip().split()
        if len(parts) > 10 and all(p.isdigit() for p in parts[:10]):
            token_line = line
            break

    if not token_line:
        print("Could not find token output")
        print(result.stdout)
        return

    # Parse and decode tokens
    token_ids = [int(x) for x in token_line.strip().split()]
    text = ""
    for tid in token_ids:
        if tid in new_to_old:
            text += tokenizer.decode([new_to_old[tid]])
        else:
            text += f"[{tid}]"

    # Print nicely (matching shakespeare.py style)
    print("=" * 60)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"🎭 CRYSTAL SHAKESPEARE - {size_mb:.1f}MB of Poetry 🎭")
    print("=" * 60)
    print()
    print(text)
    print()
    print("=" * 60)
    print(f"Generated {len(token_ids)} tokens")
    print(f"Model: {model_path} | Runtime: crystal_runtime (41KB)")
    print("=" * 60)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else 'shakespeare_788_f32.crystal'
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    generate_shakespeare(model, tokens, temp)
