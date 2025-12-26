#!/usr/bin/env python3
"""
Shakespeare Generator Helper
Runs the compiled C crystal and decodes the output to text.

Usage: python shakespeare.py
       python shakespeare.py --tokens 200
"""

import subprocess
import tiktoken
import re
import sys

def generate_shakespeare(num_tokens=100):
    """Run the C binary and decode output."""

    # Run the compiled binary
    result = subprocess.run(['./crystal_shakespeare'], capture_output=True, text=True)
    output = result.stdout

    # Extract token IDs from output
    # Look for the line with numbers
    lines = output.split('\n')
    token_line = None
    for line in lines:
        if re.match(r'^[\d\s]+$', line.strip()) and len(line.strip()) > 10:
            token_line = line
            break

    if not token_line:
        print("Could not find token output")
        print(output)
        return

    # Parse tokens
    tokens = [int(t) for t in token_line.strip().split()]

    # Decode with tiktoken
    enc = tiktoken.get_encoding('gpt2')
    text = enc.decode(tokens)

    # Print nicely
    print("=" * 60)
    print("🎭 CRYSTAL SHAKESPEARE - 788 Neurons of Poetry 🎭")
    print("=" * 60)
    print()
    print(text)
    print()
    print("=" * 60)
    print(f"Generated {len(tokens)} tokens from pure C")
    print("=" * 60)

if __name__ == "__main__":
    generate_shakespeare()
