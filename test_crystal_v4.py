#!/usr/bin/env python3
"""
Test generation quality of the v4 crystal!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

print("=" * 70)
print("TESTING V4 CRYSTAL GENERATION QUALITY")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# CRYSTAL ARCHITECTURE (same as v4)
# ============================================================================

class CrystalAttention(nn.Module):
    def __init__(self, embed_dim, initial_neurons, max_neurons=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_neurons = max_neurons
        self.num_neurons = initial_neurons
        self.positions = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.02)
        self.scales = nn.Parameter(torch.ones(max_neurons) * 5.0)
        self.values = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.02)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.register_buffer('frozen', torch.zeros(max_neurons, dtype=torch.bool))
        self.register_buffer('temperature', torch.ones(max_neurons))
        self.register_buffer('grad_ema', torch.zeros(max_neurons))
        self.register_buffer('cold_epochs', torch.zeros(max_neurons))

    def forward(self, x):
        B, T, D = x.shape
        N = self.num_neurons
        pos = self.positions[:N].unsqueeze(0).unsqueeze(0)
        x_exp = x.unsqueeze(2)
        dist = torch.norm(x_exp - pos, dim=-1)
        attn = F.softmax(self.scales[:N] / (dist + 0.1), dim=-1)
        out = torch.einsum('btn,nd->btd', attn, self.values[:N])
        return self.out_proj(out)


class CrystalGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, initial_neurons, num_blocks, max_neurons=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(1024, embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(embed_dim),
                'attn': CrystalAttention(embed_dim, initial_neurons, max_neurons),
                'ln2': nn.LayerNorm(embed_dim),
                'mlp': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
            }))
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.wte.weight

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = x + block['attn'](block['ln1'](x))
            x = x + block['mlp'](block['ln2'](x))
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1024:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n[1] Loading v4 crystal...")

checkpoint = torch.load('runs/crystal_gpt2_20251226_150401/crystal_final.pt', map_location=device)
config = checkpoint['config']
print(f"    Config: {config}")

crystal = CrystalGPT(
    vocab_size=50257,
    embed_dim=768,
    initial_neurons=config['neurons'],
    num_blocks=config['blocks'],
    max_neurons=config['max']
).to(device)

# Load state dict
crystal.load_state_dict(checkpoint['model'])

# IMPORTANT: Restore num_neurons from history!
# The history tells us how many neurons grew
final_neurons = checkpoint['history']['neurons'][-1]  # Total neurons
neurons_per_block = final_neurons // config['blocks']  # 2448 / 6 = 408
print(f"    Restoring num_neurons to {neurons_per_block} per block...")
for block in crystal.blocks:
    block['attn'].num_neurons = neurons_per_block

crystal.eval()

# Get stats
total_neurons = sum(b['attn'].num_neurons for b in crystal.blocks)
frozen_neurons = sum(b['attn'].frozen[:b['attn'].num_neurons].sum().item() for b in crystal.blocks)
active_neurons = total_neurons - frozen_neurons

print(f"\n    Total neurons: {total_neurons}")
print(f"    Frozen: {frozen_neurons} ({100*frozen_neurons/total_neurons:.1f}%)")
print(f"    Active: {active_neurons} ({100*active_neurons/total_neurons:.1f}%)")

# ============================================================================
# TEST GENERATION
# ============================================================================
print("\n" + "=" * 70)
print("[2] GENERATION TEST")
print("=" * 70)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompts = [
    "The meaning of life is",
    "Neural networks learn",
    "In the beginning,",
    "The universe began",
    "Science has shown that",
    "Artificial intelligence will",
    "The crystal structure of",
    "Knowledge is",
    "The future of computing",
    "Deep learning models",
]

print("\nCrystal Generation (temperature=0.7):\n")
print("-" * 70)

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = crystal.generate(input_ids, max_new_tokens=40, temperature=0.7, top_k=50)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  {text[:100]}...")
    print()

# ============================================================================
# COMPARE WITH DIFFERENT TEMPERATURES
# ============================================================================
print("=" * 70)
print("[3] TEMPERATURE COMPARISON")
print("=" * 70)

test_prompt = "The most important discovery in science was"
print(f"\nPrompt: \"{test_prompt}\"\n")

for temp in [0.3, 0.5, 0.7, 0.9, 1.2]:
    input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)
    output = crystal.generate(input_ids, max_new_tokens=30, temperature=temp, top_k=50)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  T={temp}: {text}")
    print()

# ============================================================================
# LONGER GENERATION
# ============================================================================
print("=" * 70)
print("[4] LONGER GENERATION (100 tokens)")
print("=" * 70)

long_prompts = [
    "Once upon a time",
    "The theory of relativity states that",
    "In modern computer science,",
]

for prompt in long_prompts:
    print(f"\nPrompt: \"{prompt}\"\n")
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = crystal.generate(input_ids, max_new_tokens=100, temperature=0.7, top_k=50)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  {text}")
    print()
    print("-" * 70)

print("\n" + "=" * 70)
print("GENERATION TEST COMPLETE")
print("=" * 70)
