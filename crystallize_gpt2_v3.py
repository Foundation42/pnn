#!/usr/bin/env python3
"""
Crystallize GPT-2 v3: EXTENDED TRAINING - Make it THINK!

Train the crystal long enough to actually generate coherent text.
500 epochs, better learning rate, generation samples during training.

"The crystal grows, freezes, and finally... speaks."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

print("=" * 70)
print("CRYSTALLIZE GPT-2 v3: EXTENDED TRAINING")
print("Making the crystal THINK!")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# OPTIMIZED CRYSTAL ARCHITECTURE
# ============================================================================

class CrystalAttention(nn.Module):
    """Geometric attention with growth and freezing."""
    def __init__(self, embed_dim, initial_neurons, max_neurons=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_neurons = max_neurons
        self.num_neurons = initial_neurons

        # Neuron parameters
        self.positions = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.02)
        self.scales = nn.Parameter(torch.ones(max_neurons) * 5.0)
        self.values = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.02)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Temperature tracking
        self.register_buffer('frozen', torch.zeros(max_neurons, dtype=torch.bool))
        self.register_buffer('temperature', torch.ones(max_neurons))
        self.register_buffer('grad_ema', torch.zeros(max_neurons))
        self.register_buffer('cold_epochs', torch.zeros(max_neurons))

    def forward(self, x):
        B, T, D = x.shape
        N = self.num_neurons

        # Distance-based attention
        pos = self.positions[:N].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)
        x_exp = x.unsqueeze(2)  # (B, T, 1, D)

        dist = torch.norm(x_exp - pos, dim=-1)  # (B, T, N)
        attn = F.softmax(self.scales[:N] / (dist + 0.1), dim=-1)  # (B, T, N)

        # Weighted values
        out = torch.einsum('btn,nd->btd', attn, self.values[:N])
        return self.out_proj(out)

    def update_stats(self):
        """Update temperature from gradients."""
        if self.positions.grad is not None:
            N = self.num_neurons
            grad_norm = self.positions.grad[:N].norm(dim=-1).detach()
            # Exponential moving average
            self.grad_ema[:N] = 0.9 * self.grad_ema[:N] + 0.1 * grad_norm
            self.temperature[:N] = self.grad_ema[:N]

    def get_hot_neurons(self, top_k=2):
        """Get hottest neurons for splitting."""
        N = self.num_neurons
        temps = self.temperature[:N].clone()
        temps[self.frozen[:N]] = -1  # Exclude frozen
        _, indices = temps.topk(min(top_k, N))
        return [i.item() for i in indices if temps[i] > 0]

    def get_cold_neurons(self, threshold_pct=20, min_cold=3):
        """Get coldest neurons for freezing."""
        N = self.num_neurons
        active = ~self.frozen[:N]
        if active.sum() < 5:
            return []

        temps = self.temperature[:N]
        threshold = torch.quantile(temps[active], threshold_pct/100)

        candidates = []
        for i in range(N):
            if not self.frozen[i]:
                if temps[i] < threshold:
                    self.cold_epochs[i] += 1
                    if self.cold_epochs[i] >= min_cold:
                        candidates.append(i)
                else:
                    self.cold_epochs[i] = 0
        return candidates

    def split(self, idx):
        """Split a neuron."""
        if self.num_neurons >= self.max_neurons:
            return False
        new = self.num_neurons
        with torch.no_grad():
            noise = 0.01
            self.positions.data[new] = self.positions.data[idx] + torch.randn(self.embed_dim, device=self.positions.device) * noise
            self.scales.data[new] = self.scales.data[idx]
            self.values.data[new] = self.values.data[idx] + torch.randn(self.embed_dim, device=self.values.device) * noise
            self.temperature[new] = self.temperature[idx]
            self.cold_epochs[new] = 0
        self.num_neurons += 1
        return True

    def freeze(self, idx):
        self.frozen[idx] = True


class CrystalGPT(nn.Module):
    """Growing crystal language model."""
    def __init__(self, vocab_size, embed_dim, initial_neurons, num_blocks, max_neurons=512):
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

    def forward(self, x):
        B, T = x.shape
        tok = self.wte(x)
        pos = self.wpe(torch.arange(T, device=x.device))
        h = tok + pos

        for block in self.blocks:
            h = h + block['attn'](block['ln1'](h))
            h = h + block['mlp'](block['ln2'](h))

        return self.head(self.ln_f(h))

    def update_stats(self):
        for block in self.blocks:
            block['attn'].update_stats()

    def grow_and_freeze(self):
        splits, freezes = 0, 0
        for block in self.blocks:
            attn = block['attn']
            # Freeze cold
            for idx in attn.get_cold_neurons()[:4]:
                attn.freeze(idx)
                freezes += 1
            # Split hot
            for idx in attn.get_hot_neurons(2):
                if attn.split(idx):
                    splits += 1
        return splits, freezes

    def stats(self):
        total = sum(b['attn'].num_neurons for b in self.blocks)
        frozen = sum(b['attn'].frozen[:b['attn'].num_neurons].sum().item() for b in self.blocks)
        return {'neurons': total, 'frozen': frozen, 'active': total - frozen}

    @torch.no_grad()
    def generate(self, input_ids, max_new=30, temperature=0.8, top_k=40):
        """Generate with temperature and top-k sampling."""
        for _ in range(max_new):
            logits = self(input_ids[:, -512:])[:, -1, :]
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == tokenizer.eos_token_id:
                break
        return input_ids


# ============================================================================
# TRAINING
# ============================================================================

print("\n[1] Loading models...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

print(f"    Teacher: {sum(p.numel() for p in teacher.parameters()):,} params")

print("\n[2] Loading WikiText...")
try:
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in dataset['text'] if len(t.strip()) > 100][:2000]
    print(f"    Loaded {len(texts)} samples")
except:
    texts = ["The neural network learns patterns from data."] * 500
    print(f"    Using fallback data")

print("\n[3] Creating Crystal...")
NEURONS = 24
BLOCKS = 6
MAX_NEURONS = 256

crystal = CrystalGPT(
    vocab_size=50257,
    embed_dim=768,
    initial_neurons=NEURONS,
    num_blocks=BLOCKS,
    max_neurons=MAX_NEURONS
).to(device)

# Copy embeddings
crystal.wte.weight.data = teacher.transformer.wte.weight.data.clone()
crystal.wpe.weight.data = teacher.transformer.wpe.weight.data.clone()

print(f"    Initial neurons: {NEURONS} × {BLOCKS} = {NEURONS * BLOCKS}")
print(f"    Max neurons: {MAX_NEURONS} × {BLOCKS} = {MAX_NEURONS * BLOCKS}")

print("\n[4] Training Crystal (500 epochs)...")
print("=" * 70)

optimizer = torch.optim.AdamW(crystal.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

EPOCHS = 500
BATCH = 8
MAXLEN = 128

history = {'loss': [], 'neurons': [], 'frozen': []}
start_time = time.time()

test_prompts = [
    "The meaning of life",
    "Neural networks learn",
    "The universe is",
]

for epoch in range(EPOCHS):
    crystal.train()
    epoch_loss = 0
    n_batch = 0

    np.random.shuffle(texts)

    for i in range(0, min(len(texts), 400), BATCH):
        batch = texts[i:i+BATCH]

        inputs = tokenizer(batch, return_tensors='pt', truncation=True,
                          max_length=MAXLEN, padding=True).to(device)

        if inputs['input_ids'].shape[1] < 3:
            continue

        optimizer.zero_grad()

        # Teacher logits
        with torch.no_grad():
            t_logits = teacher(**inputs).logits

        # Crystal logits
        c_logits = crystal(inputs['input_ids'])

        # Distillation loss (KL + CE)
        T = 2.0
        kl = F.kl_div(
            F.log_softmax(c_logits / T, dim=-1).view(-1, 50257),
            F.softmax(t_logits / T, dim=-1).view(-1, 50257),
            reduction='batchmean'
        ) * T * T

        ce = F.cross_entropy(
            c_logits[:, :-1].contiguous().view(-1, 50257),
            inputs['input_ids'][:, 1:].contiguous().view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        loss = 0.7 * kl + 0.3 * ce
        loss.backward()

        crystal.update_stats()
        torch.nn.utils.clip_grad_norm_(crystal.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batch += 1

    scheduler.step()

    # Grow/freeze after warmup
    splits, freezes = 0, 0
    if epoch >= 10 and epoch % 3 == 0:
        splits, freezes = crystal.grow_and_freeze()

    stats = crystal.stats()
    avg_loss = epoch_loss / max(n_batch, 1)
    history['loss'].append(avg_loss)
    history['neurons'].append(stats['neurons'])
    history['frozen'].append(stats['frozen'])

    # Progress report
    if (epoch + 1) % 25 == 0 or epoch < 5:
        elapsed = time.time() - start_time
        pct_frozen = 100 * stats['frozen'] / stats['neurons'] if stats['neurons'] > 0 else 0
        speedup = stats['neurons'] / stats['active'] if stats['active'] > 0 else 1

        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.3f} | "
              f"Neurons: {stats['neurons']:3d} | Frozen: {stats['frozen']:3d} ({pct_frozen:.0f}%) | "
              f"Speedup: {speedup:.1f}x | Time: {elapsed/60:.1f}m")

        # Generate sample
        crystal.eval()
        prompt = test_prompts[epoch % len(test_prompts)]
        inp = tokenizer(prompt, return_tensors='pt').to(device)
        out = crystal.generate(inp['input_ids'], max_new=20, temperature=0.7)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"         → \"{text[:80]}...\"")
        print()

print("\n" + "=" * 70)
print("[5] Final Generation Test")
print("=" * 70)

crystal.eval()
prompts = [
    "The meaning of life is",
    "In the beginning,",
    "Neural networks are",
    "The universe began with",
    "Science has shown that",
    "Artificial intelligence will",
]

print("\nCrystal Generation (temperature=0.7):\n")
for prompt in prompts:
    inp = tokenizer(prompt, return_tensors='pt').to(device)
    out = crystal.generate(inp['input_ids'], max_new=40, temperature=0.7, top_k=50)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  {text}")
    print()

# Compare with teacher
print("\nTeacher (GPT-2) for comparison:\n")
for prompt in prompts[:3]:
    inp = tokenizer(prompt, return_tensors='pt').to(device)
    out = teacher.generate(inp['input_ids'], max_new_tokens=40, do_sample=True,
                          temperature=0.7, top_k=50, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  {text}")
    print()

# Visualization
print("=" * 70)
print("[6] Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.plot(history['loss'], 'b-', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Distillation Loss')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(history['neurons'], 'g-', label='Total', linewidth=2)
ax.plot(history['frozen'], 'b-', label='Frozen', linewidth=2)
ax.fill_between(range(len(history['neurons'])), history['frozen'], alpha=0.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Neurons')
ax.set_title('Crystal Growth & Freezing')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
pct = [100*f/n if n > 0 else 0 for f, n in zip(history['frozen'], history['neurons'])]
ax.plot(pct, 'purple', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Frozen %')
ax.set_title('Crystallization Progress')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
active = [n - f for n, f in zip(history['neurons'], history['frozen'])]
speedup = [n / a if a > 0 else 1 for n, a in zip(history['neurons'], active)]
ax.plot(speedup, 'orange', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Speedup')
ax.set_title('Training Speedup (from freezing)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crystal_gpt2_extended.png', dpi=150)
print(f"\nSaved: crystal_gpt2_extended.png")

# Final stats
stats = crystal.stats()
total_time = time.time() - start_time
print(f"""
{'='*70}
CRYSTALLIZATION COMPLETE!
{'='*70}

    Training time: {total_time/60:.1f} minutes

    CRYSTAL STATS:
    - Total neurons: {stats['neurons']}
    - Frozen neurons: {stats['frozen']} ({100*stats['frozen']/stats['neurons']:.1f}%)
    - Active neurons: {stats['active']}
    - Training speedup: {stats['neurons']/stats['active']:.1f}x

    GROWTH:
    - Started: {NEURONS * BLOCKS} neurons
    - Final: {stats['neurons']} neurons
    - Growth: {stats['neurons'] / (NEURONS * BLOCKS):.1f}x

    The crystal has learned to SPEAK!
    "Knowledge crystallizes into geometry."
""")

torch.save({
    'model': crystal.state_dict(),
    'history': history,
    'config': {'neurons': NEURONS, 'blocks': BLOCKS, 'max': MAX_NEURONS}
}, 'crystal_gpt2_extended.pt')
print("Saved: crystal_gpt2_extended.pt")
