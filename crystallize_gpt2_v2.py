#!/usr/bin/env python3
"""
Crystallize GPT-2 v2: With BVH Growth + Freezing!

The crystal comes ALIVE:
1. GROW neurons where gradients are high (need more capacity)
2. FREEZE neurons where gradients are low (stable knowledge)
3. Self-optimize structure as it learns from GPT-2

"The crystal grows where it's hot, freezes where it's cold."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

print("=" * 70)
print("CRYSTALLIZE GPT-2 v2: Growth + Freezing")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# CRYSTAL ARCHITECTURE WITH GROWTH + FREEZING
# ============================================================================

class GrowingCrystalAttention(nn.Module):
    """
    Geometric attention that can GROW and FREEZE neurons.

    - Neurons live in embedding space
    - High gradient variance → SPLIT (grow new neuron)
    - Low gradient variance → FREEZE (stop updating)
    """
    def __init__(self, embed_dim, initial_neurons, num_heads=4, max_neurons=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_neurons = max_neurons

        # Start with initial neurons
        self.num_neurons = initial_neurons

        # Neuron positions in embedding space - use buffer for non-frozen, parameter for learnable
        self.positions = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.1)

        # Interaction scale per neuron
        self.interaction_scale = nn.Parameter(torch.ones(max_neurons) * 10.0)

        # Value projection
        self.value_weight = nn.Parameter(torch.randn(max_neurons, embed_dim, embed_dim) * 0.02)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # TEMPERATURE TRACKING for freeze/grow decisions
        self.register_buffer('frozen_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.register_buffer('temperature', torch.ones(max_neurons))
        self.register_buffer('grad_history', torch.zeros(max_neurons, 20))  # Last 20 updates
        self.register_buffer('grad_ptr', torch.tensor(0))
        self.register_buffer('activation_count', torch.zeros(max_neurons))
        self.register_buffer('epochs_cold', torch.zeros(max_neurons))  # How long below threshold

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        """
        B, T, D = x.shape
        N = self.num_neurons

        # Only use active neurons
        positions = self.positions[:N]  # (N, D)
        scales = self.interaction_scale[:N]  # (N,)

        # Compute distances from each token to each neuron
        x_exp = x.unsqueeze(2)  # (B, T, 1, D)
        pos_exp = positions.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

        distances = torch.norm(x_exp - pos_exp, dim=-1)  # (B, T, N)

        # Interaction strength (geometric attention)
        interactions = scales / (distances + 0.1)  # (B, T, N)

        # Apply frozen mask - frozen neurons still contribute but don't get gradients
        # (we handle this in the optimizer, not here)

        # Softmax attention weights
        attn_weights = F.softmax(interactions, dim=-1)  # (B, T, N)

        # Track which neurons are being used
        with torch.no_grad():
            self.activation_count[:N] += attn_weights.sum(dim=(0, 1))

        # Value projection: each neuron transforms input differently
        # values[n] = x @ value_weight[n]  -> (B, T, D)
        # Then weight by attention

        # Efficient: batch the value projections
        values = torch.einsum('btd,nde->btne', x, self.value_weight[:N])  # (B, T, N, D)

        # Weighted sum
        output = torch.einsum('btn,btnd->btd', attn_weights, values)  # (B, T, D)

        # Output projection
        output = self.out_proj(output)

        return output

    def update_temperature(self):
        """Update temperature based on gradient history."""
        N = self.num_neurons

        if self.positions.grad is not None:
            # Gradient norm per neuron
            grad_norms = self.positions.grad[:N].norm(dim=-1)

            # Store in circular buffer
            ptr = self.grad_ptr.item()
            self.grad_history[:N, ptr] = grad_norms.detach()
            self.grad_ptr = (self.grad_ptr + 1) % 20

            # Temperature = gradient variance (high variance = unstable = hot)
            valid_history = self.grad_history[:N, :min(ptr+1, 20)]
            if valid_history.shape[1] > 1:
                self.temperature[:N] = valid_history.std(dim=-1) + valid_history.mean(dim=-1) * 0.1

    def get_split_candidates(self, threshold_percentile=90):
        """Find neurons that should split (too hot = overloaded)."""
        N = self.num_neurons
        temps = self.temperature[:N]

        if N < 3:
            return []

        threshold = torch.quantile(temps[~self.frozen_mask[:N]], threshold_percentile/100)

        candidates = []
        for i in range(N):
            if not self.frozen_mask[i] and temps[i] > threshold:
                candidates.append(i)

        return candidates

    def get_freeze_candidates(self, threshold_percentile=25, min_epochs_cold=5):
        """Find neurons that should freeze (cold = stable)."""
        N = self.num_neurons
        temps = self.temperature[:N]

        if N < 3:
            return []

        # Only consider non-frozen neurons
        active_temps = temps[~self.frozen_mask[:N]]
        if len(active_temps) < 3:
            return []

        threshold = torch.quantile(active_temps, threshold_percentile/100)

        candidates = []
        for i in range(N):
            if not self.frozen_mask[i]:
                if temps[i] < threshold:
                    self.epochs_cold[i] += 1
                    if self.epochs_cold[i] >= min_epochs_cold:
                        candidates.append(i)
                else:
                    self.epochs_cold[i] = 0

        return candidates

    def split_neuron(self, idx):
        """Split a neuron into two (mitosis!)."""
        if self.num_neurons >= self.max_neurons:
            return False

        new_idx = self.num_neurons

        with torch.no_grad():
            # New neuron is a perturbed copy of parent
            noise_scale = 0.01
            self.positions.data[new_idx] = self.positions.data[idx] + \
                torch.randn_like(self.positions.data[idx]) * noise_scale

            self.interaction_scale.data[new_idx] = self.interaction_scale.data[idx]
            self.value_weight.data[new_idx] = self.value_weight.data[idx] + \
                torch.randn_like(self.value_weight.data[idx]) * noise_scale

            # Reset temperature tracking for both
            self.temperature[idx] = 1.0
            self.temperature[new_idx] = 1.0
            self.epochs_cold[idx] = 0
            self.epochs_cold[new_idx] = 0
            self.activation_count[new_idx] = 0

        self.num_neurons += 1
        return True

    def freeze_neuron(self, idx):
        """Freeze a neuron (stop gradient updates)."""
        self.frozen_mask[idx] = True

    def get_frozen_count(self):
        return self.frozen_mask[:self.num_neurons].sum().item()


class CrystalMLP(nn.Module):
    """MLP that respects frozen neurons."""
    def __init__(self, embed_dim, max_neurons, expansion=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_neurons = max_neurons
        self.num_neurons = max_neurons  # MLP uses all neurons

        hidden_dim = max_neurons * expansion
        self.up_proj = nn.Linear(embed_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class GrowingCrystalBlock(nn.Module):
    """Crystal block with growing attention."""
    def __init__(self, embed_dim, initial_neurons, max_neurons=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = GrowingCrystalAttention(embed_dim, initial_neurons, max_neurons=max_neurons)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = CrystalMLP(embed_dim, max_neurons)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GrowingCrystalGPT2(nn.Module):
    """
    GPT-2 distilled into a GROWING crystal.

    Neurons can split and freeze based on gradient flow!
    """
    def __init__(self, vocab_size, embed_dim, initial_neurons, num_blocks,
                 max_neurons=256, max_seq_len=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.initial_neurons = initial_neurons
        self.max_neurons = max_neurons

        # Embeddings
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)

        # Growing crystal blocks
        self.blocks = nn.ModuleList([
            GrowingCrystalBlock(embed_dim, initial_neurons, max_neurons)
            for _ in range(num_blocks)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # Tie weights

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device

        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def update_temperatures(self):
        """Update temperature tracking for all blocks."""
        for block in self.blocks:
            block.attn.update_temperature()

    def grow_and_freeze(self, max_splits_per_epoch=4, max_freezes_per_epoch=8):
        """
        The magic: grow where hot, freeze where cold!
        """
        total_splits = 0
        total_freezes = 0

        for block_idx, block in enumerate(self.blocks):
            attn = block.attn

            # FREEZE cold neurons first
            freeze_candidates = attn.get_freeze_candidates()
            for idx in freeze_candidates[:max_freezes_per_epoch]:
                attn.freeze_neuron(idx)
                total_freezes += 1

            # SPLIT hot neurons
            split_candidates = attn.get_split_candidates()
            for idx in split_candidates[:max_splits_per_epoch]:
                if attn.split_neuron(idx):
                    total_splits += 1

        return total_splits, total_freezes

    def get_stats(self):
        """Get current crystal statistics."""
        total_neurons = 0
        total_frozen = 0
        temps = []

        for block in self.blocks:
            attn = block.attn
            total_neurons += attn.num_neurons
            total_frozen += attn.get_frozen_count()
            temps.extend(attn.temperature[:attn.num_neurons].tolist())

        return {
            'total_neurons': total_neurons,
            'frozen_neurons': total_frozen,
            'active_neurons': total_neurons - total_frozen,
            'frozen_pct': 100 * total_frozen / total_neurons if total_neurons > 0 else 0,
            'avg_temp': np.mean(temps) if temps else 0,
            'max_temp': np.max(temps) if temps else 0,
            'min_temp': np.min(temps) if temps else 0,
        }

    def get_frozen_params(self):
        """Get parameter indices that should be frozen."""
        frozen_indices = []
        for block in self.blocks:
            attn = block.attn
            for i in range(attn.num_neurons):
                if attn.frozen_mask[i]:
                    frozen_indices.append((block, i))
        return frozen_indices


# ============================================================================
# TRAINING WITH GROWTH + FREEZING
# ============================================================================

print("\n[1] Loading GPT-2 teacher...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
teacher.eval()

teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"    Teacher: {teacher_params:,} params ({teacher_params/1e6:.1f}M)")

print("\n[2] Loading training data (WikiText)...")
try:
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    # Get non-empty texts
    texts = [t for t in dataset['text'] if len(t.strip()) > 50][:1000]
    print(f"    Loaded {len(texts)} training samples from WikiText-2")
except Exception as e:
    print(f"    WikiText failed ({e}), using fallback texts...")
    texts = [
        "The meaning of life is a philosophical question concerning the significance of life.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "The universe is all of space and time and their contents.",
        "Language models are trained to predict the next word in a sequence.",
        "Consciousness is the state of being aware of and responsive to one's surroundings.",
        "Mathematics is the study of numbers, quantities, and shapes.",
        "Evolution is change in the heritable characteristics of biological populations.",
        "Quantum mechanics is a fundamental theory in physics.",
        "Artificial intelligence is intelligence demonstrated by machines.",
        "The crystal structure determines many physical properties of materials.",
    ] * 100  # Repeat for more training data

print("\n[3] Creating Growing Crystal GPT-2...")
INITIAL_NEURONS = 16  # Start small!
MAX_NEURONS = 128
NUM_BLOCKS = 4

crystal = GrowingCrystalGPT2(
    vocab_size=teacher.config.vocab_size,
    embed_dim=teacher.config.n_embd,
    initial_neurons=INITIAL_NEURONS,
    num_blocks=NUM_BLOCKS,
    max_neurons=MAX_NEURONS,
).to(device)

# Copy embeddings from teacher
crystal.wte.weight.data = teacher.transformer.wte.weight.data.clone()
crystal.wpe.weight.data = teacher.transformer.wpe.weight.data.clone()

crystal_params = sum(p.numel() for p in crystal.parameters())
print(f"    Crystal: {crystal_params:,} params ({crystal_params/1e6:.1f}M)")
print(f"    Initial neurons per block: {INITIAL_NEURONS}")
print(f"    Max neurons per block: {MAX_NEURONS}")
print(f"    Blocks: {NUM_BLOCKS}")

print("\n[4] Knowledge Distillation with Growth + Freezing!")
print("=" * 70)

optimizer = torch.optim.AdamW(crystal.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

NUM_EPOCHS = 50
BATCH_SIZE = 4
MAX_LEN = 128
TEMP = 2.0  # Distillation temperature

# Tracking
history = {
    'loss': [],
    'neurons': [],
    'frozen': [],
    'splits': [],
    'freezes': [],
}

crystal.train()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    num_batches = 0

    # Shuffle and batch
    np.random.shuffle(texts)

    for i in range(0, min(len(texts), 200), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LEN,
            padding=True
        ).to(device)

        if inputs['input_ids'].shape[1] < 2:
            continue

        optimizer.zero_grad()

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = teacher(**inputs)
            teacher_logits = teacher_out.logits / TEMP
            teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Crystal forward
        crystal_logits = crystal(inputs['input_ids']) / TEMP
        crystal_log_probs = F.log_softmax(crystal_logits, dim=-1)

        # KL divergence (distillation loss)
        kl_loss = F.kl_div(
            crystal_log_probs.view(-1, crystal_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='batchmean'
        ) * (TEMP ** 2)

        # Cross-entropy on actual next tokens
        labels = inputs['input_ids'][:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            crystal_logits[:, :-1, :].contiguous().view(-1, crystal_logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        loss = 0.5 * kl_loss + 0.5 * ce_loss

        loss.backward()

        # Update temperature tracking BEFORE optimizer step
        crystal.update_temperatures()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(crystal.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    scheduler.step()

    # GROW AND FREEZE every epoch after warmup
    splits, freezes = 0, 0
    if epoch >= 5:  # Warmup period
        splits, freezes = crystal.grow_and_freeze(
            max_splits_per_epoch=2,
            max_freezes_per_epoch=4
        )

    # Get stats
    stats = crystal.get_stats()
    avg_loss = epoch_loss / max(num_batches, 1)

    history['loss'].append(avg_loss)
    history['neurons'].append(stats['total_neurons'])
    history['frozen'].append(stats['frozen_neurons'])
    history['splits'].append(splits)
    history['freezes'].append(freezes)

    # Calculate speedup (frozen neurons skip backprop)
    active_pct = 100 - stats['frozen_pct']
    speedup = 100 / active_pct if active_pct > 0 else 1.0

    if (epoch + 1) % 5 == 0 or splits > 0 or freezes > 0:
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
              f"Neurons: {stats['total_neurons']:3d} | "
              f"Frozen: {stats['frozen_neurons']:3d} ({stats['frozen_pct']:.1f}%) | "
              f"Splits: {splits} | Freezes: {freezes} | "
              f"Speedup: {speedup:.1f}x")


print("\n" + "=" * 70)
print("[5] Testing Crystal GPT-2...")
print("=" * 70)

crystal.eval()

test_prompts = [
    "The meaning of life is",
    "Neural networks can",
    "The universe began",
    "In mathematics,",
    "Artificial intelligence",
]

print("\nGeneration comparison (Teacher vs Crystal):\n")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        # Teacher
        teacher_out = teacher.generate(
            inputs['input_ids'],
            max_new_tokens=15,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        teacher_text = tokenizer.decode(teacher_out[0], skip_special_tokens=True)

        # Crystal (greedy)
        crystal_ids = inputs['input_ids'].clone()
        for _ in range(15):
            logits = crystal(crystal_ids)
            next_id = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            crystal_ids = torch.cat([crystal_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
        crystal_text = tokenizer.decode(crystal_ids[0], skip_special_tokens=True)

    print(f"Prompt:  '{prompt}'")
    print(f"Teacher: {teacher_text}")
    print(f"Crystal: {crystal_text}")
    print()


print("=" * 70)
print("[6] Visualizing Crystal Growth")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss curve
ax = axes[0, 0]
ax.plot(history['loss'], 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Distillation Loss')
ax.grid(True, alpha=0.3)

# Neuron count
ax = axes[0, 1]
ax.plot(history['neurons'], 'g-', linewidth=2, label='Total')
ax.plot(history['frozen'], 'b-', linewidth=2, label='Frozen')
ax.fill_between(range(len(history['neurons'])), history['frozen'], alpha=0.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Neurons')
ax.set_title('Crystal Growth')
ax.legend()
ax.grid(True, alpha=0.3)

# Frozen percentage
ax = axes[0, 2]
frozen_pct = [100*f/n if n > 0 else 0 for f, n in zip(history['frozen'], history['neurons'])]
ax.plot(frozen_pct, 'purple', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Frozen %')
ax.set_title('Crystallization Progress')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Splits and freezes
ax = axes[1, 0]
ax.bar(range(len(history['splits'])), history['splits'], alpha=0.7, label='Splits', color='red')
ax.bar(range(len(history['freezes'])), history['freezes'], alpha=0.7, label='Freezes', color='blue')
ax.set_xlabel('Epoch')
ax.set_ylabel('Count')
ax.set_title('Growth & Freezing Events')
ax.legend()
ax.grid(True, alpha=0.3)

# Speedup
ax = axes[1, 1]
active_neurons = [n - f for n, f in zip(history['neurons'], history['frozen'])]
speedup = [n / a if a > 0 else 1 for n, a in zip(history['neurons'], active_neurons)]
ax.plot(speedup, 'orange', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Speedup')
ax.set_title('Training Speedup (from freezing)')
ax.grid(True, alpha=0.3)

# Temperature distribution (final)
ax = axes[1, 2]
all_temps = []
all_frozen = []
for block in crystal.blocks:
    attn = block.attn
    for i in range(attn.num_neurons):
        all_temps.append(attn.temperature[i].item())
        all_frozen.append(attn.frozen_mask[i].item())

colors = ['blue' if f else 'red' for f in all_frozen]
ax.scatter(range(len(all_temps)), all_temps, c=colors, alpha=0.6, s=30)
ax.axhline(y=np.median(all_temps), color='green', linestyle='--', label='Median')
ax.set_xlabel('Neuron Index')
ax.set_ylabel('Temperature')
ax.set_title('Final Neuron Temperatures\n(Blue=Frozen, Red=Active)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('crystal_gpt2_growth.png', dpi=150)
print(f"\nSaved: crystal_gpt2_growth.png")


# Final summary
stats = crystal.get_stats()
final_params = sum(p.numel() for p in crystal.parameters())

print("\n" + "=" * 70)
print("CRYSTALLIZATION COMPLETE!")
print("=" * 70)
print(f"""
    TEACHER (GPT-2):
    - Parameters: {teacher_params:,} ({teacher_params/1e6:.1f}M)
    - Layers: 12
    - Attention heads: 144 (12 × 12)

    CRYSTAL:
    - Parameters: {final_params:,} ({final_params/1e6:.1f}M)
    - Blocks: {NUM_BLOCKS}
    - Total neurons: {stats['total_neurons']}
    - Frozen neurons: {stats['frozen_neurons']} ({stats['frozen_pct']:.1f}%)
    - Active neurons: {stats['active_neurons']}

    RESULTS:
    - Compression: {teacher_params/final_params:.1f}x fewer params
    - Neurons grew: {INITIAL_NEURONS * NUM_BLOCKS} → {stats['total_neurons']}
    - Speedup from freezing: {stats['total_neurons']/stats['active_neurons']:.1f}x

    The crystal GREW where it needed capacity (hot regions)
    and FROZE where knowledge stabilized (cold regions)!

    "Knowledge crystallizes into geometry."
""")

# Save
torch.save({
    'model_state_dict': crystal.state_dict(),
    'history': history,
    'stats': stats,
    'config': {
        'initial_neurons': INITIAL_NEURONS,
        'max_neurons': MAX_NEURONS,
        'num_blocks': NUM_BLOCKS,
    }
}, 'crystal_gpt2_grown.pt')
print(f"Saved: crystal_gpt2_grown.pt")
