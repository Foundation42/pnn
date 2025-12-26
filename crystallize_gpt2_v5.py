#!/usr/bin/env python3
"""
Crystallize GPT-2 v5: GROW FROM SCRATCH

Instead of distilling from GPT-2, grow the crystal directly on WikiText-2.
Hypothesis: A model that learns the task directly will develop better
semantic coherence than one mimicking another model's outputs.

Key differences from v4:
- No teacher model - direct language modeling loss
- Crystal learns its own representations
- Should develop more organic structure
- Expected to be more efficient

"Let the crystal grow its own intelligence."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os
import time
from datetime import datetime

# Create output directory
RUN_NAME = f"crystal_scratch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(f"runs/{RUN_NAME}", exist_ok=True)

print("=" * 70)
print("CRYSTALLIZE GPT-2 v5: GROW FROM SCRATCH")
print(f"Output: runs/{RUN_NAME}/")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
INITIAL_NEURONS = 16      # Start small
MAX_NEURONS = 512         # Per block - based on v4 findings
NUM_BLOCKS = 6
EMBED_DIM = 768
VOCAB_SIZE = 50257
SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 600
LR = 3e-4
VIZ_EVERY = 10            # Visualization every N epochs

# Growth/freeze parameters
GROW_EVERY = 5            # Check for growth every N epochs
FREEZE_THRESHOLD_PCT = 20 # Bottom 20% by temperature
MIN_COLD_EPOCHS = 3       # Must be cold for 3 epochs to freeze

print(f"""
Configuration:
  - Initial neurons: {INITIAL_NEURONS} per block ({INITIAL_NEURONS * NUM_BLOCKS} total)
  - Max neurons: {MAX_NEURONS} per block ({MAX_NEURONS * NUM_BLOCKS} total)
  - Blocks: {NUM_BLOCKS}
  - Embed dim: {EMBED_DIM}
  - Sequence length: {SEQ_LEN}
  - Batch size: {BATCH_SIZE}
  - Epochs: {EPOCHS}
  - Learning rate: {LR}
""")

# ============================================================================
# CRYSTAL ARCHITECTURE
# ============================================================================

class CrystalAttention(nn.Module):
    """Geometric attention with growth and freezing."""
    def __init__(self, embed_dim, initial_neurons, max_neurons):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_neurons = max_neurons
        self.num_neurons = initial_neurons

        # Geometric parameters
        self.positions = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.02)
        self.scales = nn.Parameter(torch.ones(max_neurons) * 5.0)
        self.values = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.02)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # State tracking
        self.register_buffer('frozen', torch.zeros(max_neurons, dtype=torch.bool))
        self.register_buffer('temperature', torch.ones(max_neurons))
        self.register_buffer('grad_ema', torch.zeros(max_neurons))
        self.register_buffer('cold_epochs', torch.zeros(max_neurons))

    def forward(self, x):
        B, T, D = x.shape
        N = self.num_neurons

        # Geometric attention via RBF
        pos = self.positions[:N].unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)
        x_exp = x.unsqueeze(2)  # (B, T, 1, D)
        dist = torch.norm(x_exp - pos, dim=-1)  # (B, T, N)

        attn = F.softmax(self.scales[:N] / (dist + 0.1), dim=-1)
        out = torch.einsum('btn,nd->btd', attn, self.values[:N])

        return self.out_proj(out)

    def update_stats(self):
        """Update temperature based on gradient activity."""
        if self.positions.grad is not None:
            N = self.num_neurons
            grad_norm = self.positions.grad[:N].norm(dim=-1).detach()
            self.grad_ema[:N] = 0.9 * self.grad_ema[:N] + 0.1 * grad_norm
            self.temperature[:N] = self.grad_ema[:N]

    def get_hot_neurons(self, top_k=2):
        """Find neurons with highest gradient activity (candidates for split)."""
        N = self.num_neurons
        temps = self.temperature[:N].clone()
        temps[self.frozen[:N]] = -1  # Exclude frozen
        if (temps > 0).sum() < top_k:
            return []
        _, indices = temps.topk(min(top_k, N))
        return [i.item() for i in indices if temps[i] > 0]

    def get_cold_neurons(self, threshold_pct=20, min_cold=3):
        """Find neurons with lowest gradient activity (candidates for freeze)."""
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
        """Split a hot neuron into two."""
        if self.num_neurons >= self.max_neurons:
            return False

        new = self.num_neurons
        with torch.no_grad():
            noise = 0.01
            self.positions.data[new] = self.positions.data[idx] + \
                torch.randn(self.embed_dim, device=self.positions.device) * noise
            self.scales.data[new] = self.scales.data[idx]
            self.values.data[new] = self.values.data[idx] + \
                torch.randn(self.embed_dim, device=self.values.device) * noise
            self.temperature[new] = self.temperature[idx]
            self.cold_epochs[new] = 0

        self.num_neurons += 1
        return True

    def freeze(self, idx):
        """Freeze a cold neuron."""
        self.frozen[idx] = True


class CrystalGPT(nn.Module):
    """Crystal language model - grown from scratch."""
    def __init__(self, vocab_size, embed_dim, initial_neurons, num_blocks, max_neurons):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        # Embeddings
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(1024, embed_dim)

        # Crystal blocks
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
        self.head.weight = self.wte.weight  # Weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = x + block['attn'](block['ln1'](x))
            x = x + block['mlp'](block['ln2'](x))

        logits = self.head(self.ln_f(x))

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

        return logits

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

    def get_stats(self):
        """Get crystal statistics."""
        total = sum(b['attn'].num_neurons for b in self.blocks)
        frozen = sum(b['attn'].frozen[:b['attn'].num_neurons].sum().item() for b in self.blocks)
        return total, frozen, total - frozen

    def grow_and_freeze(self):
        """Perform growth and freezing operations."""
        grown = 0
        frozen = 0

        for block in self.blocks:
            attn = block['attn']
            attn.update_stats()

            # Freeze cold neurons
            cold = attn.get_cold_neurons(FREEZE_THRESHOLD_PCT, MIN_COLD_EPOCHS)
            for idx in cold:
                attn.freeze(idx)
                frozen += 1

            # Split hot neurons
            hot = attn.get_hot_neurons(top_k=2)
            for idx in hot:
                if attn.split(idx):
                    grown += 1

        return grown, frozen


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_crystal(crystal, epoch, loss, pca, save_path):
    """Create visualization frame."""
    fig = plt.figure(figsize=(20, 12))

    # Collect neuron data
    all_positions = []
    all_frozen = []
    all_temps = []
    block_labels = []

    for block_idx, block in enumerate(crystal.blocks):
        attn = block['attn']
        N = attn.num_neurons
        positions = attn.positions[:N].detach().cpu().numpy()
        frozen = attn.frozen[:N].detach().cpu().numpy()
        temps = attn.temperature[:N].detach().cpu().numpy()

        all_positions.append(positions)
        all_frozen.append(frozen)
        all_temps.append(temps)
        block_labels.extend([block_idx] * N)

    if len(all_positions) == 0:
        plt.close()
        return

    all_positions = np.vstack(all_positions)
    all_frozen = np.concatenate(all_frozen)
    all_temps = np.concatenate(all_temps)
    block_labels = np.array(block_labels)

    total = len(all_frozen)
    frozen_count = all_frozen.sum()
    active_count = total - frozen_count
    frozen_pct = 100 * frozen_count / total if total > 0 else 0
    speedup = total / max(active_count, 1)

    # PCA projection
    if pca is None:
        pca = PCA(n_components=3)
        positions_3d = pca.fit_transform(all_positions)
    else:
        positions_3d = pca.transform(all_positions)

    # 1. 3D scatter - frozen vs active
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    colors = ['blue' if f else 'red' for f in all_frozen]
    sizes = [20 if f else 50 for f in all_frozen]
    ax1.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
               c=colors, s=sizes, alpha=0.6)
    ax1.set_title(f'Epoch {epoch}\nFrozen: {frozen_count} | Active: {active_count}')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')

    # 2. Temperature view
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    scatter = ax2.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                         c=all_temps, cmap='coolwarm', s=30, alpha=0.7)
    ax2.set_title('Temperature (Gradient Activity)\nHot=Red, Cold=Blue')
    plt.colorbar(scatter, ax=ax2, shrink=0.5)

    # 3. By layer
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    scatter = ax3.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                         c=block_labels, cmap='tab10', s=30, alpha=0.7)
    ax3.set_title(f'By Layer (Block)\nTotal: {total} neurons')

    # 4. 2D projection
    ax4 = fig.add_subplot(2, 4, 4)
    colors = ['blue' if f else 'red' for f in all_frozen]
    ax4.scatter(positions_3d[:, 0], positions_3d[:, 1], c=colors, s=20, alpha=0.6)
    ax4.set_title('2D Projection (PC1 vs PC2)')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')

    # 5. Temperature distribution
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.hist(all_temps[all_frozen], bins=30, alpha=0.7, label='Frozen', color='blue')
    ax5.hist(all_temps[~all_frozen], bins=30, alpha=0.7, label='Active', color='red')
    ax5.set_xlabel('Temperature')
    ax5.set_ylabel('Count')
    ax5.set_title('Temperature Distribution')
    ax5.legend()

    # 6. Neurons per block
    ax6 = fig.add_subplot(2, 4, 6)
    block_totals = []
    block_frozen = []
    for block in crystal.blocks:
        attn = block['attn']
        block_totals.append(attn.num_neurons)
        block_frozen.append(attn.frozen[:attn.num_neurons].sum().item())

    x = np.arange(len(block_totals))
    width = 0.35
    ax6.bar(x, block_totals, width, label='Total', color='green', alpha=0.7)
    ax6.bar(x, block_frozen, width, label='Frozen', color='blue', alpha=0.7)
    ax6.set_xlabel('Block')
    ax6.set_ylabel('Neurons')
    ax6.set_title('Neurons per Block')
    ax6.legend()

    # 7. Stats text
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.axis('off')
    stats_text = f"""
CRYSTAL STATS (Epoch {epoch})
{'='*30}

Total Neurons: {total}
Frozen: {frozen_count} ({frozen_pct:.1f}%)
Active: {active_count} ({100-frozen_pct:.1f}%)

Speedup: {speedup:.1f}x
Loss: {loss:.4f}

Per Block:
"""
    for i, block in enumerate(crystal.blocks):
        attn = block['attn']
        f = attn.frozen[:attn.num_neurons].sum().item()
        stats_text += f"Block {i}: {attn.num_neurons} neurons, {f} frozen\n"

    ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    # 8. Big stats display
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    ax8.text(0.5, 0.6, f'Epoch {epoch}', transform=ax8.transAxes, fontsize=24,
            ha='center', fontweight='bold')
    ax8.text(0.5, 0.4, f'Frozen: {frozen_pct:.1f}%', transform=ax8.transAxes, fontsize=20,
            ha='center', color='blue')
    ax8.text(0.5, 0.2, f'Speedup: {speedup:.1f}x', transform=ax8.transAxes, fontsize=20,
            ha='center', color='green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

    return pca


# ============================================================================
# DATA LOADING
# ============================================================================

print("\n[1] Loading WikiText-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

# Tokenize and create sequences
all_tokens = []
for text in dataset['text'][:5000]:  # Use more data for from-scratch training
    if len(text.strip()) > 50:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

# Create training sequences
sequences = []
for i in range(0, len(all_tokens) - SEQ_LEN - 1, SEQ_LEN // 2):
    seq = all_tokens[i:i + SEQ_LEN + 1]
    if len(seq) == SEQ_LEN + 1:
        sequences.append(seq)

print(f"    Created {len(sequences)} training sequences")

# ============================================================================
# CREATE MODEL
# ============================================================================

print("\n[2] Creating Crystal (from scratch)...")
crystal = CrystalGPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    initial_neurons=INITIAL_NEURONS,
    num_blocks=NUM_BLOCKS,
    max_neurons=MAX_NEURONS
).to(device)

total_params = sum(p.numel() for p in crystal.parameters())
print(f"    Total parameters: {total_params:,}")
print(f"    Initial neurons: {INITIAL_NEURONS * NUM_BLOCKS}")

# Optimizer - only optimize non-frozen parameters
optimizer = torch.optim.AdamW(crystal.parameters(), lr=LR)

# ============================================================================
# TRAINING
# ============================================================================

print(f"\n[3] Training Crystal ({EPOCHS} epochs)...")
print("=" * 70)

history = {'loss': [], 'neurons': [], 'frozen': [], 'active': [], 'speedup': []}
pca = None
start_time = time.time()

for epoch in range(EPOCHS):
    crystal.train()
    epoch_loss = 0
    num_batches = 0

    # Shuffle sequences
    np.random.shuffle(sequences)

    for i in range(0, len(sequences), BATCH_SIZE):
        batch_seqs = sequences[i:i + BATCH_SIZE]
        if len(batch_seqs) < BATCH_SIZE:
            continue

        batch = torch.tensor(batch_seqs, device=device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        optimizer.zero_grad()

        # Forward pass - only compute gradients for non-frozen
        logits, loss = crystal(inputs, targets)

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(crystal.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)

    # Growth and freezing
    if (epoch + 1) % GROW_EVERY == 0:
        grown, frozen_count = crystal.grow_and_freeze()

    # Get stats
    total, frozen, active = crystal.get_stats()
    speedup = total / max(active, 1)

    # Record history
    history['loss'].append(avg_loss)
    history['neurons'].append(total)
    history['frozen'].append(frozen)
    history['active'].append(active)
    history['speedup'].append(speedup)

    # Visualization
    if (epoch + 1) % VIZ_EVERY == 0 or epoch == 0:
        pca = visualize_crystal(crystal, epoch + 1, avg_loss, pca,
                               f"runs/{RUN_NAME}/epoch_{epoch+1:03d}.png")

    # Progress
    elapsed = (time.time() - start_time) / 60
    frozen_pct = 100 * frozen / total if total > 0 else 0

    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Neurons: {total} | "
              f"Frozen: {frozen} ({frozen_pct:.0f}%) | Speedup: {speedup:.1f}x | Time: {elapsed:.1f}m")

        # Generate sample
        crystal.eval()
        prompt = tokenizer.encode("The meaning of life", return_tensors='pt').to(device)
        output = crystal.generate(prompt, max_new_tokens=30, temperature=0.7)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"         -> \"{text[:80]}...\"")
        print()

# ============================================================================
# FINAL VISUALIZATION
# ============================================================================

print("\n[4] Creating final visualization...")
pca = visualize_crystal(crystal, EPOCHS, history['loss'][-1], pca,
                        f"runs/{RUN_NAME}/epoch_{EPOCHS:03d}_final.png")

# Summary plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Loss
axes[0, 0].plot(history['loss'], 'b-', linewidth=1)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')

# Growth
axes[0, 1].plot(history['neurons'], 'g-', label='Total', linewidth=2)
axes[0, 1].plot(history['frozen'], 'b-', label='Frozen', linewidth=2)
axes[0, 1].plot(history['active'], 'r-', label='Active', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Neurons')
axes[0, 1].set_title('Crystal Growth & Freezing')
axes[0, 1].legend()

# Crystallization
crystal_pct = [100 * f / n if n > 0 else 0 for f, n in zip(history['frozen'], history['neurons'])]
axes[0, 2].plot(crystal_pct, 'm-', linewidth=2)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Frozen %')
axes[0, 2].set_title('Crystallization Progress')

# Speedup
axes[1, 0].plot(history['speedup'], 'orange', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Speedup')
axes[1, 0].set_title('Training Speedup (from freezing)')

# Final crystal structure
total, frozen, active = crystal.get_stats()
all_positions = []
all_frozen = []
for block in crystal.blocks:
    attn = block['attn']
    N = attn.num_neurons
    all_positions.append(attn.positions[:N].detach().cpu().numpy())
    all_frozen.append(attn.frozen[:N].detach().cpu().numpy())

all_positions = np.vstack(all_positions)
all_frozen = np.concatenate(all_frozen)

if pca is not None:
    positions_3d = pca.transform(all_positions)
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    colors = ['blue' if f else 'red' for f in all_frozen]
    ax.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
              c=colors, s=20, alpha=0.6)
    ax.set_title(f'Final Crystal Structure\n{total} neurons, {frozen} frozen')

# Stats text
axes[1, 2].axis('off')
elapsed = (time.time() - start_time) / 60
stats_text = f"""
CRYSTALLIZATION COMPLETE
{'='*30}

Training: {EPOCHS} epochs, {elapsed:.1f} minutes

GROWTH:
  Started: {INITIAL_NEURONS * NUM_BLOCKS} neurons
  Final: {total} neurons
  Growth: {total / (INITIAL_NEURONS * NUM_BLOCKS):.1f}x

CRYSTALLIZATION:
  Frozen: {frozen} ({100*frozen/total:.1f}%)
  Active: {active} ({100*active/total:.1f}%)
  Speedup: {history['speedup'][-1]:.1f}x

LEARNING:
  Initial Loss: {history['loss'][0]:.3f}
  Final Loss: {history['loss'][-1]:.3f}

"Intelligence grown from scratch."
"""
axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(f"runs/{RUN_NAME}/summary.png", dpi=150)
print(f"    Saved: runs/{RUN_NAME}/summary.png")

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n[5] Saving model...")
torch.save({
    'model': crystal.state_dict(),
    'history': history,
    'config': {
        'neurons': INITIAL_NEURONS,
        'blocks': NUM_BLOCKS,
        'max': MAX_NEURONS,
        'embed_dim': EMBED_DIM,
        'vocab_size': VOCAB_SIZE
    }
}, f"runs/{RUN_NAME}/crystal_final.pt")
print(f"    Saved: runs/{RUN_NAME}/crystal_final.pt")

# ============================================================================
# FINAL GENERATION TEST
# ============================================================================

print("\n" + "=" * 70)
print("[6] GENERATION TEST")
print("=" * 70)

crystal.eval()
prompts = [
    "The meaning of life is",
    "Neural networks learn",
    "Science has shown that",
    "In the beginning,",
    "The president said",
]

print("\nCrystal Generation (temperature=0.7):\n")
for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = crystal.generate(input_ids, max_new_tokens=40, temperature=0.7)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  {text}")
    print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("CRYSTALLIZATION COMPLETE!")
print("=" * 70)
print(f"""
    Training time: {elapsed:.1f} minutes

    CRYSTAL STATS:
    - Total neurons: {total}
    - Frozen neurons: {frozen} ({100*frozen/total:.1f}%)
    - Active neurons: {active}
    - Training speedup: {history['speedup'][-1]:.1f}x

    GROWTH:
    - Started: {INITIAL_NEURONS * NUM_BLOCKS} neurons
    - Final: {total} neurons
    - Growth: {total / (INITIAL_NEURONS * NUM_BLOCKS):.1f}x

    LEARNING:
    - Initial loss: {history['loss'][0]:.4f}
    - Final loss: {history['loss'][-1]:.4f}

    Output: runs/{RUN_NAME}/

    "Intelligence grown from scratch into geometry."
""")
