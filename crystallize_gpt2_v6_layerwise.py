#!/usr/bin/env python3
"""
Crystallize GPT-2 v6: LAYER-WISE DISTILLATION

Instead of just matching GPT-2's final output, match hidden states
at each layer. This preserves the hierarchical structure:
- Early layers: syntax, basic patterns
- Middle layers: phrase structure
- Late layers: semantics, context

GPT-2 has 12 layers, Crystal has 6 blocks.
We match every other GPT-2 layer to each crystal block.

"Distill the hierarchy, not just the output."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os
import time
from datetime import datetime

# Create output directory
RUN_NAME = f"crystal_layerwise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(f"runs/{RUN_NAME}", exist_ok=True)

print("=" * 70)
print("CRYSTALLIZE GPT-2 v6: LAYER-WISE DISTILLATION")
print(f"Output: runs/{RUN_NAME}/")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
INITIAL_NEURONS = 16      # Start small
MAX_NEURONS = 512         # Per block
NUM_BLOCKS = 6            # Crystal blocks (GPT-2 has 12, we use every other)
EMBED_DIM = 768
VOCAB_SIZE = 50257
SEQ_LEN = 128
BATCH_SIZE = 8
EPOCHS = 600
LR = 1e-4
VIZ_EVERY = 10

# Loss weights
LAYER_LOSS_WEIGHT = 1.0   # Weight for layer-wise matching
OUTPUT_LOSS_WEIGHT = 0.5  # Weight for final output matching

# Growth/freeze parameters
GROW_EVERY = 5
FREEZE_THRESHOLD_PCT = 20
MIN_COLD_EPOCHS = 3

# GPT-2 layers to match (every other layer: 1, 3, 5, 7, 9, 11)
GPT2_LAYER_INDICES = [1, 3, 5, 7, 9, 11]

print(f"""
Configuration:
  - Initial neurons: {INITIAL_NEURONS} per block ({INITIAL_NEURONS * NUM_BLOCKS} total)
  - Max neurons: {MAX_NEURONS} per block ({MAX_NEURONS * NUM_BLOCKS} total)
  - Blocks: {NUM_BLOCKS}
  - GPT-2 layers to match: {GPT2_LAYER_INDICES}
  - Layer loss weight: {LAYER_LOSS_WEIGHT}
  - Output loss weight: {OUTPUT_LOSS_WEIGHT}
  - Epochs: {EPOCHS}
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

    def update_stats(self):
        if self.positions.grad is not None:
            N = self.num_neurons
            grad_norm = self.positions.grad[:N].norm(dim=-1).detach()
            self.grad_ema[:N] = 0.9 * self.grad_ema[:N] + 0.1 * grad_norm
            self.temperature[:N] = self.grad_ema[:N]

    def get_hot_neurons(self, top_k=2):
        N = self.num_neurons
        temps = self.temperature[:N].clone()
        temps[self.frozen[:N]] = -1
        if (temps > 0).sum() < top_k:
            return []
        _, indices = temps.topk(min(top_k, N))
        return [i.item() for i in indices if temps[i] > 0]

    def get_cold_neurons(self, threshold_pct=20, min_cold=3):
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
        self.frozen[idx] = True


class CrystalBlock(nn.Module):
    """Single crystal block with attention and MLP."""
    def __init__(self, embed_dim, initial_neurons, max_neurons):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CrystalAttention(embed_dim, initial_neurons, max_neurons)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrystalGPT(nn.Module):
    """Crystal language model with layer-wise output access."""
    def __init__(self, vocab_size, embed_dim, initial_neurons, num_blocks, max_neurons):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(1024, embed_dim)

        self.blocks = nn.ModuleList([
            CrystalBlock(embed_dim, initial_neurons, max_neurons)
            for _ in range(num_blocks)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.wte.weight

    def forward(self, idx, return_hidden=False):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        hidden_states = []
        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hidden_states.append(x)

        logits = self.head(self.ln_f(x))

        if return_hidden:
            return logits, hidden_states
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
        total = sum(b.attn.num_neurons for b in self.blocks)
        frozen = sum(b.attn.frozen[:b.attn.num_neurons].sum().item() for b in self.blocks)
        return total, frozen, total - frozen

    def grow_and_freeze(self):
        grown = 0
        frozen = 0
        for block in self.blocks:
            attn = block.attn
            attn.update_stats()
            cold = attn.get_cold_neurons(FREEZE_THRESHOLD_PCT, MIN_COLD_EPOCHS)
            for idx in cold:
                attn.freeze(idx)
                frozen += 1
            hot = attn.get_hot_neurons(top_k=2)
            for idx in hot:
                if attn.split(idx):
                    grown += 1
        return grown, frozen


# ============================================================================
# GPT-2 WITH HIDDEN STATE ACCESS
# ============================================================================

class GPT2WithHidden(nn.Module):
    """Wrapper around GPT-2 to extract hidden states at specific layers."""
    def __init__(self, layer_indices):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.layer_indices = layer_indices

        # Freeze GPT-2
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        # Get all hidden states
        outputs = self.gpt2(input_ids, output_hidden_states=True)

        # Extract hidden states at specified layers
        # hidden_states[0] is embeddings, hidden_states[i] is after layer i-1
        all_hidden = outputs.hidden_states

        selected_hidden = [all_hidden[i + 1] for i in self.layer_indices]

        return outputs.logits, selected_hidden


# ============================================================================
# VISUALIZATION (same as v5)
# ============================================================================

def visualize_crystal(crystal, epoch, loss, pca, save_path):
    """Create visualization frame."""
    fig = plt.figure(figsize=(20, 12))

    all_positions = []
    all_frozen = []
    all_temps = []
    block_labels = []

    for block_idx, block in enumerate(crystal.blocks):
        attn = block.attn
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
        return pca

    all_positions = np.vstack(all_positions)
    all_frozen = np.concatenate(all_frozen)
    all_temps = np.concatenate(all_temps)
    block_labels = np.array(block_labels)

    total = len(all_frozen)
    frozen_count = all_frozen.sum()
    active_count = total - frozen_count
    frozen_pct = 100 * frozen_count / total if total > 0 else 0
    speedup = total / max(active_count, 1)

    if pca is None:
        pca = PCA(n_components=3)
        positions_3d = pca.fit_transform(all_positions)
    else:
        positions_3d = pca.transform(all_positions)

    # 1. 3D frozen vs active
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    colors = ['blue' if f else 'red' for f in all_frozen]
    sizes = [20 if f else 50 for f in all_frozen]
    ax1.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
               c=colors, s=sizes, alpha=0.6)
    ax1.set_title(f'Epoch {epoch}\nFrozen: {frozen_count} | Active: {active_count}')

    # 2. Temperature
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    scatter = ax2.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                         c=all_temps, cmap='coolwarm', s=30, alpha=0.7)
    ax2.set_title('Temperature (Gradient Activity)')
    plt.colorbar(scatter, ax=ax2, shrink=0.5)

    # 3. By layer
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    ax3.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
               c=block_labels, cmap='tab10', s=30, alpha=0.7)
    ax3.set_title(f'By Layer\nTotal: {total} neurons')

    # 4. 2D projection
    ax4 = fig.add_subplot(2, 4, 4)
    colors = ['blue' if f else 'red' for f in all_frozen]
    ax4.scatter(positions_3d[:, 0], positions_3d[:, 1], c=colors, s=20, alpha=0.6)
    ax4.set_title('2D Projection')

    # 5. Temperature distribution
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.hist(all_temps[all_frozen], bins=30, alpha=0.7, label='Frozen', color='blue')
    ax5.hist(all_temps[~all_frozen], bins=30, alpha=0.7, label='Active', color='red')
    ax5.set_title('Temperature Distribution')
    ax5.legend()

    # 6. Neurons per block
    ax6 = fig.add_subplot(2, 4, 6)
    block_totals = [b.attn.num_neurons for b in crystal.blocks]
    block_frozen = [b.attn.frozen[:b.attn.num_neurons].sum().item() for b in crystal.blocks]
    x = np.arange(len(block_totals))
    ax6.bar(x, block_totals, 0.35, label='Total', color='green', alpha=0.7)
    ax6.bar(x, block_frozen, 0.35, label='Frozen', color='blue', alpha=0.7)
    ax6.set_title('Neurons per Block')
    ax6.legend()

    # 7. Stats
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.axis('off')
    stats_text = f"""
LAYER-WISE DISTILLATION
Epoch {epoch}
{'='*25}

Total Neurons: {total}
Frozen: {frozen_count} ({frozen_pct:.1f}%)
Active: {active_count}
Speedup: {speedup:.1f}x
Loss: {loss:.4f}

Per Block (→ GPT-2 Layer):
"""
    for i, block in enumerate(crystal.blocks):
        f = block.attn.frozen[:block.attn.num_neurons].sum().item()
        stats_text += f"B{i}→L{GPT2_LAYER_INDICES[i]}: {block.attn.num_neurons}n, {f}f\n"

    ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace')

    # 8. Big display
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    ax8.text(0.5, 0.7, f'Epoch {epoch}', transform=ax8.transAxes, fontsize=24,
            ha='center', fontweight='bold')
    ax8.text(0.5, 0.5, f'Frozen: {frozen_pct:.1f}%', transform=ax8.transAxes, fontsize=20,
            ha='center', color='blue')
    ax8.text(0.5, 0.3, f'Speedup: {speedup:.1f}x', transform=ax8.transAxes, fontsize=20,
            ha='center', color='green')
    ax8.text(0.5, 0.1, 'LAYER-WISE', transform=ax8.transAxes, fontsize=14,
            ha='center', color='purple')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

    return pca


# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

print("\n[1] Loading GPT-2 teacher with hidden state access...")
teacher = GPT2WithHidden(GPT2_LAYER_INDICES).to(device)
teacher.eval()
print(f"    GPT-2 parameters: {sum(p.numel() for p in teacher.gpt2.parameters()):,}")
print(f"    Matching layers: {GPT2_LAYER_INDICES}")

print("\n[2] Loading WikiText-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

texts = []
for text in dataset['text']:
    if len(text.strip()) > 50:
        texts.append(text)
texts = texts[:2000]

print(f"    Loaded {len(texts)} texts")

print("\n[3] Creating Crystal...")
crystal = CrystalGPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    initial_neurons=INITIAL_NEURONS,
    num_blocks=NUM_BLOCKS,
    max_neurons=MAX_NEURONS
).to(device)

print(f"    Initial neurons: {INITIAL_NEURONS * NUM_BLOCKS}")

optimizer = torch.optim.AdamW(crystal.parameters(), lr=LR)

# ============================================================================
# TRAINING WITH LAYER-WISE DISTILLATION
# ============================================================================

print(f"\n[4] Training with layer-wise distillation ({EPOCHS} epochs)...")
print("=" * 70)

history = {'loss': [], 'layer_loss': [], 'output_loss': [],
           'neurons': [], 'frozen': [], 'active': [], 'speedup': []}
pca = None
start_time = time.time()

for epoch in range(EPOCHS):
    crystal.train()
    epoch_loss = 0
    epoch_layer_loss = 0
    epoch_output_loss = 0
    num_batches = 0

    np.random.shuffle(texts)

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        if len(batch_texts) < BATCH_SIZE:
            continue

        # Tokenize
        encoded = tokenizer(batch_texts, return_tensors='pt', padding=True,
                          truncation=True, max_length=SEQ_LEN)
        input_ids = encoded['input_ids'].to(device)

        if input_ids.shape[1] < 10:
            continue

        optimizer.zero_grad()

        # Teacher forward (with hidden states)
        with torch.no_grad():
            teacher_logits, teacher_hidden = teacher(input_ids)

        # Student forward (with hidden states)
        student_logits, student_hidden = crystal(input_ids, return_hidden=True)

        # Layer-wise loss: match hidden states at each layer
        layer_loss = 0
        for s_hidden, t_hidden in zip(student_hidden, teacher_hidden):
            layer_loss += F.mse_loss(s_hidden, t_hidden)
        layer_loss = layer_loss / len(student_hidden)

        # Output loss: match final logits (KL divergence)
        output_loss = F.kl_div(
            F.log_softmax(student_logits / 2.0, dim=-1),
            F.softmax(teacher_logits / 2.0, dim=-1),
            reduction='batchmean'
        ) * (2.0 ** 2)

        # Combined loss
        loss = LAYER_LOSS_WEIGHT * layer_loss + OUTPUT_LOSS_WEIGHT * output_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(crystal.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_layer_loss += layer_loss.item()
        epoch_output_loss += output_loss.item()
        num_batches += 1

    avg_loss = epoch_loss / max(num_batches, 1)
    avg_layer_loss = epoch_layer_loss / max(num_batches, 1)
    avg_output_loss = epoch_output_loss / max(num_batches, 1)

    # Growth and freezing
    if (epoch + 1) % GROW_EVERY == 0:
        crystal.grow_and_freeze()

    # Stats
    total, frozen, active = crystal.get_stats()
    speedup = total / max(active, 1)

    history['loss'].append(avg_loss)
    history['layer_loss'].append(avg_layer_loss)
    history['output_loss'].append(avg_output_loss)
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
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} (L:{avg_layer_loss:.3f} O:{avg_output_loss:.3f}) | "
              f"N: {total} | F: {frozen} ({frozen_pct:.0f}%) | S: {speedup:.1f}x | T: {elapsed:.1f}m")

        crystal.eval()
        prompt = tokenizer.encode("The meaning of life", return_tensors='pt').to(device)
        output = crystal.generate(prompt, max_new_tokens=30, temperature=0.7)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"         -> \"{text[:80]}...\"")
        print()

# ============================================================================
# FINAL VISUALIZATION AND SAVE
# ============================================================================

print("\n[5] Creating final visualization...")
pca = visualize_crystal(crystal, EPOCHS, history['loss'][-1], pca,
                        f"runs/{RUN_NAME}/epoch_{EPOCHS:03d}_final.png")

# Summary plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].plot(history['loss'], 'b-', label='Total', linewidth=2)
axes[0, 0].plot(history['layer_loss'], 'g--', label='Layer', linewidth=1)
axes[0, 0].plot(history['output_loss'], 'r--', label='Output', linewidth=1)
axes[0, 0].set_title('Loss (Layer-wise + Output)')
axes[0, 0].legend()

axes[0, 1].plot(history['neurons'], 'g-', label='Total', linewidth=2)
axes[0, 1].plot(history['frozen'], 'b-', label='Frozen', linewidth=2)
axes[0, 1].plot(history['active'], 'r-', label='Active', linewidth=2)
axes[0, 1].set_title('Crystal Growth & Freezing')
axes[0, 1].legend()

crystal_pct = [100 * f / n if n > 0 else 0 for f, n in zip(history['frozen'], history['neurons'])]
axes[0, 2].plot(crystal_pct, 'm-', linewidth=2)
axes[0, 2].set_title('Crystallization Progress')

axes[1, 0].plot(history['speedup'], 'orange', linewidth=2)
axes[1, 0].set_title('Training Speedup')

# Final structure
total, frozen, active = crystal.get_stats()
all_positions = []
all_frozen_arr = []
for block in crystal.blocks:
    attn = block.attn
    N = attn.num_neurons
    all_positions.append(attn.positions[:N].detach().cpu().numpy())
    all_frozen_arr.append(attn.frozen[:N].detach().cpu().numpy())

all_positions = np.vstack(all_positions)
all_frozen_arr = np.concatenate(all_frozen_arr)

if pca is not None:
    positions_3d = pca.transform(all_positions)
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    colors = ['blue' if f else 'red' for f in all_frozen_arr]
    ax.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
              c=colors, s=20, alpha=0.6)
    ax.set_title(f'Final Structure\n{total} neurons, {frozen} frozen')

# Stats
axes[1, 2].axis('off')
elapsed = (time.time() - start_time) / 60
stats_text = f"""
LAYER-WISE DISTILLATION COMPLETE
{'='*35}

Training: {EPOCHS} epochs, {elapsed:.1f} minutes

GPT-2 Layers Matched: {GPT2_LAYER_INDICES}

GROWTH:
  Started: {INITIAL_NEURONS * NUM_BLOCKS} neurons
  Final: {total} neurons
  Growth: {total / (INITIAL_NEURONS * NUM_BLOCKS):.1f}x

CRYSTALLIZATION:
  Frozen: {frozen} ({100*frozen/total:.1f}%)
  Active: {active}
  Speedup: {history['speedup'][-1]:.1f}x

LOSS:
  Initial: {history['loss'][0]:.3f}
  Final: {history['loss'][-1]:.3f}
  Layer: {history['layer_loss'][-1]:.3f}
  Output: {history['output_loss'][-1]:.3f}

"Hierarchy preserved in geometry."
"""
axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(f"runs/{RUN_NAME}/summary.png", dpi=150)
print(f"    Saved: runs/{RUN_NAME}/summary.png")

# Save model
print("\n[6] Saving model...")
torch.save({
    'model': crystal.state_dict(),
    'history': history,
    'config': {
        'neurons': INITIAL_NEURONS,
        'blocks': NUM_BLOCKS,
        'max': MAX_NEURONS,
        'layer_indices': GPT2_LAYER_INDICES
    }
}, f"runs/{RUN_NAME}/crystal_final.pt")

# ============================================================================
# GENERATION TEST
# ============================================================================

print("\n" + "=" * 70)
print("[7] GENERATION TEST")
print("=" * 70)

crystal.eval()
prompts = [
    "The meaning of life is",
    "Neural networks learn",
    "Science has shown that",
    "The president said",
    "In the beginning,",
]

print("\nCrystal Generation (layer-wise distillation):\n")
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
print("LAYER-WISE DISTILLATION COMPLETE!")
print("=" * 70)
print(f"""
    Training time: {elapsed:.1f} minutes

    LAYER MATCHING:
    - GPT-2 layers: {GPT2_LAYER_INDICES}
    - Crystal blocks: [0, 1, 2, 3, 4, 5]

    CRYSTAL STATS:
    - Total neurons: {total}
    - Frozen: {frozen} ({100*frozen/total:.1f}%)
    - Active: {active}
    - Speedup: {history['speedup'][-1]:.1f}x

    LOSS BREAKDOWN:
    - Layer loss: {history['layer_loss'][-1]:.4f}
    - Output loss: {history['output_loss'][-1]:.4f}

    Output: runs/{RUN_NAME}/

    "Each crystal layer learned from its GPT-2 counterpart."
    "Hierarchy preserved in geometry."
""")
