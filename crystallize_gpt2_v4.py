#!/usr/bin/env python3
"""
Crystallize GPT-2 v4: VISUALIZATION RUN

Clean training with progressive imagery:
- Visualize crystal growth every N epochs
- No hard neuron cap (let it find natural size)
- Track cluster formation over time
- Generate animation-ready frames

"Watch the crystal grow, freeze, and organize."
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
from sklearn.cluster import KMeans
import os
import time
from datetime import datetime

# Create output directory
RUN_NAME = f"crystal_gpt2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(f"runs/{RUN_NAME}", exist_ok=True)

print("=" * 70)
print("CRYSTALLIZE GPT-2 v4: VISUALIZATION RUN")
print(f"Output: runs/{RUN_NAME}/")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# CRYSTAL ARCHITECTURE (No hard cap!)
# ============================================================================

class CrystalAttention(nn.Module):
    """Geometric attention with unlimited growth potential."""
    def __init__(self, embed_dim, initial_neurons, max_neurons=2048):  # Higher cap!
        super().__init__()
        self.embed_dim = embed_dim
        self.max_neurons = max_neurons
        self.num_neurons = initial_neurons

        # Pre-allocate for max size
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

    def grow_and_freeze(self, splits_per_block=2, freezes_per_block=4):
        splits, freezes = 0, 0
        for block in self.blocks:
            attn = block['attn']
            for idx in attn.get_cold_neurons()[:freezes_per_block]:
                attn.freeze(idx)
                freezes += 1
            for idx in attn.get_hot_neurons(splits_per_block):
                if attn.split(idx):
                    splits += 1
        return splits, freezes

    def stats(self):
        total = sum(b['attn'].num_neurons for b in self.blocks)
        frozen = sum(b['attn'].frozen[:b['attn'].num_neurons].sum().item() for b in self.blocks)
        return {'neurons': total, 'frozen': frozen, 'active': total - frozen}

    def get_all_positions(self):
        """Get positions and frozen status for visualization."""
        positions = []
        frozen_mask = []
        block_ids = []
        temps = []
        for bid, block in enumerate(self.blocks):
            attn = block['attn']
            n = attn.num_neurons
            positions.append(attn.positions[:n].detach().cpu().numpy())
            frozen_mask.append(attn.frozen[:n].detach().cpu().numpy())
            block_ids.extend([bid] * n)
            temps.append(attn.temperature[:n].detach().cpu().numpy())
        return {
            'positions': np.vstack(positions),
            'frozen': np.concatenate(frozen_mask),
            'blocks': np.array(block_ids),
            'temps': np.concatenate(temps)
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new=30, temperature=0.8, top_k=40):
        for _ in range(max_new):
            logits = self(input_ids[:, -512:])[:, -1, :]
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
        return input_ids


def visualize_crystal(crystal, epoch, loss, pca, save_path):
    """Create visualization frame for current state."""
    data = crystal.get_all_positions()
    stats = crystal.stats()

    # Project to 3D
    pos_3d = pca.transform(data['positions'])

    fig = plt.figure(figsize=(20, 10))

    # 3D scatter - frozen vs active
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    colors = ['blue' if f else 'red' for f in data['frozen']]
    sizes = [10 if f else 30 for f in data['frozen']]
    ax1.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], c=colors, s=sizes, alpha=0.6)
    ax1.set_title(f'Epoch {epoch}\nFrozen: {stats["frozen"]} | Active: {stats["active"]}')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')

    # 3D scatter - by temperature
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    scatter = ax2.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2],
                         c=data['temps'], cmap='coolwarm', s=20, alpha=0.6)
    ax2.set_title(f'Temperature (Gradient Activity)\nHot=Red, Cold=Blue')
    plt.colorbar(scatter, ax=ax2, shrink=0.5)

    # 3D scatter - by block
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    scatter = ax3.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2],
                         c=data['blocks'], cmap='tab10', s=20, alpha=0.6)
    ax3.set_title(f'By Layer (Block)\nTotal: {stats["neurons"]} neurons')

    # 2D projection (top view)
    ax4 = fig.add_subplot(2, 4, 4)
    colors = ['blue' if f else 'red' for f in data['frozen']]
    ax4.scatter(pos_3d[:, 0], pos_3d[:, 1], c=colors, s=10, alpha=0.5)
    ax4.set_title('2D Projection (PC1 vs PC2)')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')

    # Stats text
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.axis('off')
    frozen_pct = 100 * stats['frozen'] / stats['neurons'] if stats['neurons'] > 0 else 0
    speedup = stats['neurons'] / stats['active'] if stats['active'] > 0 else 1
    text = f"""
    CRYSTAL STATS (Epoch {epoch})
    ═══════════════════════════

    Total Neurons: {stats['neurons']}
    Frozen: {stats['frozen']} ({frozen_pct:.1f}%)
    Active: {stats['active']} ({100-frozen_pct:.1f}%)

    Speedup: {speedup:.1f}x
    Loss: {loss:.4f}

    Per Block:
    """
    for i, block in enumerate(crystal.blocks):
        attn = block['attn']
        n = attn.num_neurons
        f = attn.frozen[:n].sum().item()
        text += f"\n    Block {i}: {n} neurons, {f} frozen"

    ax5.text(0.1, 0.9, text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')

    # Temperature histogram
    ax6 = fig.add_subplot(2, 4, 6)
    frozen_temps = data['temps'][data['frozen']]
    active_temps = data['temps'][~data['frozen']]
    if len(frozen_temps) > 0:
        ax6.hist(frozen_temps, bins=30, alpha=0.7, label='Frozen', color='blue')
    if len(active_temps) > 0:
        ax6.hist(active_temps, bins=30, alpha=0.7, label='Active', color='red')
    ax6.set_xlabel('Temperature')
    ax6.set_ylabel('Count')
    ax6.set_title('Temperature Distribution')
    ax6.legend()

    # Neurons by block
    ax7 = fig.add_subplot(2, 4, 7)
    block_neurons = [b['attn'].num_neurons for b in crystal.blocks]
    block_frozen = [b['attn'].frozen[:b['attn'].num_neurons].sum().item() for b in crystal.blocks]
    x = np.arange(len(block_neurons))
    ax7.bar(x, block_neurons, label='Total', alpha=0.7, color='green')
    ax7.bar(x, block_frozen, label='Frozen', alpha=0.7, color='blue')
    ax7.set_xlabel('Block')
    ax7.set_ylabel('Neurons')
    ax7.set_title('Neurons per Block')
    ax7.legend()

    # Frozen percentage over time (will be filled in later with history)
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.text(0.5, 0.5, f'Epoch {epoch}\n\nFrozen: {frozen_pct:.1f}%\nSpeedup: {speedup:.1f}x',
            ha='center', va='center', fontsize=20, transform=ax8.transAxes)
    ax8.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


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
INITIAL_NEURONS = 16
NUM_BLOCKS = 6
MAX_NEURONS = 512  # Per block - higher limit!

crystal = CrystalGPT(
    vocab_size=50257,
    embed_dim=768,
    initial_neurons=INITIAL_NEURONS,
    num_blocks=NUM_BLOCKS,
    max_neurons=MAX_NEURONS
).to(device)

# Copy embeddings
crystal.wte.weight.data = teacher.transformer.wte.weight.data.clone()
crystal.wpe.weight.data = teacher.transformer.wpe.weight.data.clone()

initial_total = INITIAL_NEURONS * NUM_BLOCKS
max_total = MAX_NEURONS * NUM_BLOCKS
print(f"    Initial: {initial_total} neurons ({INITIAL_NEURONS} × {NUM_BLOCKS})")
print(f"    Max: {max_total} neurons ({MAX_NEURONS} × {NUM_BLOCKS})")

# Initialize PCA with random positions for consistent projection
print("\n[4] Initializing PCA...")
init_data = crystal.get_all_positions()
pca = PCA(n_components=3)
pca.fit(init_data['positions'])

print(f"\n[5] Training Crystal ({RUN_NAME})...")
print("=" * 70)

optimizer = torch.optim.AdamW(crystal.parameters(), lr=3e-4, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

EPOCHS = 600
BATCH = 8
MAXLEN = 128
VIZ_EVERY = 10  # Save visualization every N epochs

history = {'loss': [], 'neurons': [], 'frozen': [], 'active': [], 'speedup': []}
start_time = time.time()

# Save initial state
visualize_crystal(crystal, 0, 0.0, pca, f"runs/{RUN_NAME}/epoch_000.png")

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

        with torch.no_grad():
            t_logits = teacher(**inputs).logits

        c_logits = crystal(inputs['input_ids'])

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
        splits, freezes = crystal.grow_and_freeze(splits_per_block=2, freezes_per_block=4)

    # Record stats
    stats = crystal.stats()
    avg_loss = epoch_loss / max(n_batch, 1)
    frozen_pct = 100 * stats['frozen'] / stats['neurons'] if stats['neurons'] > 0 else 0
    speedup = stats['neurons'] / stats['active'] if stats['active'] > 0 else 1

    history['loss'].append(avg_loss)
    history['neurons'].append(stats['neurons'])
    history['frozen'].append(stats['frozen'])
    history['active'].append(stats['active'])
    history['speedup'].append(speedup)

    # Progress report
    elapsed = time.time() - start_time
    if (epoch + 1) % 25 == 0 or epoch < 10 or splits > 0 or freezes > 0:
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.3f} | "
              f"Neurons: {stats['neurons']:4d} | Frozen: {stats['frozen']:4d} ({frozen_pct:.0f}%) | "
              f"Speedup: {speedup:.1f}x | +{splits}/-{freezes} | {elapsed/60:.1f}m")

    # Save visualization
    if (epoch + 1) % VIZ_EVERY == 0 or epoch == 0:
        # Refit PCA if neurons have grown significantly
        if stats['neurons'] > len(pca.components_[0]) * 1.5:
            data = crystal.get_all_positions()
            pca.fit(data['positions'])

        visualize_crystal(crystal, epoch + 1, avg_loss, pca,
                         f"runs/{RUN_NAME}/epoch_{epoch+1:03d}.png")

        # Also generate a sample
        crystal.eval()
        prompt = "The meaning of life is"
        inp = tokenizer(prompt, return_tensors='pt').to(device)
        out = crystal.generate(inp['input_ids'], max_new=25, temperature=0.7)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"         → \"{text[:70]}...\"")

# ============================================================================
# FINAL VISUALIZATION
# ============================================================================

print("\n" + "=" * 70)
print("[6] Final Visualization")
print("=" * 70)

# Save final state
visualize_crystal(crystal, EPOCHS, history['loss'][-1], pca,
                 f"runs/{RUN_NAME}/epoch_{EPOCHS:03d}_final.png")

# Create summary plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Loss
ax = axes[0, 0]
ax.plot(history['loss'], 'b-', alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Distillation Loss')
ax.grid(True, alpha=0.3)

# Growth
ax = axes[0, 1]
ax.plot(history['neurons'], 'g-', label='Total', linewidth=2)
ax.plot(history['frozen'], 'b-', label='Frozen', linewidth=2)
ax.plot(history['active'], 'r-', label='Active', linewidth=2)
ax.fill_between(range(len(history['neurons'])), history['frozen'], alpha=0.3, color='blue')
ax.set_xlabel('Epoch')
ax.set_ylabel('Neurons')
ax.set_title('Crystal Growth & Freezing')
ax.legend()
ax.grid(True, alpha=0.3)

# Frozen %
ax = axes[0, 2]
frozen_pct = [100*f/n if n > 0 else 0 for f, n in zip(history['frozen'], history['neurons'])]
ax.plot(frozen_pct, 'purple', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Frozen %')
ax.set_title('Crystallization Progress')
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

# Speedup
ax = axes[1, 0]
ax.plot(history['speedup'], 'orange', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Speedup')
ax.set_title('Training Speedup (from freezing)')
ax.grid(True, alpha=0.3)

# Final 3D view
ax = axes[1, 1]
ax.remove()
ax = fig.add_subplot(2, 3, 5, projection='3d')
data = crystal.get_all_positions()
pos_3d = pca.transform(data['positions'])
colors = ['blue' if f else 'red' for f in data['frozen']]
ax.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], c=colors, s=10, alpha=0.5)
ax.set_title(f'Final Crystal Structure\n{stats["neurons"]} neurons, {stats["frozen"]} frozen')

# Summary text
ax = axes[1, 2]
ax.axis('off')
final_stats = crystal.stats()
final_pct = 100 * final_stats['frozen'] / final_stats['neurons']
final_speedup = final_stats['neurons'] / final_stats['active'] if final_stats['active'] > 0 else float('inf')

summary = f"""
CRYSTALLIZATION COMPLETE
═══════════════════════════════════

Training: {EPOCHS} epochs, {elapsed/60:.1f} minutes

GROWTH:
  Started: {initial_total} neurons
  Final: {final_stats['neurons']} neurons
  Growth: {final_stats['neurons']/initial_total:.1f}x

CRYSTALLIZATION:
  Frozen: {final_stats['frozen']} ({final_pct:.1f}%)
  Active: {final_stats['active']} ({100-final_pct:.1f}%)
  Speedup: {final_speedup:.1f}x

LEARNING:
  Initial Loss: {history['loss'][0]:.3f}
  Final Loss: {history['loss'][-1]:.3f}

"Intelligence crystallizes into geometry."
"""
ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
       verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(f"runs/{RUN_NAME}/summary.png", dpi=150)
print(f"    Saved: runs/{RUN_NAME}/summary.png")

# Save checkpoint
torch.save({
    'model': crystal.state_dict(),
    'history': history,
    'config': {'neurons': INITIAL_NEURONS, 'blocks': NUM_BLOCKS, 'max': MAX_NEURONS}
}, f"runs/{RUN_NAME}/crystal_final.pt")
print(f"    Saved: runs/{RUN_NAME}/crystal_final.pt")

# Final generation test
print("\n" + "=" * 70)
print("[7] Final Generation Test")
print("=" * 70)

crystal.eval()
prompts = [
    "The meaning of life is",
    "Neural networks learn",
    "The universe began",
    "Science has shown that",
]

for prompt in prompts:
    inp = tokenizer(prompt, return_tensors='pt').to(device)
    out = crystal.generate(inp['input_ids'], max_new=35, temperature=0.7, top_k=50)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\n  {text}")

print(f"\n\nAll visualizations saved to: runs/{RUN_NAME}/")
print(f"Frames: epoch_000.png through epoch_{EPOCHS:03d}_final.png")
print(f"\nTo create animation: ffmpeg -framerate 5 -pattern_type glob -i 'runs/{RUN_NAME}/epoch_*.png' -c:v libx264 crystal_growth.mp4")
