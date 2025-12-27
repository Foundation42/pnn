"""
Multi-Scale Crystal - Smooth attention scaling tied to neuron age/freezing

Key insight: No discrete layers! Instead:
- Each neuron has its own attention scale
- Scale grows with neuron age
- When frozen, scale locks in
- Result: Early-frozen neurons = local patterns, late-frozen = global patterns

The crystal self-organizes into multi-scale structure!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import tiktoken
from datetime import datetime
import json

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 128
CONTEXT_LEN = 64
BATCH_SIZE = 32
EPOCHS = 600
MAX_SEQUENCES = 10000

# Crystal config
INITIAL_NEURONS = 64
MAX_NEURONS = 1024
GROWTH_INTERVAL = 10

# Multi-scale config
MIN_ATTENTION_SCALE = 0.05   # Young neurons: very local
MAX_ATTENTION_SCALE = 1.0    # Old neurons: global
SCALE_GROWTH_RATE = 0.02     # How fast scale grows per epoch of age


class MultiScaleGeometricAttention(nn.Module):
    """
    Geometric attention where each neuron has its own attention scale.
    Scale grows with age - young neurons are local, old neurons are global.
    """

    def __init__(self, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        # Neuron positions in embedding space
        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.5)
        self.values = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.02)

        # Base temperature (learnable per neuron)
        self.temperature = nn.Parameter(torch.ones(num_neurons) * 0.5)

        # Per-neuron attention scale (computed from age, not learned)
        self.register_buffer('attention_scale', torch.ones(num_neurons) * MIN_ATTENTION_SCALE)

        # Tracking
        self.register_buffer('gradient_acc', torch.zeros(num_neurons))
        self.register_buffer('gradient_history', torch.zeros(num_neurons, 10))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('frozen', torch.zeros(num_neurons, dtype=torch.bool))
        self.register_buffer('birth_epoch', torch.zeros(num_neurons))
        self.register_buffer('frozen_scale', torch.zeros(num_neurons))  # Scale when frozen
        self.current_epoch = 0

    def update_attention_scales(self):
        """Update attention scales based on neuron age. Frozen neurons keep their scale."""
        ages = self.current_epoch - self.birth_epoch

        # Scale grows with age: min + (max-min) * (1 - exp(-rate * age))
        computed_scale = MIN_ATTENTION_SCALE + (MAX_ATTENTION_SCALE - MIN_ATTENTION_SCALE) * (
            1 - torch.exp(-SCALE_GROWTH_RATE * ages)
        )

        # Frozen neurons keep their frozen scale
        self.attention_scale = torch.where(
            self.frozen,
            self.frozen_scale,
            computed_scale
        )

    def forward(self, x):
        B, T, D = x.shape

        # Update scales based on current ages
        self.update_attention_scales()

        x_flat = x.reshape(-1, D)  # (B*T, D)

        # Distance from tokens to neurons
        dists = torch.cdist(x_flat, self.positions)  # (B*T, N)

        # Each neuron has its own attention scale
        # Larger scale = wider attention = more global
        effective_temp = self.temperature.abs() + 0.1
        effective_temp = effective_temp * self.attention_scale  # Scale modulates temperature

        # RBF attention weights
        weights = torch.exp(-dists / effective_temp)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Gather values
        out = weights @ self.values

        return out.reshape(B, T, D)

    def accumulate_gradients(self):
        if self.positions.grad is not None:
            grad_norm = self.positions.grad.norm(dim=1)
            self.gradient_acc = 0.9 * self.gradient_acc + 0.1 * grad_norm.detach()

            ptr = self.history_ptr.item() % 10
            self.gradient_history[:, ptr] = grad_norm.detach()
            self.history_ptr += 1

    def grow_neurons(self, num_new, device):
        """Add new neurons - they start with minimum attention scale"""
        if num_new <= 0:
            return 0

        activity = self.gradient_acc.clone()
        activity[self.frozen] = -float('inf')
        num_active = (~self.frozen).sum().item()
        if num_active == 0:
            return 0

        num_to_split = min(num_new, num_active)
        _, hot_idx = torch.topk(activity, num_to_split)

        new_positions = []
        new_values = []
        new_temps = []

        for idx in hot_idx:
            offset = torch.randn(self.embed_dim, device=device) * 0.1
            new_positions.append(self.positions[idx] + offset)
            new_values.append(self.values[idx] * 0.5)
            new_temps.append(self.temperature[idx].clone())

        if new_positions:
            n_new = len(new_positions)
            new_pos = torch.stack(new_positions)
            new_val = torch.stack(new_values)
            new_temp = torch.stack(new_temps)

            self.positions = nn.Parameter(torch.cat([self.positions.data, new_pos]))
            self.values = nn.Parameter(torch.cat([self.values.data, new_val]))
            self.temperature = nn.Parameter(torch.cat([self.temperature.data, new_temp]))

            # New neurons start with minimum scale
            self.attention_scale = torch.cat([
                self.attention_scale,
                torch.ones(n_new, device=device) * MIN_ATTENTION_SCALE
            ])
            self.gradient_acc = torch.cat([self.gradient_acc, torch.zeros(n_new, device=device)])
            self.gradient_history = torch.cat([self.gradient_history, torch.zeros(n_new, 10, device=device)])
            self.frozen = torch.cat([self.frozen, torch.zeros(n_new, dtype=torch.bool, device=device)])
            self.birth_epoch = torch.cat([self.birth_epoch, torch.full((n_new,), self.current_epoch, device=device)])
            self.frozen_scale = torch.cat([self.frozen_scale, torch.zeros(n_new, device=device)])

            self.num_neurons = len(self.positions)
            return n_new
        return 0

    def freeze_cold_neurons(self, epoch, total_epochs):
        """Freeze neurons and lock in their current attention scale"""

        # No freezing in early phase
        if epoch < total_epochs * 0.15:
            return 0

        progress = epoch / total_epochs

        # Aggression increases over training
        if progress < 0.4:
            aggression = (progress - 0.15) / 0.25 * 0.3
        elif progress < 0.7:
            aggression = 0.3 + (progress - 0.4) / 0.3 * 0.3
        else:
            aggression = 0.6 + (progress - 0.7) / 0.3 * 0.4

        min_age = 30
        neuron_age = epoch - self.birth_epoch
        too_young = neuron_age < min_age

        if self.history_ptr < 10:
            return 0

        grad_variance = self.gradient_history.var(dim=1)
        grad_mean = self.gradient_history.mean(dim=1)

        candidates = (~self.frozen) & (~too_young)
        if candidates.sum() < 5:
            return 0

        coldness = torch.zeros(self.num_neurons, device=self.positions.device)
        coldness[candidates] = -grad_mean[candidates] - grad_variance[candidates] * 10
        coldness[~candidates] = float('-inf')

        max_freeze = int(candidates.sum().item() * aggression * 0.1)
        max_freeze = max(1, min(max_freeze, 10))

        _, freeze_idx = torch.topk(coldness, min(max_freeze, candidates.sum().item()))

        median_activity = grad_mean[candidates].median()
        actually_cold = grad_mean[freeze_idx] < median_activity

        to_freeze = freeze_idx[actually_cold]

        # Lock in their attention scale when freezing!
        self.frozen_scale[to_freeze] = self.attention_scale[to_freeze].clone()
        self.frozen[to_freeze] = True

        return len(to_freeze)

    def zero_frozen_grads(self):
        if self.positions.grad is not None:
            self.positions.grad[self.frozen] = 0
            self.values.grad[self.frozen] = 0
            self.temperature.grad[self.frozen] = 0

    def scale_stats(self):
        """Get statistics about attention scales"""
        active_mask = ~self.frozen
        frozen_mask = self.frozen

        stats = {
            'active_mean_scale': self.attention_scale[active_mask].mean().item() if active_mask.any() else 0,
            'active_min_scale': self.attention_scale[active_mask].min().item() if active_mask.any() else 0,
            'active_max_scale': self.attention_scale[active_mask].max().item() if active_mask.any() else 0,
            'frozen_mean_scale': self.frozen_scale[frozen_mask].mean().item() if frozen_mask.any() else 0,
            'frozen_min_scale': self.frozen_scale[frozen_mask].min().item() if frozen_mask.any() else 0,
            'frozen_max_scale': self.frozen_scale[frozen_mask].max().item() if frozen_mask.any() else 0,
        }
        return stats


class MultiScaleCrystalLM(nn.Module):
    """Language model with multi-scale geometric attention"""

    def __init__(self, vocab_size, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        self.attention = MultiScaleGeometricAttention(embed_dim, num_neurons)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        pos = torch.arange(T, device=x.device)
        h = self.token_embed(x) + self.pos_embed(pos)

        h = h + self.attention(self.norm1(h))
        h = h + self.ffn(self.norm2(h))

        logits = self.head(h)

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return logits

    @property
    def num_neurons(self):
        return self.attention.num_neurons

    @property
    def num_frozen(self):
        return self.attention.frozen.sum().item()


def load_shakespeare():
    """Load TinyShakespeare"""
    cache_paths = ["data/tinyshakespeare.txt", "../data/tinyshakespeare.txt"]
    for path in cache_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs("data", exist_ok=True)
    print("Downloading TinyShakespeare...")
    import requests
    response = requests.get(url)
    text = response.text
    with open("data/tinyshakespeare.txt", 'w') as f:
        f.write(text)
    return text


def create_batches(tokens, batch_size, context_len):
    """Create training batches"""
    n_tokens = len(tokens)
    n_possible = n_tokens - context_len - 1

    if n_possible <= MAX_SEQUENCES:
        sequences = []
        for i in range(0, n_tokens - context_len - 1, context_len // 2):
            seq = tokens[i:i + context_len + 1]
            if len(seq) == context_len + 1:
                sequences.append(seq)
    else:
        starts = np.random.randint(0, n_possible, size=MAX_SEQUENCES)
        sequences = [tokens[i:i + context_len + 1] for i in starts]

    indices = torch.randperm(len(sequences))
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        if len(batch_idx) == batch_size:
            batch = torch.stack([torch.tensor(sequences[j]) for j in batch_idx])
            batches.append(batch)

    return batches


def generate(model, tokenizer, prompt="ROMEO:", max_tokens=100, temperature=0.8):
    """Generate text"""
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = tokens[-CONTEXT_LEN:]

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([tokens[-CONTEXT_LEN:]], device=DEVICE)
            logits = model(x)
            logits = logits[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

    return tokenizer.decode(tokens)


def visualize_multiscale(model, epoch, run_dir, sample_text=""):
    """Visualize neurons colored by their attention scale"""
    positions = model.attention.positions.detach().cpu().numpy()
    frozen = model.attention.frozen.cpu().numpy()
    scales = model.attention.attention_scale.cpu().numpy()
    frozen_scales = model.attention.frozen_scale.cpu().numpy()

    # Use frozen_scale for frozen neurons, current scale for active
    display_scales = np.where(frozen, frozen_scales, scales)

    # PCA to 2D
    if positions.shape[0] > 2:
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(positions)
    else:
        pos_2d = positions[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Crystal structure colored by scale
    ax = axes[0]

    # Color by scale (blue=local, red=global)
    colors = plt.cm.coolwarm(display_scales / MAX_ATTENTION_SCALE)

    # Size by frozen status
    sizes = np.where(frozen, 80, 40)

    # Edge color: gray for frozen, black for active
    edge_colors = ['gray' if f else 'black' for f in frozen]

    scatter = ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
                         c=display_scales, cmap='coolwarm',
                         s=sizes, alpha=0.7, edgecolors=edge_colors, linewidths=0.5,
                         vmin=MIN_ATTENTION_SCALE, vmax=MAX_ATTENTION_SCALE)

    plt.colorbar(scatter, ax=ax, label='Attention Scale (blue=local, red=global)')

    n_frozen = frozen.sum()
    frozen_pct = 100 * n_frozen / len(frozen)
    ax.set_title(f'Multi-Scale Crystal - Epoch {epoch}\n{len(frozen)} neurons, {n_frozen} frozen ({frozen_pct:.0f}%)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Scale histogram
    ax = axes[1]
    ax.hist(display_scales[~frozen], bins=20, alpha=0.7, label='Active', color='blue', range=(0, MAX_ATTENTION_SCALE))
    ax.hist(display_scales[frozen], bins=20, alpha=0.7, label='Frozen', color='gray', range=(0, MAX_ATTENTION_SCALE))
    ax.set_xlabel('Attention Scale')
    ax.set_ylabel('Count')
    ax.set_title('Scale Distribution\n(Active vs Frozen)')
    ax.legend()
    ax.axvline(x=display_scales[~frozen].mean() if (~frozen).any() else 0, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=display_scales[frozen].mean() if frozen.any() else 0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{run_dir}/epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def main():
    print("=" * 70)
    print("MULTI-SCALE CRYSTAL - Attention scale grows with neuron age")
    print("=" * 70)
    print(f"Scale range: {MIN_ATTENTION_SCALE} (local) -> {MAX_ATTENTION_SCALE} (global)")
    print(f"Scale growth rate: {SCALE_GROWTH_RATE} per epoch of age")
    print("Young neurons = local patterns, Old neurons = global patterns")
    print("=" * 70)

    # Load data
    text = load_shakespeare()
    print(f"Loaded {len(text):,} characters")

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")

    # Create model
    model = MultiScaleCrystalLM(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        num_neurons=INITIAL_NEURONS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/multiscale_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nTraining on {DEVICE}")
    print(f"Initial neurons: {INITIAL_NEURONS}, Max: {MAX_NEURONS}")
    print(f"Output: {run_dir}/")
    print("=" * 70)

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.attention.current_epoch = epoch

        batches = create_batches(tokens, BATCH_SIZE, CONTEXT_LEN)

        total_loss = 0
        for batch in batches:
            batch = batch.to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:]

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()

            model.attention.accumulate_gradients()
            model.attention.zero_frozen_grads()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batches)

        # Growth and freezing
        if epoch % GROWTH_INTERVAL == 0:
            model.attention.freeze_cold_neurons(epoch, EPOCHS)

            if model.num_neurons < MAX_NEURONS:
                active = model.num_neurons - model.num_frozen
                progress = epoch / EPOCHS

                if progress < 0.3:
                    base_grow = 24
                elif progress < 0.6:
                    base_grow = 12
                else:
                    base_grow = 4

                grow = min(base_grow, MAX_NEURONS - model.num_neurons, max(active // 2, 2))
                model.attention.grow_neurons(grow, DEVICE)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Stats
        scale_stats = model.attention.scale_stats()
        frozen_pct = 100 * model.num_frozen / model.num_neurons

        # Generate sample
        sample = ""
        if epoch % 20 == 0:
            sample = generate(model, tokenizer, "ROMEO:", max_tokens=60, temperature=0.8)
            visualize_multiscale(model, epoch, run_dir, sample)

        # Progress
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | N: {model.num_neurons} | F: {model.num_frozen} ({frozen_pct:.0f}%) | "
              f"Scale A:{scale_stats['active_mean_scale']:.2f} F:{scale_stats['frozen_mean_scale']:.2f}")

        if sample:
            print(f"         -> {sample[:70]}...")

        # Save history
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'neurons': model.num_neurons,
            'frozen': model.num_frozen,
            'scale_stats': scale_stats,
            'sample': sample[:200] if sample else ""
        }
        with open(f'{run_dir}/history.jsonl', 'a') as f:
            f.write(json.dumps(epoch_data) + '\n')

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'config': {
                    'vocab_size': tokenizer.n_vocab,
                    'embed_dim': EMBED_DIM,
                    'num_neurons': model.num_neurons
                },
                'epoch': epoch,
                'loss': avg_loss
            }, f'{run_dir}/best_model.pt')

    # Final
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Final: {model.num_neurons} neurons, {model.num_frozen} frozen ({100*model.num_frozen/model.num_neurons:.0f}%)")
    print(f"Best loss: {best_loss:.4f}")

    scale_stats = model.attention.scale_stats()
    print(f"\nScale distribution:")
    print(f"  Active neurons: mean={scale_stats['active_mean_scale']:.3f}, range=[{scale_stats['active_min_scale']:.3f}, {scale_stats['active_max_scale']:.3f}]")
    print(f"  Frozen neurons: mean={scale_stats['frozen_mean_scale']:.3f}, range=[{scale_stats['frozen_min_scale']:.3f}, {scale_stats['frozen_max_scale']:.3f}]")
    print("=" * 70)

    sample = generate(model, tokenizer, "ROMEO:", max_tokens=200, temperature=0.8)
    print("\nFinal generation:")
    print(sample)


if __name__ == "__main__":
    main()
