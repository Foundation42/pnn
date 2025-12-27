"""
Rising Tide Crystal - Neurons imprinted at birth with current aperture

Key insight: The tide rises once over the entire training run.
When a neuron is born, it's permanently assigned the current tide level.

Result:
- Early-born neurons (low tide) → local patterns
- Mid-born neurons (medium tide) → phrase patterns
- Late-born neurons (high tide) → global patterns

No resets, no discontinuities. Multi-scale from birth timing!
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

# Rising tide config
MIN_SCALE = 0.05      # Start of training (local)
MAX_SCALE = 0.80      # End of training (global)


def get_tide_level(epoch, total_epochs=EPOCHS):
    """Linear rise from MIN to MAX over training"""
    progress = epoch / total_epochs
    return MIN_SCALE + (MAX_SCALE - MIN_SCALE) * progress


class RisingTideAttention(nn.Module):
    """
    Geometric attention where each neuron has a fixed scale assigned at birth.
    The "tide" (current scale for new neurons) rises over training.
    """

    def __init__(self, embed_dim, num_neurons, birth_epoch=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.5)
        self.values = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(num_neurons) * 0.5)

        # Each neuron's FIXED scale (assigned at birth, never changes)
        initial_scale = get_tide_level(birth_epoch)
        self.register_buffer('neuron_scale', torch.ones(num_neurons) * initial_scale)
        self.register_buffer('birth_epoch', torch.ones(num_neurons) * birth_epoch)

        # Tracking
        self.register_buffer('gradient_acc', torch.zeros(num_neurons))
        self.register_buffer('gradient_history', torch.zeros(num_neurons, 10))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('frozen', torch.zeros(num_neurons, dtype=torch.bool))

        self.current_epoch = birth_epoch
        self.current_tide = initial_scale

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        dists = torch.cdist(x_flat, self.positions)

        # Each neuron uses its birth-assigned scale
        effective_temp = (self.temperature.abs() + 0.1) * self.neuron_scale

        weights = torch.exp(-dists / effective_temp.unsqueeze(0))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

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
        """Add new neurons with current tide level as their scale"""
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

            # New neurons get CURRENT tide level as their permanent scale!
            current_scale = get_tide_level(self.current_epoch)
            self.neuron_scale = torch.cat([
                self.neuron_scale,
                torch.ones(n_new, device=device) * current_scale
            ])
            self.birth_epoch = torch.cat([
                self.birth_epoch,
                torch.ones(n_new, device=device) * self.current_epoch
            ])

            self.gradient_acc = torch.cat([self.gradient_acc, torch.zeros(n_new, device=device)])
            self.gradient_history = torch.cat([self.gradient_history, torch.zeros(n_new, 10, device=device)])
            self.frozen = torch.cat([self.frozen, torch.zeros(n_new, dtype=torch.bool, device=device)])

            self.num_neurons = len(self.positions)
            return n_new
        return 0

    def freeze_cold_neurons(self, epoch, total_epochs):
        """Freeze stabilized neurons (they keep their birth scale)"""

        if epoch < total_epochs * 0.15:
            return 0

        progress = epoch / total_epochs

        if progress < 0.4:
            aggression = 0.2
        elif progress < 0.7:
            aggression = 0.4
        else:
            aggression = 0.7

        min_age = 25
        neuron_age = epoch - self.birth_epoch
        too_young = neuron_age < min_age

        if self.history_ptr < 8:
            return 0

        grad_variance = self.gradient_history.var(dim=1)
        grad_mean = self.gradient_history.mean(dim=1)

        candidates = (~self.frozen) & (~too_young)
        if candidates.sum() < 3:
            return 0

        coldness = torch.zeros(self.num_neurons, device=self.positions.device)
        coldness[candidates] = -grad_mean[candidates] - grad_variance[candidates] * 10
        coldness[~candidates] = float('-inf')

        max_freeze = int(candidates.sum().item() * aggression * 0.12)
        max_freeze = max(1, min(max_freeze, 15))

        _, freeze_idx = torch.topk(coldness, min(max_freeze, candidates.sum().item()))

        median_activity = grad_mean[candidates].median()
        actually_cold = grad_mean[freeze_idx] < median_activity

        to_freeze = freeze_idx[actually_cold]
        self.frozen[to_freeze] = True

        return len(to_freeze)

    def zero_frozen_grads(self):
        if self.positions.grad is not None:
            self.positions.grad[self.frozen] = 0
            self.values.grad[self.frozen] = 0
            self.temperature.grad[self.frozen] = 0

    def scale_stats(self):
        """Get scale distribution statistics"""
        active_mask = ~self.frozen
        frozen_mask = self.frozen

        stats = {}

        if active_mask.any():
            active_scales = self.neuron_scale[active_mask]
            stats['active_mean'] = active_scales.mean().item()
            stats['active_min'] = active_scales.min().item()
            stats['active_max'] = active_scales.max().item()
        else:
            stats['active_mean'] = stats['active_min'] = stats['active_max'] = 0

        if frozen_mask.any():
            frozen_scales = self.neuron_scale[frozen_mask]
            stats['frozen_mean'] = frozen_scales.mean().item()
            stats['frozen_min'] = frozen_scales.min().item()
            stats['frozen_max'] = frozen_scales.max().item()
        else:
            stats['frozen_mean'] = stats['frozen_min'] = stats['frozen_max'] = 0

        # Overall distribution
        stats['all_mean'] = self.neuron_scale.mean().item()
        stats['all_std'] = self.neuron_scale.std().item()

        return stats


class RisingTideCrystalLM(nn.Module):
    """Language model with rising tide attention"""

    def __init__(self, vocab_size, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        self.attention = RisingTideAttention(embed_dim, num_neurons)
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
    cache_paths = ["data/tinyshakespeare.txt", "../data/tinyshakespeare.txt"]
    for path in cache_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs("data", exist_ok=True)
    import requests
    response = requests.get(url)
    text = response.text
    with open("data/tinyshakespeare.txt", 'w') as f:
        f.write(text)
    return text


def create_batches(tokens, batch_size, context_len):
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


def visualize_rising_tide(model, epoch, run_dir, tide_history, loss_history):
    """Visualize the rising tide crystal"""
    positions = model.attention.positions.detach().cpu().numpy()
    frozen = model.attention.frozen.cpu().numpy()
    scales = model.attention.neuron_scale.cpu().numpy()
    birth_epochs = model.attention.birth_epoch.cpu().numpy()

    # PCA
    if positions.shape[0] > 2:
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(positions)
    else:
        pos_2d = positions[:, :2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Crystal colored by birth scale
    ax = axes[0, 0]
    sizes = np.where(frozen, 80, 40)
    edge_colors = ['dimgray' if f else 'black' for f in frozen]

    scatter = ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
                         c=scales, cmap='coolwarm',
                         s=sizes, alpha=0.7, edgecolors=edge_colors, linewidths=0.5,
                         vmin=MIN_SCALE, vmax=MAX_SCALE)
    plt.colorbar(scatter, ax=ax, label='Birth Scale (blue=local, red=global)')

    n_frozen = frozen.sum()
    current_tide = get_tide_level(epoch)
    ax.set_title(f'Rising Tide Crystal - Epoch {epoch}\n{len(frozen)}N, {n_frozen}F ({100*n_frozen/len(frozen):.0f}%), Tide={current_tide:.2f}')

    # 2. Tide + Loss history
    ax = axes[0, 1]
    epochs_plot = list(range(1, len(tide_history) + 1))

    ax.plot(epochs_plot, tide_history, 'b-', linewidth=2, label='Tide (scale)')
    ax.set_ylabel('Tide Level', color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_ylim(0, MAX_SCALE * 1.1)

    ax2 = ax.twinx()
    ax2.plot(epochs_plot, loss_history, 'r-', linewidth=1.5, alpha=0.7, label='Loss')
    ax2.set_ylabel('Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_xlabel('Epoch')
    ax.set_title('Rising Tide & Loss')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 3. Scale distribution by birth epoch
    ax = axes[1, 0]
    ax.scatter(birth_epochs, scales, c=scales, cmap='coolwarm', alpha=0.6,
               vmin=MIN_SCALE, vmax=MAX_SCALE, s=30)
    ax.plot([1, epoch], [get_tide_level(1), get_tide_level(epoch)], 'k--', alpha=0.5, label='Tide line')
    ax.set_xlabel('Birth Epoch')
    ax.set_ylabel('Neuron Scale')
    ax.set_title('Scale vs Birth Time\n(neurons follow the rising tide)')
    ax.legend()

    # 4. Scale histogram
    ax = axes[1, 1]
    ax.hist(scales[~frozen], bins=20, alpha=0.7, label='Active', color='steelblue',
            range=(MIN_SCALE, MAX_SCALE), edgecolor='black')
    ax.hist(scales[frozen], bins=20, alpha=0.7, label='Frozen', color='gray',
            range=(MIN_SCALE, MAX_SCALE), edgecolor='black')
    ax.set_xlabel('Scale')
    ax.set_ylabel('Count')
    ax.set_title('Scale Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{run_dir}/rising_epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def main():
    print("=" * 70)
    print("RISING TIDE CRYSTAL - Neurons imprinted at birth")
    print("=" * 70)
    print(f"Tide rises linearly: {MIN_SCALE} (epoch 1) -> {MAX_SCALE} (epoch {EPOCHS})")
    print("Early neurons = local, Late neurons = global")
    print("No resets, no discontinuities!")
    print("=" * 70)

    text = load_shakespeare()
    print(f"Loaded {len(text):,} characters")

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")

    model = RisingTideCrystalLM(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        num_neurons=INITIAL_NEURONS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/rising_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nTraining on {DEVICE}")
    print(f"Output: {run_dir}/")
    print("=" * 70)

    best_loss = float('inf')
    tide_history = []
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.attention.current_epoch = epoch

        current_tide = get_tide_level(epoch)
        model.attention.current_tide = current_tide
        tide_history.append(current_tide)

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
        loss_history.append(avg_loss)

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

        # Tide indicator
        tide_bar = "~" * int(current_tide / MAX_SCALE * 10)

        # Generate sample
        sample = ""
        if epoch % 20 == 0:
            sample = generate(model, tokenizer, "ROMEO:", max_tokens=60, temperature=0.8)
            visualize_rising_tide(model, epoch, run_dir, tide_history, loss_history)

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | N: {model.num_neurons} | F: {model.num_frozen} ({frozen_pct:.0f}%) | "
              f"Tide: {current_tide:.2f} [{tide_bar:<10}] | "
              f"Scales: {scale_stats['all_mean']:.2f}±{scale_stats['all_std']:.2f}")

        if sample:
            print(f"         -> {sample[:70]}...")

        # Save history
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'neurons': model.num_neurons,
            'frozen': model.num_frozen,
            'tide': current_tide,
            'scale_stats': scale_stats,
            'sample': sample[:200] if sample else ""
        }
        with open(f'{run_dir}/history.jsonl', 'a') as f:
            f.write(json.dumps(epoch_data) + '\n')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
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
    print(f"  All neurons: {scale_stats['all_mean']:.3f} ± {scale_stats['all_std']:.3f}")
    print(f"  Active: [{scale_stats['active_min']:.2f}, {scale_stats['active_max']:.2f}]")
    print(f"  Frozen: [{scale_stats['frozen_min']:.2f}, {scale_stats['frozen_max']:.2f}]")
    print("=" * 70)

    sample = generate(model, tokenizer, "ROMEO:", max_tokens=200, temperature=0.8)
    print("\nFinal generation:")
    print(sample)

    visualize_rising_tide(model, EPOCHS, run_dir, tide_history, loss_history)


if __name__ == "__main__":
    main()
