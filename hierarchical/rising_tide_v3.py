"""
Rising Tide v3 - Soft Freeze (Learning Rate Decay)

Key insight: Don't hard freeze! Instead, gracefully decay learning rate
based on neuron age. Old neurons can still learn, just slowly.

This solves the "dead vs deed" problem - early neurons can still
receive feedback from what later neurons learn about context.

Young neurons: Full learning rate (exploring)
Old neurons: 1% learning rate (stable but adaptable)
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
NEURONS_PER_EPOCH = 1  # Ultra-smooth: one neuron at a time

# Rising tide config
MIN_SCALE = 0.05
MAX_SCALE = 0.80

# Soft freeze config
LR_DECAY_HALFLIFE = 50   # Epochs for LR to halve
MIN_LR_MULTIPLIER = 0.01  # Floor - never fully freeze


def get_tide_level(epoch, total_epochs=EPOCHS):
    """Linear rise from MIN to MAX over training"""
    progress = epoch / total_epochs
    return MIN_SCALE + (MAX_SCALE - MIN_SCALE) * progress


def get_lr_multiplier(age, halflife=LR_DECAY_HALFLIFE):
    """
    Exponential decay of learning rate with age.

    age=0: multiplier=1.0 (full learning)
    age=halflife: multiplier=0.5
    age=2*halflife: multiplier=0.25
    ...asymptotes to MIN_LR_MULTIPLIER
    """
    decay = 0.5 ** (age / halflife)
    return max(MIN_LR_MULTIPLIER, decay)


class SoftFreezeAttention(nn.Module):
    """
    Geometric attention with soft freeze (LR decay based on age).
    No hard freezing - all neurons can always learn, just at different rates.
    """

    def __init__(self, embed_dim, num_neurons, birth_epoch=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.5)
        self.values = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(num_neurons) * 0.5)

        # Birth info for scale and age
        initial_scale = get_tide_level(birth_epoch)
        self.register_buffer('neuron_scale', torch.ones(num_neurons) * initial_scale)
        self.register_buffer('birth_epoch', torch.ones(num_neurons) * birth_epoch)

        # Gradient tracking for growth decisions
        self.register_buffer('gradient_acc', torch.zeros(num_neurons))

        self.current_epoch = birth_epoch

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

    def get_ages(self):
        """Get current age of each neuron"""
        return self.current_epoch - self.birth_epoch

    def get_lr_multipliers(self):
        """Get learning rate multiplier for each neuron based on age"""
        ages = self.get_ages()
        multipliers = torch.zeros_like(ages)
        for i, age in enumerate(ages):
            multipliers[i] = get_lr_multiplier(age.item())
        return multipliers

    def apply_soft_freeze_to_gradients(self):
        """Scale gradients by age-based LR multiplier"""
        if self.positions.grad is None:
            return

        multipliers = self.get_lr_multipliers()

        # Scale gradients for each neuron
        # Positions: (num_neurons, embed_dim) - scale each row
        self.positions.grad *= multipliers.unsqueeze(1)
        self.values.grad *= multipliers.unsqueeze(1)
        self.temperature.grad *= multipliers

    def accumulate_gradients(self):
        """Track gradient magnitudes for growth decisions"""
        if self.positions.grad is not None:
            grad_norm = self.positions.grad.norm(dim=1)
            self.gradient_acc = 0.9 * self.gradient_acc + 0.1 * grad_norm.detach()

    def grow_neurons(self, num_new, device):
        """Add new neurons with current tide level"""
        if num_new <= 0:
            return 0

        # Find neurons with highest gradient activity to split
        # Prefer younger neurons (they're more active)
        ages = self.get_ages()
        activity = self.gradient_acc.clone()

        # Weight by youth (younger = higher priority for splitting)
        youth_weight = 1.0 / (ages + 1)
        weighted_activity = activity * youth_weight

        num_to_split = min(num_new, self.num_neurons)
        _, hot_idx = torch.topk(weighted_activity, num_to_split)

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

            # New neurons get current tide as birth scale
            current_scale = get_tide_level(self.current_epoch)

            self.neuron_scale = torch.cat([
                self.neuron_scale,
                torch.ones(n_new, device=device) * current_scale
            ])
            self.birth_epoch = torch.cat([
                self.birth_epoch,
                torch.ones(n_new, device=device) * self.current_epoch
            ])
            self.gradient_acc = torch.cat([
                self.gradient_acc,
                torch.zeros(n_new, device=device)
            ])

            self.num_neurons = len(self.positions)
            return n_new
        return 0

    def get_stats(self):
        """Get detailed statistics"""
        ages = self.get_ages()
        lr_mults = self.get_lr_multipliers()

        # Define "effectively frozen" as LR < 0.1 (just for stats)
        frozen_mask = lr_mults < 0.1
        active_mask = ~frozen_mask

        stats = {
            'num_neurons': self.num_neurons,
            'num_active': active_mask.sum().item(),
            'num_slow': frozen_mask.sum().item(),  # "slow learners" not "frozen"
            'avg_lr_mult': lr_mults.mean().item(),
            'min_lr_mult': lr_mults.min().item(),
            'max_lr_mult': lr_mults.max().item(),
        }

        if active_mask.any():
            stats['active_scale_mean'] = self.neuron_scale[active_mask].mean().item()
            stats['active_age_mean'] = ages[active_mask].mean().item()
        else:
            stats['active_scale_mean'] = 0
            stats['active_age_mean'] = 0

        if frozen_mask.any():
            stats['slow_scale_mean'] = self.neuron_scale[frozen_mask].mean().item()
        else:
            stats['slow_scale_mean'] = 0

        stats['all_scale_mean'] = self.neuron_scale.mean().item()
        stats['all_scale_std'] = self.neuron_scale.std().item()

        return stats


class SoftFreezeCrystalLM(nn.Module):
    """Language model with soft freeze attention"""

    def __init__(self, vocab_size, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        self.attention = SoftFreezeAttention(embed_dim, num_neurons)
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


def visualize(model, epoch, run_dir, tide_history, loss_history):
    """Visualize the soft-freeze crystal"""
    positions = model.attention.positions.detach().cpu().numpy()
    scales = model.attention.neuron_scale.cpu().numpy()
    ages = model.attention.get_ages().cpu().numpy()
    lr_mults = model.attention.get_lr_multipliers().cpu().numpy()

    if positions.shape[0] > 2:
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(positions)
    else:
        pos_2d = positions[:, :2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Crystal colored by LR multiplier (blue=fast learning, red=slow)
    ax = axes[0, 0]

    scatter = ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
                         c=lr_mults, cmap='coolwarm_r',  # reversed: blue=high LR
                         s=50, alpha=0.7, edgecolors='black', linewidths=0.3,
                         vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label='LR Multiplier (blue=fast, red=slow)')

    current_tide = get_tide_level(epoch)
    stats = model.attention.get_stats()
    ax.set_title(f'Soft Freeze Crystal - Epoch {epoch}\n{stats["num_neurons"]}N, Avg LR: {stats["avg_lr_mult"]:.2f}')

    # 2. Tide + Loss
    ax = axes[0, 1]
    epochs_plot = list(range(1, len(tide_history) + 1))

    ax.plot(epochs_plot, tide_history, 'b-', linewidth=2, label='Tide')
    ax.set_ylabel('Tide', color='blue')
    ax.set_ylim(0, MAX_SCALE * 1.1)

    ax2 = ax.twinx()
    ax2.plot(epochs_plot, loss_history, 'r-', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Loss', color='red')

    ax.set_xlabel('Epoch')
    ax.set_title('Tide & Loss')

    # 3. Age vs LR multiplier
    ax = axes[1, 0]
    ax.scatter(ages, lr_mults, c=scales, cmap='viridis', alpha=0.6, s=30)

    # Overlay theoretical curve
    age_range = np.linspace(0, ages.max() if len(ages) > 0 else 200, 100)
    theoretical_lr = [get_lr_multiplier(a) for a in age_range]
    ax.plot(age_range, theoretical_lr, 'r--', linewidth=2, label='Decay curve')

    ax.set_xlabel('Age (epochs)')
    ax.set_ylabel('LR Multiplier')
    ax.set_title(f'Learning Rate Decay (halflife={LR_DECAY_HALFLIFE})')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # 4. Scale distribution colored by LR
    ax = axes[1, 1]
    scatter = ax.scatter(scales, lr_mults, c=ages, cmap='plasma', alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax, label='Age')
    ax.set_xlabel('Scale (birth tide)')
    ax.set_ylabel('LR Multiplier')
    ax.set_title('Scale vs Learning Rate\n(colored by age)')

    plt.tight_layout()
    plt.savefig(f'{run_dir}/epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def main():
    print("=" * 70)
    print("RISING TIDE v3 - Soft Freeze (LR Decay)")
    print("=" * 70)
    print(f"Tide: {MIN_SCALE} -> {MAX_SCALE} over {EPOCHS} epochs")
    print(f"Growth: {NEURONS_PER_EPOCH} neurons/epoch")
    print(f"LR Decay: halflife={LR_DECAY_HALFLIFE} epochs, floor={MIN_LR_MULTIPLIER}")
    print("No hard freezing - all neurons can always learn!")
    print("=" * 70)

    text = load_shakespeare()
    print(f"Loaded {len(text):,} characters")

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")

    model = SoftFreezeCrystalLM(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        num_neurons=INITIAL_NEURONS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/rising_v3_{timestamp}"
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
        tide_history.append(current_tide)

        batches = create_batches(tokens, BATCH_SIZE, CONTEXT_LEN)

        total_loss = 0
        for batch in batches:
            batch = batch.to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:]

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()

            # Track gradients before scaling
            model.attention.accumulate_gradients()

            # SOFT FREEZE: Scale gradients by age-based LR multiplier
            model.attention.apply_soft_freeze_to_gradients()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batches)
        loss_history.append(avg_loss)

        # Continuous growth
        grown = 0
        if model.num_neurons < MAX_NEURONS:
            grow = min(NEURONS_PER_EPOCH, MAX_NEURONS - model.num_neurons)
            grown = model.attention.grow_neurons(grow, DEVICE)
            if grown > 0:
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Stats
        stats = model.attention.get_stats()

        # Tide indicator
        tide_bar = "~" * int(current_tide / MAX_SCALE * 10)

        # Sample
        sample = ""
        if epoch % 20 == 0:
            sample = generate(model, tokenizer, "ROMEO:", max_tokens=60, temperature=0.8)
            visualize(model, epoch, run_dir, tide_history, loss_history)

        # Status line
        grow_str = f"+{grown}N" if grown > 0 else ""
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | N: {model.num_neurons} | "
              f"AvgLR: {stats['avg_lr_mult']:.2f} | Slow: {stats['num_slow']} | "
              f"Tide: {current_tide:.2f} [{tide_bar:<10}] {grow_str}")

        if sample:
            print(f"         -> {sample[:70]}...")

        # Save
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'neurons': model.num_neurons,
            'avg_lr_mult': stats['avg_lr_mult'],
            'num_slow': stats['num_slow'],
            'tide': current_tide,
            'stats': stats,
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
    stats = model.attention.get_stats()
    print(f"Final: {model.num_neurons} neurons")
    print(f"LR distribution: avg={stats['avg_lr_mult']:.3f}, min={stats['min_lr_mult']:.3f}, max={stats['max_lr_mult']:.3f}")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 70)

    sample = generate(model, tokenizer, "ROMEO:", max_tokens=200, temperature=0.8)
    print("\nFinal generation:")
    print(sample)

    visualize(model, EPOCHS, run_dir, tide_history, loss_history)


if __name__ == "__main__":
    main()
