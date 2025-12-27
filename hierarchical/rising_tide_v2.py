"""
Rising Tide v2 - Age-based freezing tied to birth scale

Key insight: Local patterns lock in FAST. Global patterns need time.

- Neurons born early (low scale) → freeze after ~30 epochs
- Neurons born late (high scale) → freeze after ~200 epochs
- Crystal solidifies from the inside out!

No gradient-based freezing. Pure age + scale.
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
INITIAL_NEURONS = 2
MAX_NEURONS = 1024
NEURONS_PER_EPOCH = 2  # Continuous growth: add 2 neurons every epoch

# Rising tide config
MIN_SCALE = 0.05
MAX_SCALE = 0.80

# Age-based freezing config
MIN_LIFESPAN = 30    # Local neurons (scale ~0.05) live this long
MAX_LIFESPAN = 200   # Global neurons (scale ~0.80) live this long


def get_tide_level(epoch, total_epochs=EPOCHS):
    """Linear rise from MIN to MAX over training"""
    progress = epoch / total_epochs
    return MIN_SCALE + (MAX_SCALE - MIN_SCALE) * progress


def get_lifespan(birth_scale):
    """How long a neuron lives before freezing, based on its birth scale"""
    # Normalize scale to 0-1 range
    scale_normalized = (birth_scale - MIN_SCALE) / (MAX_SCALE - MIN_SCALE)
    # Linear interpolation between min and max lifespan
    return MIN_LIFESPAN + scale_normalized * (MAX_LIFESPAN - MIN_LIFESPAN)


def get_freeze_epoch(birth_epoch, birth_scale):
    """When should this neuron freeze?"""
    lifespan = get_lifespan(birth_scale)
    return birth_epoch + lifespan


class RisingTideAttentionV2(nn.Module):
    """
    Geometric attention with:
    - Birth-imprinted scale (from rising tide)
    - Age-based freezing (local freezes fast, global freezes slow)
    """

    def __init__(self, embed_dim, num_neurons, birth_epoch=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.5)
        self.values = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(num_neurons) * 0.5)

        # Birth info
        initial_scale = get_tide_level(birth_epoch)
        self.register_buffer('neuron_scale', torch.ones(num_neurons) * initial_scale)
        self.register_buffer('birth_epoch', torch.ones(num_neurons) * birth_epoch)

        # Freeze epoch (pre-computed from birth scale)
        freeze_ep = get_freeze_epoch(birth_epoch, initial_scale)
        self.register_buffer('freeze_epoch', torch.ones(num_neurons) * freeze_ep)

        # Frozen status
        self.register_buffer('frozen', torch.zeros(num_neurons, dtype=torch.bool))

        # Gradient tracking (for growth decisions only, not freezing)
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

    def accumulate_gradients(self):
        """Track gradients for growth decisions"""
        if self.positions.grad is not None:
            grad_norm = self.positions.grad.norm(dim=1)
            self.gradient_acc = 0.9 * self.gradient_acc + 0.1 * grad_norm.detach()

    def check_age_freezing(self, current_epoch):
        """Freeze neurons that have reached their freeze epoch"""
        # Find neurons that should freeze now
        should_freeze = (~self.frozen) & (current_epoch >= self.freeze_epoch)

        newly_frozen = should_freeze.sum().item()
        self.frozen[should_freeze] = True

        return newly_frozen

    def grow_neurons(self, num_new, device):
        """Add new neurons with current tide level"""
        if num_new <= 0:
            return 0

        # Find hottest non-frozen neurons to split
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

            # New neurons get current tide as birth scale
            current_scale = get_tide_level(self.current_epoch)
            current_freeze_epoch = get_freeze_epoch(self.current_epoch, current_scale)

            self.neuron_scale = torch.cat([
                self.neuron_scale,
                torch.ones(n_new, device=device) * current_scale
            ])
            self.birth_epoch = torch.cat([
                self.birth_epoch,
                torch.ones(n_new, device=device) * self.current_epoch
            ])
            self.freeze_epoch = torch.cat([
                self.freeze_epoch,
                torch.ones(n_new, device=device) * current_freeze_epoch
            ])
            self.frozen = torch.cat([
                self.frozen,
                torch.zeros(n_new, dtype=torch.bool, device=device)
            ])
            self.gradient_acc = torch.cat([
                self.gradient_acc,
                torch.zeros(n_new, device=device)
            ])

            self.num_neurons = len(self.positions)
            return n_new
        return 0

    def zero_frozen_grads(self):
        if self.positions.grad is not None:
            self.positions.grad[self.frozen] = 0
            self.values.grad[self.frozen] = 0
            self.temperature.grad[self.frozen] = 0

    def get_stats(self):
        """Get detailed statistics"""
        active_mask = ~self.frozen
        frozen_mask = self.frozen

        stats = {
            'num_active': active_mask.sum().item(),
            'num_frozen': frozen_mask.sum().item(),
        }

        if active_mask.any():
            stats['active_scale_mean'] = self.neuron_scale[active_mask].mean().item()
            stats['active_scale_min'] = self.neuron_scale[active_mask].min().item()
            stats['active_scale_max'] = self.neuron_scale[active_mask].max().item()
            stats['active_age_mean'] = (self.current_epoch - self.birth_epoch[active_mask]).mean().item()
        else:
            stats['active_scale_mean'] = 0
            stats['active_scale_min'] = 0
            stats['active_scale_max'] = 0
            stats['active_age_mean'] = 0

        if frozen_mask.any():
            stats['frozen_scale_mean'] = self.neuron_scale[frozen_mask].mean().item()
            stats['frozen_scale_min'] = self.neuron_scale[frozen_mask].min().item()
            stats['frozen_scale_max'] = self.neuron_scale[frozen_mask].max().item()
        else:
            stats['frozen_scale_mean'] = 0
            stats['frozen_scale_min'] = 0
            stats['frozen_scale_max'] = 0

        return stats


class RisingTideCrystalLMV2(nn.Module):
    """Language model with rising tide + age-based freezing"""

    def __init__(self, vocab_size, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        self.attention = RisingTideAttentionV2(embed_dim, num_neurons)
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

    @property
    def num_active(self):
        return self.num_neurons - self.num_frozen


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
    """Visualize the crystal"""
    positions = model.attention.positions.detach().cpu().numpy()
    frozen = model.attention.frozen.cpu().numpy()
    scales = model.attention.neuron_scale.cpu().numpy()
    birth_epochs = model.attention.birth_epoch.cpu().numpy()

    if positions.shape[0] > 2:
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(positions)
    else:
        pos_2d = positions[:, :2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Crystal colored by scale, frozen shown differently
    ax = axes[0, 0]

    # Active neurons: colored by scale, smaller
    active_mask = ~frozen
    if active_mask.any():
        ax.scatter(pos_2d[active_mask, 0], pos_2d[active_mask, 1],
                   c=scales[active_mask], cmap='coolwarm',
                   s=40, alpha=0.8, edgecolors='black', linewidths=0.5,
                   vmin=MIN_SCALE, vmax=MAX_SCALE, label='Active')

    # Frozen neurons: colored by scale, larger, different edge
    if frozen.any():
        scatter = ax.scatter(pos_2d[frozen, 0], pos_2d[frozen, 1],
                   c=scales[frozen], cmap='coolwarm',
                   s=100, alpha=0.6, edgecolors='gold', linewidths=2,
                   vmin=MIN_SCALE, vmax=MAX_SCALE, marker='s', label='Frozen')

    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(MIN_SCALE, MAX_SCALE)),
                 ax=ax, label='Scale')

    n_frozen = frozen.sum()
    n_total = len(frozen)
    current_tide = get_tide_level(epoch)
    ax.set_title(f'Crystal - Epoch {epoch}\n{n_total}N, {n_frozen}F ({100*n_frozen/n_total:.0f}%), Tide={current_tide:.2f}')
    ax.legend()

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

    # 3. Birth epoch vs scale (should show rising tide line)
    ax = axes[1, 0]
    colors = ['gold' if f else 'steelblue' for f in frozen]
    ax.scatter(birth_epochs, scales, c=colors, alpha=0.6, s=30)

    # Overlay the tide line
    ep_range = np.linspace(1, epoch, 100)
    tide_line = [get_tide_level(e) for e in ep_range]
    ax.plot(ep_range, tide_line, 'k--', alpha=0.5, linewidth=2, label='Tide line')

    ax.set_xlabel('Birth Epoch')
    ax.set_ylabel('Scale')
    ax.set_title('Birth Time vs Scale\nGold=Frozen, Blue=Active')
    ax.legend()

    # 4. Age distribution of active vs frozen
    ax = axes[1, 1]
    ages = epoch - birth_epochs

    if active_mask.any():
        ax.hist(ages[active_mask], bins=20, alpha=0.7, label='Active', color='steelblue')
    if frozen.any():
        ax.hist(ages[frozen], bins=20, alpha=0.7, label='Frozen', color='gold')

    ax.set_xlabel('Age (epochs)')
    ax.set_ylabel('Count')
    ax.set_title('Neuron Age Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{run_dir}/epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def main():
    print("=" * 70)
    print("RISING TIDE v2 - Continuous growth + Age-based freezing")
    print("=" * 70)
    print(f"Tide: {MIN_SCALE} -> {MAX_SCALE} over {EPOCHS} epochs")
    print(f"Growth: {NEURONS_PER_EPOCH} neurons/epoch (smooth, no batch discontinuities)")
    print(f"Lifespan: {MIN_LIFESPAN} epochs (local) -> {MAX_LIFESPAN} epochs (global)")
    print("Local neurons freeze fast, global neurons freeze slow!")
    print("=" * 70)

    text = load_shakespeare()
    print(f"Loaded {len(text):,} characters")

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")

    model = RisingTideCrystalLMV2(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        num_neurons=INITIAL_NEURONS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/rising_v2_{timestamp}"
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

            model.attention.accumulate_gradients()
            model.attention.zero_frozen_grads()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batches)
        loss_history.append(avg_loss)

        # Age-based freezing - check every epoch!
        newly_frozen = model.attention.check_age_freezing(epoch)

        # Continuous growth - add neurons every epoch (smooth, no discontinuities)
        grown = 0
        if model.num_neurons < MAX_NEURONS:
            grow = min(NEURONS_PER_EPOCH, MAX_NEURONS - model.num_neurons)
            grown = model.attention.grow_neurons(grow, DEVICE)
            if grown > 0:
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Stats
        stats = model.attention.get_stats()
        frozen_pct = 100 * model.num_frozen / model.num_neurons

        # Tide indicator
        tide_bar = "~" * int(current_tide / MAX_SCALE * 10)

        # Sample
        sample = ""
        if epoch % 20 == 0:
            sample = generate(model, tokenizer, "ROMEO:", max_tokens=60, temperature=0.8)
            visualize(model, epoch, run_dir, tide_history, loss_history)

        # Status line
        changes = []
        if grown > 0:
            changes.append(f"+{grown}N")
        if newly_frozen > 0:
            changes.append(f"+{newly_frozen}F")
        change_str = " ".join(changes) if changes else ""

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | N: {model.num_neurons} A: {model.num_active} F: {model.num_frozen} ({frozen_pct:.0f}%) {change_str} | "
              f"Tide: {current_tide:.2f} [{tide_bar:<10}] | "
              f"A_scale: {stats['active_scale_mean']:.2f}")

        if sample:
            print(f"         -> {sample[:70]}...")

        # Save
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'neurons': model.num_neurons,
            'frozen': model.num_frozen,
            'newly_frozen': newly_frozen,
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
    print(f"Final: {model.num_neurons} neurons, {model.num_frozen} frozen ({100*model.num_frozen/model.num_neurons:.0f}%)")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 70)

    sample = generate(model, tokenizer, "ROMEO:", max_tokens=200, temperature=0.8)
    print("\nFinal generation:")
    print(sample)

    visualize(model, EPOCHS, run_dir, tide_history, loss_history)


if __name__ == "__main__":
    main()
