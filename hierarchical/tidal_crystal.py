"""
Tidal Crystal - Oscillating attention scale (day/night cycle)

Key insight: The attention "aperture" breathes in and out over training.
Neurons that freeze lock in whatever scale the tide was at.

Result: Natural multi-scale structure from the rhythm of the tide!
- Neurons frozen at low tide → local patterns
- Neurons frozen at high tide → global patterns
- The crystal stratifies by when each neuron crystallized

Like sediment layers deposited at different tide levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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

# Tidal config
MIN_SCALE = 0.05      # Low tide (narrow, local)
MAX_SCALE = 0.8       # High tide (wide, global)
CYCLE_LENGTH = 60     # Epochs per full tide cycle
TIDE_MODE = "sawtooth"  # "sawtooth" or "sine"


def get_tide_scale(epoch):
    """Get the current tide level (global attention scale)"""
    phase = (epoch % CYCLE_LENGTH) / CYCLE_LENGTH  # 0 to 1

    if TIDE_MODE == "sawtooth":
        # Ramps up then resets
        scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * phase
    else:
        # Sinusoidal - smooth breathing
        scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * (0.5 + 0.5 * math.sin(2 * math.pi * phase - math.pi/2))

    return scale


def get_tide_phase_name(epoch):
    """Get a friendly name for the current tide phase"""
    phase = (epoch % CYCLE_LENGTH) / CYCLE_LENGTH
    if TIDE_MODE == "sawtooth":
        if phase < 0.25:
            return "rising"
        elif phase < 0.5:
            return "rising+"
        elif phase < 0.75:
            return "high"
        else:
            return "peak→reset"
    else:
        if phase < 0.25:
            return "low→rising"
        elif phase < 0.5:
            return "high"
        elif phase < 0.75:
            return "high→falling"
        else:
            return "low"


class TidalGeometricAttention(nn.Module):
    """
    Geometric attention where the scale oscillates with a global tide.
    Frozen neurons lock in the tide level when they crystallized.
    """

    def __init__(self, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.5)
        self.values = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(num_neurons) * 0.5)

        # Per-neuron frozen scale (only used when frozen)
        self.register_buffer('frozen_scale', torch.zeros(num_neurons))

        # Tracking
        self.register_buffer('gradient_acc', torch.zeros(num_neurons))
        self.register_buffer('gradient_history', torch.zeros(num_neurons, 10))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('frozen', torch.zeros(num_neurons, dtype=torch.bool))
        self.register_buffer('birth_epoch', torch.zeros(num_neurons))
        self.register_buffer('freeze_epoch', torch.zeros(num_neurons))  # When each neuron froze
        self.current_epoch = 0
        self.current_tide = MIN_SCALE

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        # Distance from tokens to neurons
        dists = torch.cdist(x_flat, self.positions)

        # Get effective scale for each neuron
        # Frozen neurons use their locked-in scale, active neurons use current tide
        effective_scale = torch.where(
            self.frozen,
            self.frozen_scale,
            torch.full_like(self.frozen_scale, self.current_tide)
        )

        # Temperature modulated by scale
        effective_temp = (self.temperature.abs() + 0.1) * effective_scale

        # RBF attention
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
        """Add new neurons"""
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

            self.frozen_scale = torch.cat([self.frozen_scale, torch.zeros(n_new, device=device)])
            self.gradient_acc = torch.cat([self.gradient_acc, torch.zeros(n_new, device=device)])
            self.gradient_history = torch.cat([self.gradient_history, torch.zeros(n_new, 10, device=device)])
            self.frozen = torch.cat([self.frozen, torch.zeros(n_new, dtype=torch.bool, device=device)])
            self.birth_epoch = torch.cat([self.birth_epoch, torch.full((n_new,), self.current_epoch, device=device)])
            self.freeze_epoch = torch.cat([self.freeze_epoch, torch.zeros(n_new, device=device)])

            self.num_neurons = len(self.positions)
            return n_new
        return 0

    def freeze_cold_neurons(self, epoch, total_epochs):
        """Freeze neurons and lock in current tide level"""

        # Start freezing after first cycle completes
        if epoch < CYCLE_LENGTH * 0.5:
            return 0

        progress = epoch / total_epochs

        # Gradual aggression increase
        if progress < 0.4:
            aggression = 0.2
        elif progress < 0.7:
            aggression = 0.4
        else:
            aggression = 0.7

        min_age = 20
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

        max_freeze = int(candidates.sum().item() * aggression * 0.15)
        max_freeze = max(1, min(max_freeze, 12))

        _, freeze_idx = torch.topk(coldness, min(max_freeze, candidates.sum().item()))

        # Only freeze if actually cold
        median_activity = grad_mean[candidates].median()
        actually_cold = grad_mean[freeze_idx] < median_activity

        to_freeze = freeze_idx[actually_cold]

        # Lock in current tide level!
        self.frozen_scale[to_freeze] = self.current_tide
        self.freeze_epoch[to_freeze] = epoch
        self.frozen[to_freeze] = True

        return len(to_freeze)

    def zero_frozen_grads(self):
        if self.positions.grad is not None:
            self.positions.grad[self.frozen] = 0
            self.values.grad[self.frozen] = 0
            self.temperature.grad[self.frozen] = 0

    def scale_stats(self):
        """Get statistics about frozen scales"""
        frozen_mask = self.frozen

        if frozen_mask.any():
            frozen_scales = self.frozen_scale[frozen_mask]
            return {
                'frozen_mean': frozen_scales.mean().item(),
                'frozen_min': frozen_scales.min().item(),
                'frozen_max': frozen_scales.max().item(),
                'frozen_std': frozen_scales.std().item() if len(frozen_scales) > 1 else 0,
            }
        return {
            'frozen_mean': 0, 'frozen_min': 0, 'frozen_max': 0, 'frozen_std': 0
        }


class TidalCrystalLM(nn.Module):
    """Language model with tidal geometric attention"""

    def __init__(self, vocab_size, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        self.attention = TidalGeometricAttention(embed_dim, num_neurons)
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
    print("Downloading TinyShakespeare...")
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


def visualize_tidal(model, epoch, run_dir, tide_history):
    """Visualize the tidal crystal"""
    positions = model.attention.positions.detach().cpu().numpy()
    frozen = model.attention.frozen.cpu().numpy()
    frozen_scales = model.attention.frozen_scale.cpu().numpy()
    current_tide = model.attention.current_tide

    # PCA
    if positions.shape[0] > 2:
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(positions)
    else:
        pos_2d = positions[:, :2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Crystal structure colored by frozen scale
    ax = axes[0, 0]

    # Active neurons: color by current tide, frozen: by their locked scale
    display_scales = np.where(frozen, frozen_scales, current_tide)

    colors = plt.cm.coolwarm(display_scales / MAX_SCALE)
    sizes = np.where(frozen, 80, 40)
    edge_colors = ['gray' if f else 'black' for f in frozen]

    scatter = ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
                         c=display_scales, cmap='coolwarm',
                         s=sizes, alpha=0.7, edgecolors=edge_colors, linewidths=0.5,
                         vmin=MIN_SCALE, vmax=MAX_SCALE)
    plt.colorbar(scatter, ax=ax, label='Scale (blue=local, red=global)')

    n_frozen = frozen.sum()
    ax.set_title(f'Tidal Crystal - Epoch {epoch}\n{len(frozen)}N, {n_frozen}F, Tide={current_tide:.2f}')

    # 2. Tide history
    ax = axes[0, 1]
    epochs_plot = list(range(1, len(tide_history) + 1))
    ax.plot(epochs_plot, tide_history, 'b-', linewidth=1.5)
    ax.axhline(y=current_tide, color='r', linestyle='--', alpha=0.5)
    ax.fill_between(epochs_plot, MIN_SCALE, tide_history, alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Tide Level (Attention Scale)')
    ax.set_title(f'Tide History ({TIDE_MODE})')
    ax.set_ylim(0, MAX_SCALE * 1.1)

    # 3. Frozen scale distribution
    ax = axes[1, 0]
    if frozen.any():
        ax.hist(frozen_scales[frozen], bins=20, alpha=0.7, color='steelblue',
                range=(MIN_SCALE, MAX_SCALE), edgecolor='black')
        ax.axvline(x=frozen_scales[frozen].mean(), color='red', linestyle='--',
                   label=f'Mean: {frozen_scales[frozen].mean():.2f}')
        ax.legend()
    ax.set_xlabel('Frozen Scale')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Frozen Scales\n(When did each neuron crystallize?)')

    # 4. Freeze timing vs scale
    ax = axes[1, 1]
    if frozen.any():
        freeze_epochs = model.attention.freeze_epoch[model.attention.frozen].cpu().numpy()
        ax.scatter(freeze_epochs, frozen_scales[frozen], alpha=0.6, c='steelblue')
        ax.set_xlabel('Epoch Frozen')
        ax.set_ylabel('Scale When Frozen')
        ax.set_title('Freeze Timing vs Locked Scale')

        # Show tide curve overlay
        ax2 = ax.twinx()
        ax2.plot(epochs_plot, tide_history, 'r-', alpha=0.3, linewidth=2)
        ax2.set_ylabel('Tide Level', color='red', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{run_dir}/tidal_epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def main():
    print("=" * 70)
    print(f"TIDAL CRYSTAL - {TIDE_MODE.upper()} Attention Waves")
    print("=" * 70)
    print(f"Tide range: {MIN_SCALE} (low/local) <-> {MAX_SCALE} (high/global)")
    print(f"Cycle length: {CYCLE_LENGTH} epochs")
    print("Neurons freeze at different tide levels → natural multi-scale!")
    print("=" * 70)

    # Load data
    text = load_shakespeare()
    print(f"Loaded {len(text):,} characters")

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")

    # Create model
    model = TidalCrystalLM(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        num_neurons=INITIAL_NEURONS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/tidal_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nTraining on {DEVICE}")
    print(f"Output: {run_dir}/")
    print("=" * 70)

    best_loss = float('inf')
    tide_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.attention.current_epoch = epoch

        # Update tide
        current_tide = get_tide_scale(epoch)
        model.attention.current_tide = current_tide
        tide_history.append(current_tide)

        tide_phase = get_tide_phase_name(epoch)

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
            visualize_tidal(model, epoch, run_dir, tide_history)

        # Tide indicator
        tide_bar = "~" * int(current_tide / MAX_SCALE * 10)

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | N: {model.num_neurons} | F: {model.num_frozen} ({frozen_pct:.0f}%) | "
              f"Tide: {current_tide:.2f} [{tide_bar:<10}] {tide_phase} | "
              f"F_scale: {scale_stats['frozen_mean']:.2f}±{scale_stats['frozen_std']:.2f}")

        if sample:
            print(f"         -> {sample[:70]}...")

        # Save history
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'neurons': model.num_neurons,
            'frozen': model.num_frozen,
            'tide': current_tide,
            'tide_phase': tide_phase,
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
                'epoch': epoch,
                'loss': avg_loss
            }, f'{run_dir}/best_model.pt')

    # Final
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Final: {model.num_neurons} neurons, {model.num_frozen} frozen ({100*model.num_frozen/model.num_neurons:.0f}%)")
    print(f"Best loss: {best_loss:.4f}")

    scale_stats = model.attention.scale_stats()
    print(f"\nFrozen scale distribution:")
    print(f"  Mean: {scale_stats['frozen_mean']:.3f}")
    print(f"  Range: [{scale_stats['frozen_min']:.3f}, {scale_stats['frozen_max']:.3f}]")
    print(f"  Std: {scale_stats['frozen_std']:.3f}")
    print("=" * 70)

    sample = generate(model, tokenizer, "ROMEO:", max_tokens=200, temperature=0.8)
    print("\nFinal generation:")
    print(sample)

    # Final visualization
    visualize_tidal(model, EPOCHS, run_dir, tide_history)


if __name__ == "__main__":
    main()
