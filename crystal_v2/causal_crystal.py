"""
Causal Crystal - Geometric LM with cross-position attention!

The baseline crystal had a critical flaw: each position could only see ITSELF.
Tokens queried the neuron field independently with no knowledge of context.

This version adds CAUSAL SELF-ATTENTION before the geometric attention:
1. Tokens first attend to previous tokens (build context)
2. Then query the geometric neuron field with contextualized representations
3. Neurons now learn patterns over SEQUENCES, not isolated positions

Let's see if this breaks the 2.55 plateau!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import requests
import tiktoken  # GPT-2's BPE tokenizer

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 128
CONTEXT_LEN = 64
BATCH_SIZE = 32
EPOCHS = 600
INITIAL_NEURONS = 64
MAX_NEURONS = 1024
GROWTH_INTERVAL = 10
FREEZE_THRESHOLD = 0.001

# Causal attention config
NUM_HEADS = 4  # Multi-head attention


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention so tokens can see previous tokens"""

    def __init__(self, embed_dim, num_heads, context_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask - lower triangular
        mask = torch.tril(torch.ones(context_len, context_len))
        self.register_buffer('mask', mask.view(1, 1, context_len, context_len))

    def forward(self, x):
        B, T, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D/H)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Apply causal mask - positions can only attend to previous positions
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # Gather values
        out = attn @ v  # (B, H, T, D/H)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)


class GeometricAttention(nn.Module):
    """Attention through geometric proximity in embedding space"""

    def __init__(self, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        # Neuron positions in embedding space
        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.5)

        # Neuron properties
        self.values = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(num_neurons) * 0.5)

        # Track activity for growth/freeze
        self.register_buffer('gradient_acc', torch.zeros(num_neurons))
        self.register_buffer('gradient_history', torch.zeros(num_neurons, 10))  # Last 10 readings
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('frozen', torch.zeros(num_neurons, dtype=torch.bool))
        self.register_buffer('birth_epoch', torch.zeros(num_neurons))  # When each neuron was created
        self.register_buffer('update_count', torch.tensor(0))
        self.current_epoch = 0

    def forward(self, x):
        B, T, D = x.shape

        # Distance from tokens to neurons
        x_flat = x.reshape(-1, D)  # (B*T, D)

        # Compute distances
        dists = torch.cdist(x_flat, self.positions)  # (B*T, N)

        # RBF attention weights
        weights = torch.exp(-dists / (self.temperature.abs() + 0.1))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Gather values
        out = weights @ self.values  # (B*T, D)

        return out.reshape(B, T, D)

    def accumulate_gradients(self):
        if self.positions.grad is not None:
            grad_norm = self.positions.grad.norm(dim=1)
            self.gradient_acc = 0.9 * self.gradient_acc + 0.1 * grad_norm.detach()

            # Store in history ring buffer
            ptr = self.history_ptr.item() % 10
            self.gradient_history[:, ptr] = grad_norm.detach()
            self.history_ptr += 1

            self.update_count += 1

    def grow_neurons(self, num_new):
        """Add new neurons by splitting the hottest ones"""
        if num_new <= 0:
            return

        # Find hottest non-frozen neurons
        activity = self.gradient_acc.clone()
        activity[self.frozen] = -float('inf')
        _, hot_idx = torch.topk(activity, min(num_new, (~self.frozen).sum()))

        # Create new neurons near hot ones
        new_positions = []
        new_values = []
        new_temps = []

        for idx in hot_idx:
            offset = torch.randn(self.embed_dim, device=self.positions.device) * 0.1
            new_positions.append(self.positions[idx] + offset)
            new_values.append(self.values[idx] * 0.5)
            new_temps.append(self.temperature[idx].clone())

        if new_positions:
            # Expand parameters
            new_pos = torch.stack(new_positions)
            new_val = torch.stack(new_values)
            new_temp = torch.stack(new_temps)

            self.positions = nn.Parameter(torch.cat([self.positions.data, new_pos]))
            self.values = nn.Parameter(torch.cat([self.values.data, new_val]))
            self.temperature = nn.Parameter(torch.cat([self.temperature.data, new_temp]))

            # Expand buffers
            n_new = len(new_positions)
            self.gradient_acc = torch.cat([self.gradient_acc, torch.zeros(n_new, device=DEVICE)])
            self.gradient_history = torch.cat([self.gradient_history, torch.zeros(n_new, 10, device=DEVICE)])
            self.frozen = torch.cat([self.frozen, torch.zeros(n_new, dtype=torch.bool, device=DEVICE)])
            self.birth_epoch = torch.cat([self.birth_epoch, torch.full((n_new,), self.current_epoch, device=DEVICE)])

            self.num_neurons = len(self.positions)

    def freeze_cold_neurons(self, epoch, total_epochs):
        """Graceful freezing schedule - more aggressive over time"""

        # Phase 1: No freezing for first 20% of training (pure growth)
        if epoch < total_epochs * 0.2:
            return 0

        # Phase 2: Very conservative freezing 20-50%
        # Phase 3: Moderate freezing 50-80%
        # Phase 4: Aggressive freezing 80-100%

        progress = epoch / total_epochs

        # Calculate freeze aggression (0 to 1)
        if progress < 0.5:
            aggression = (progress - 0.2) / 0.3 * 0.3  # 0 to 0.3
        elif progress < 0.8:
            aggression = 0.3 + (progress - 0.5) / 0.3 * 0.3  # 0.3 to 0.6
        else:
            aggression = 0.6 + (progress - 0.8) / 0.2 * 0.4  # 0.6 to 1.0

        # Don't freeze neurons younger than 50 epochs
        min_age = 50
        neuron_age = epoch - self.birth_epoch
        too_young = neuron_age < min_age

        # Compute stability: neurons with consistently low gradient variance are stable
        if self.history_ptr < 10:
            return 0  # Need full history

        grad_variance = self.gradient_history.var(dim=1)
        grad_mean = self.gradient_history.mean(dim=1)

        # Candidates: not frozen, old enough, low mean activity, low variance (stable)
        candidates = (~self.frozen) & (~too_young)
        if candidates.sum() < 10:
            return 0

        # Score by coldness (low activity + stable)
        coldness = torch.zeros(self.num_neurons, device=DEVICE)
        coldness[candidates] = -grad_mean[candidates] - grad_variance[candidates] * 10

        # How many to freeze this round (scales with aggression)
        max_freeze = int(candidates.sum().item() * aggression * 0.1)  # Up to 10% of candidates
        max_freeze = max(1, min(max_freeze, 15))  # Between 1 and 15

        # Get coldest candidates
        coldness[~candidates] = float('-inf')
        _, freeze_idx = torch.topk(coldness, min(max_freeze, candidates.sum().item()))

        # Only freeze if they're truly cold (below median activity)
        median_activity = grad_mean[candidates].median()
        actually_cold = grad_mean[freeze_idx] < median_activity

        to_freeze = freeze_idx[actually_cold]
        self.frozen[to_freeze] = True

        # Zero gradients for frozen neurons
        if self.positions.grad is not None:
            self.positions.grad[self.frozen] = 0
            self.values.grad[self.frozen] = 0
            self.temperature.grad[self.frozen] = 0

        return len(to_freeze)


class CrystalLM(nn.Module):
    """Growing crystal language model with causal attention"""

    def __init__(self, vocab_size, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        # NEW: Causal self-attention - lets tokens see previous tokens
        self.causal_attn = CausalSelfAttention(embed_dim, NUM_HEADS, CONTEXT_LEN)
        self.norm_causal = nn.LayerNorm(embed_dim)

        # Geometric attention - queries the neuron field
        self.geo_attn = GeometricAttention(embed_dim, num_neurons)
        self.norm_geo = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        pos = torch.arange(T, device=x.device)
        h = self.token_embed(x) + self.pos_embed(pos)

        # 1. Causal self-attention - build context from previous tokens
        h = h + self.causal_attn(self.norm_causal(h))

        # 2. Geometric attention - query neuron field with contextualized tokens
        h = h + self.geo_attn(self.norm_geo(h))

        # 3. FFN
        h = h + self.ffn(self.norm_ffn(h))

        logits = self.head(h)

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return logits

    @property
    def num_neurons(self):
        return self.geo_attn.num_neurons

    @property
    def num_frozen(self):
        return self.geo_attn.frozen.sum().item()


def load_shakespeare():
    """Load TinyShakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_file = "data/tinyshakespeare.txt"

    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            text = f.read()
    else:
        print("Downloading TinyShakespeare...")
        response = requests.get(url)
        text = response.text
        with open(cache_file, 'w') as f:
            f.write(text)

    return text


def create_batches(tokens, batch_size, context_len):
    """Create training batches with overlapping sequences"""
    sequences = []
    for i in range(0, len(tokens) - context_len - 1, context_len // 2):
        seq = tokens[i:i + context_len + 1]
        if len(seq) == context_len + 1:
            sequences.append(seq)

    # Shuffle and batch
    indices = torch.randperm(len(sequences))
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        if len(batch_idx) == batch_size:
            batch = torch.stack([torch.tensor(sequences[j]) for j in batch_idx])
            batches.append(batch)

    return batches


def visualize_crystal(model, epoch, run_dir, loss_history, neuron_history, sample_text=""):
    """Visualize the crystal with 2x2 layout like rising tide experiments"""
    positions = model.geo_attn.positions.detach().cpu().numpy()
    frozen = model.geo_attn.frozen.cpu().numpy()
    activity = model.geo_attn.gradient_acc.cpu().numpy()

    # PCA to 2D
    if positions.shape[0] > 2:
        pca = PCA(n_components=2)
        pos_2d = pca.fit_transform(positions)
    else:
        pos_2d = positions[:, :2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Crystal structure - colored by frozen status, sized by activity
    ax = axes[0, 0]
    colors = np.where(frozen, 0.3, activity / (activity.max() + 1e-8))
    sizes = np.clip(30 + activity * 300, 20, 200)

    scatter = ax.scatter(pos_2d[:, 0], pos_2d[:, 1],
                         c=colors, cmap='coolwarm',
                         s=sizes, alpha=0.7,
                         edgecolors='black', linewidths=0.3,
                         vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax, label='Activity (gray=frozen)')

    frozen_pct = 100 * model.num_frozen / model.num_neurons
    ax.set_title(f'Crystal Structure - Epoch {epoch}\n{model.num_neurons} neurons, {model.num_frozen} frozen ({frozen_pct:.0f}%)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Adaptive axis limits with padding
    if len(pos_2d) > 0:
        pad = 0.2
        x_range = pos_2d[:, 0].max() - pos_2d[:, 0].min()
        y_range = pos_2d[:, 1].max() - pos_2d[:, 1].min()
        ax.set_xlim(pos_2d[:, 0].min() - pad * x_range, pos_2d[:, 0].max() + pad * x_range)
        ax.set_ylim(pos_2d[:, 1].min() - pad * y_range, pos_2d[:, 1].max() + pad * y_range)

    # 2. Loss + Neurons over time
    ax = axes[0, 1]
    epochs_plot = list(range(1, len(loss_history) + 1))

    ax.plot(epochs_plot, loss_history, 'r-', linewidth=2, label='Loss')
    ax.set_ylabel('Loss', color='red')
    ax.set_xlabel('Epoch')

    ax2 = ax.twinx()
    ax2.plot(epochs_plot, neuron_history, 'b-', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Neurons', color='blue')

    ax.set_title('Training Progress')
    ax.legend(loc='upper right')

    # 3. Activity distribution
    ax = axes[1, 0]
    active_activity = activity[~frozen]
    frozen_activity = activity[frozen]

    if len(active_activity) > 0:
        ax.hist(active_activity, bins=30, alpha=0.7, label=f'Active ({len(active_activity)})', color='blue')
    if len(frozen_activity) > 0:
        ax.hist(frozen_activity, bins=30, alpha=0.5, label=f'Frozen ({len(frozen_activity)})', color='gray')

    ax.set_xlabel('Gradient Activity')
    ax.set_ylabel('Count')
    ax.set_title('Neuron Activity Distribution')
    ax.legend()

    # 4. Generated sample
    ax = axes[1, 1]
    # Wrap text nicely
    wrapped = '\n'.join([sample_text[i:i+60] for i in range(0, min(len(sample_text), 480), 60)])
    ax.text(0.05, 0.95, wrapped, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Generated Sample')

    plt.tight_layout()
    plt.savefig(f'{run_dir}/epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def generate(model, tokenizer, prompt="ROMEO:", max_tokens=100, temperature=0.8):
    """Generate text from prompt"""
    model.eval()

    tokens = tokenizer.encode(prompt)
    tokens = tokens[-CONTEXT_LEN:]  # Truncate if needed

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([tokens[-CONTEXT_LEN:]], device=DEVICE)
            logits = model(x)
            logits = logits[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

    return tokenizer.decode(tokens)


def main():
    print("=" * 70)
    print("CAUSAL CRYSTAL - Now tokens can see previous tokens!")
    print("=" * 70)
    print("Architecture: Causal Self-Attention -> Geometric Attention -> FFN")
    print(f"Causal heads: {NUM_HEADS}")

    # Load data
    text = load_shakespeare()
    print(f"Loaded {len(text):,} characters of Shakespeare")

    # BPE tokenization
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")
    print(f"Vocab size: {tokenizer.n_vocab:,}")

    # Create model
    model = CrystalLM(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        num_neurons=INITIAL_NEURONS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Run directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/causal_crystal_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nTraining on {DEVICE}")
    print(f"Initial neurons: {INITIAL_NEURONS}, Max: {MAX_NEURONS}")
    print(f"Output: {run_dir}/")
    print("=" * 70)

    best_loss = float('inf')
    history = {
        'loss': [],
        'neurons': [],
        'frozen': [],
        'samples': [],
        'config': {
            'epochs': EPOCHS,
            'embed_dim': EMBED_DIM,
            'context_len': CONTEXT_LEN,
            'batch_size': BATCH_SIZE,
            'initial_neurons': INITIAL_NEURONS,
            'max_neurons': MAX_NEURONS,
            'num_heads': NUM_HEADS
        }
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batches = create_batches(tokens, BATCH_SIZE, CONTEXT_LEN)

        total_loss = 0
        for batch in batches:
            batch = batch.to(DEVICE)
            x, y = batch[:, :-1], batch[:, 1:]

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()

            # Track gradients before step
            model.geo_attn.accumulate_gradients()

            # Zero frozen gradients
            if model.geo_attn.positions.grad is not None:
                model.geo_attn.positions.grad[model.geo_attn.frozen] = 0
                model.geo_attn.values.grad[model.geo_attn.frozen] = 0
                model.geo_attn.temperature.grad[model.geo_attn.frozen] = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batches)

        # Growth and freezing
        if epoch % GROWTH_INTERVAL == 0:
            # Update epoch tracker
            model.geo_attn.current_epoch = epoch

            # Graceful freezing
            model.geo_attn.freeze_cold_neurons(epoch, EPOCHS)

            # Grow if below max - be aggressive early, taper off later
            if model.num_neurons < MAX_NEURONS:
                active = model.num_neurons - model.num_frozen
                progress = epoch / EPOCHS

                # Growth rate: aggressive early, conservative late
                if progress < 0.3:
                    base_grow = 24  # Fast early growth
                elif progress < 0.6:
                    base_grow = 12  # Moderate mid-training
                else:
                    base_grow = 4   # Slow late growth

                grow = min(base_grow, MAX_NEURONS - model.num_neurons, max(active // 2, 2))
                model.geo_attn.grow_neurons(grow)

                # Update optimizer
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Generate sample
        sample = generate(model, tokenizer, "ROMEO:", max_tokens=80, temperature=0.8)

        # Track history
        history['loss'].append(avg_loss)
        history['neurons'].append(model.num_neurons)
        history['frozen'].append(model.num_frozen)
        history['samples'].append(sample[:200])  # Save truncated sample

        # Calculate speedup (frozen neurons = free computation)
        active_neurons = model.num_neurons - model.num_frozen
        speedup = model.num_neurons / max(active_neurons, 1)

        # Determine phase
        progress = epoch / EPOCHS
        if progress < 0.2:
            phase = "GROW"
        elif progress < 0.5:
            phase = "grow+freeze"
        elif progress < 0.8:
            phase = "FREEZE"
        else:
            phase = "CRYSTALLIZE"

        # Progress
        frozen_pct = 100 * model.num_frozen / model.num_neurons
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | N: {model.num_neurons} | F: {model.num_frozen} ({frozen_pct:.0f}%) | S: {speedup:.1f}x | {phase}")
        print(f"         -> {sample[:80]}...")

        # Visualize every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            visualize_crystal(model, epoch, run_dir, history['loss'], history['neurons'], sample)

        # Append to JSONL after every epoch (safe incremental save)
        import json
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'neurons': model.num_neurons,
            'frozen': model.num_frozen,
            'sample': sample[:200],
            'phase': phase
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
                'history': history
            }, f'{run_dir}/best_model.pt')

    # Final visualization
    sample = generate(model, tokenizer, "ROMEO:", max_tokens=200, temperature=0.8)
    visualize_crystal(model, EPOCHS, run_dir, history['loss'], history['neurons'], sample)

    # Save config as separate JSON (history is already in JSONL)
    import json
    config_data = {
        'config': history['config'],
        'final_neurons': model.num_neurons,
        'final_frozen': model.num_frozen,
        'best_loss': best_loss,
        'total_epochs': EPOCHS
    }
    with open(f'{run_dir}/config.json', 'w') as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config to {run_dir}/config.json")
    print(f"History saved incrementally to {run_dir}/history.jsonl")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Final: {model.num_neurons} neurons, {model.num_frozen} frozen ({100*model.num_frozen/model.num_neurons:.0f}%)")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 70)
    print("\nFinal generation:")
    print(sample)


if __name__ == "__main__":
    main()
