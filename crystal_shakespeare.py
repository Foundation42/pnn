"""
Crystal Shakespeare - Growing geometric LM on clean beautiful text
Uses GPT-2's BPE tokenizer for proper subword encoding
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
MAX_NEURONS = 1024  # More room to grow
GROWTH_INTERVAL = 10
FREEZE_THRESHOLD = 0.001  # Base threshold (adjusted by schedule)


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
    """Growing crystal language model"""

    def __init__(self, vocab_size, embed_dim, num_neurons):
        super().__init__()
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        self.attention = GeometricAttention(embed_dim, num_neurons)
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

        # Attention block
        h = h + self.attention(self.norm1(h))

        # FFN block
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
    """Create training batches"""
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


def visualize_crystal(model, epoch, run_dir, sample_text="", pca_model=None, axis_limits=None):
    """Visualize the crystal structure with consistent axes"""
    positions = model.attention.positions.detach().cpu().numpy()
    frozen = model.attention.frozen.cpu().numpy()
    activity = model.attention.gradient_acc.cpu().numpy()

    # PCA to 2D - reuse same PCA if provided for consistency
    if positions.shape[0] > 2:
        if pca_model is None:
            pca_model = PCA(n_components=2)
            pos_2d = pca_model.fit_transform(positions)
        else:
            # Project using existing PCA (pad if needed)
            if positions.shape[1] == pca_model.components_.shape[1]:
                pos_2d = pca_model.transform(positions)
            else:
                pca_model = PCA(n_components=2)
                pos_2d = pca_model.fit_transform(positions)
    else:
        pos_2d = positions[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Crystal structure
    ax = axes[0]
    colors = ['blue' if not f else 'gray' for f in frozen]
    sizes = 50 + activity * 500
    sizes = np.clip(sizes, 20, 300)

    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=sizes, alpha=0.6)
    ax.set_title(f'Crystal Structure - Epoch {epoch}\n{model.num_neurons} neurons, {model.num_frozen} frozen ({100*model.num_frozen/model.num_neurons:.0f}%)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Use fixed axis limits if provided
    if axis_limits:
        ax.set_xlim(axis_limits['x'])
        ax.set_ylim(axis_limits['y'])
    else:
        # Set reasonable default limits
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    # Sample text
    ax = axes[1]
    ax.text(0.1, 0.5, sample_text[:500], fontsize=9, family='monospace',
            verticalalignment='center', wrap=True,
            transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Generated Sample')

    plt.tight_layout()
    plt.savefig(f'{run_dir}/epoch_{epoch:03d}.png', dpi=100)
    plt.close()

    return pca_model


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
    print("Crystal Shakespeare v2 - Graceful Freezing Schedule")
    print("=" * 70)
    print("Phases: GROW (0-20%) -> grow+freeze (20-50%) -> FREEZE (50-80%) -> CRYSTALLIZE (80-100%)")

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
    run_dir = f"runs/crystal_shakespeare_{timestamp}"
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
            'max_neurons': MAX_NEURONS
        }
    }
    pca_model = None
    axis_limits = {'x': (-10, 10), 'y': (-10, 10)}

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
            model.attention.accumulate_gradients()

            # Zero frozen gradients
            if model.attention.positions.grad is not None:
                model.attention.positions.grad[model.attention.frozen] = 0
                model.attention.values.grad[model.attention.frozen] = 0
                model.attention.temperature.grad[model.attention.frozen] = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batches)

        # Growth and freezing
        if epoch % GROWTH_INTERVAL == 0:
            # Update epoch tracker
            model.attention.current_epoch = epoch

            # Graceful freezing
            model.attention.freeze_cold_neurons(epoch, EPOCHS)

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
                model.attention.grow_neurons(grow)

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
            pca_model = visualize_crystal(model, epoch, run_dir, sample, pca_model, axis_limits)

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
    visualize_crystal(model, EPOCHS, run_dir, sample, pca_model, axis_limits)

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
