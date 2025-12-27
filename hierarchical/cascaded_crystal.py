"""
Cascaded Crystal - Hierarchical Geometric Language Model

Breaking through the loss plateau by adding depth!

Key insight: Flat geometry captures spatial relationships,
but language needs HIERARCHICAL abstraction.

Architecture:
    Token → Layer 0 → Layer 1 → ... → Layer N → Output
           (syntax)  (phrases)       (context)

Each layer is a crystal that can grow and freeze independently.
Early layers freeze more aggressively (universal syntax patterns).
Late layers stay flexible (context-specific).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import requests
import tiktoken
from datetime import datetime
import json

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 128
CONTEXT_LEN = 64
BATCH_SIZE = 32
EPOCHS = 600
MAX_SEQUENCES = 10000  # Cap sequences per epoch for large corpora

# HIERARCHICAL CONFIG
NUM_LAYERS = 4  # Start with 4 layers, can tune
NEURONS_PER_LAYER = 128  # Smaller per layer since we have multiple
MAX_NEURONS_PER_LAYER = 256

# Per-layer freeze aggressiveness: early layers freeze more
# Values are multipliers on base freeze threshold
LAYER_FREEZE_MULTIPLIER = [0.5, 0.7, 1.0, 1.5]  # Layer 0 freezes 2x faster

# Per-layer growth rates
LAYER_GROWTH_RATE = [8, 6, 4, 2]  # Early layers grow faster

GROWTH_INTERVAL = 10


class GeometricAttention(nn.Module):
    """Attention through geometric proximity in embedding space"""

    def __init__(self, embed_dim, num_neurons, layer_idx=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons
        self.layer_idx = layer_idx

        # Neuron positions in embedding space
        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.5)

        # Neuron properties
        self.values = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(num_neurons) * 0.5)

        # Track activity for growth/freeze
        self.register_buffer('gradient_acc', torch.zeros(num_neurons))
        self.register_buffer('gradient_history', torch.zeros(num_neurons, 10))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('frozen', torch.zeros(num_neurons, dtype=torch.bool))
        self.register_buffer('birth_epoch', torch.zeros(num_neurons))
        self.register_buffer('update_count', torch.tensor(0))
        self.current_epoch = 0

    def forward(self, x):
        B, T, D = x.shape

        x_flat = x.reshape(-1, D)

        # Distance from tokens to neurons
        dists = torch.cdist(x_flat, self.positions)

        # RBF attention weights
        weights = torch.exp(-dists / (self.temperature.abs() + 0.1))
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
            self.update_count += 1

    def grow_neurons(self, num_new, device):
        """Add new neurons by splitting the hottest ones"""
        if num_new <= 0:
            return

        # Find hottest non-frozen neurons
        activity = self.gradient_acc.clone()
        activity[self.frozen] = -float('inf')
        num_to_split = min(num_new, (~self.frozen).sum().item())
        if num_to_split == 0:
            return

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
            new_pos = torch.stack(new_positions)
            new_val = torch.stack(new_values)
            new_temp = torch.stack(new_temps)

            self.positions = nn.Parameter(torch.cat([self.positions.data, new_pos]))
            self.values = nn.Parameter(torch.cat([self.values.data, new_val]))
            self.temperature = nn.Parameter(torch.cat([self.temperature.data, new_temp]))

            n_new = len(new_positions)
            self.gradient_acc = torch.cat([self.gradient_acc, torch.zeros(n_new, device=device)])
            self.gradient_history = torch.cat([self.gradient_history, torch.zeros(n_new, 10, device=device)])
            self.frozen = torch.cat([self.frozen, torch.zeros(n_new, dtype=torch.bool, device=device)])
            self.birth_epoch = torch.cat([self.birth_epoch, torch.full((n_new,), self.current_epoch, device=device)])

            self.num_neurons = len(self.positions)

    def freeze_cold_neurons(self, epoch, total_epochs, freeze_multiplier=1.0):
        """Graceful freezing with per-layer adjustment"""

        # No freezing in early phase
        if epoch < total_epochs * 0.2:
            return 0

        progress = epoch / total_epochs

        # Calculate base aggression
        if progress < 0.5:
            aggression = (progress - 0.2) / 0.3 * 0.3
        elif progress < 0.8:
            aggression = 0.3 + (progress - 0.5) / 0.3 * 0.3
        else:
            aggression = 0.6 + (progress - 0.8) / 0.2 * 0.4

        # Apply layer-specific multiplier (lower = freeze more)
        # E.g., multiplier=0.5 means this layer's threshold is halved (freezes more easily)
        aggression = aggression / freeze_multiplier

        min_age = 50
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

        max_freeze = int(candidates.sum().item() * aggression * 0.1)
        max_freeze = max(1, min(max_freeze, 10))

        coldness[~candidates] = float('-inf')
        _, freeze_idx = torch.topk(coldness, min(max_freeze, candidates.sum().item()))

        median_activity = grad_mean[candidates].median()
        actually_cold = grad_mean[freeze_idx] < median_activity

        to_freeze = freeze_idx[actually_cold]
        self.frozen[to_freeze] = True

        return len(to_freeze)

    def zero_frozen_grads(self):
        """Zero gradients for frozen neurons"""
        if self.positions.grad is not None:
            self.positions.grad[self.frozen] = 0
            self.values.grad[self.frozen] = 0
            self.temperature.grad[self.frozen] = 0


class CrystalLayer(nn.Module):
    """A single crystal layer with geometric attention + FFN"""

    def __init__(self, embed_dim, num_neurons, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = GeometricAttention(embed_dim, num_neurons, layer_idx)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        # Pre-norm residual architecture
        h = x + self.attention(self.norm1(x))
        h = h + self.ffn(self.norm2(h))
        return h


class CascadedCrystalLM(nn.Module):
    """Hierarchical crystal language model with multiple layers"""

    def __init__(self, vocab_size, embed_dim, neurons_per_layer, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        # Stack of crystal layers
        self.layers = nn.ModuleList([
            CrystalLayer(embed_dim, neurons_per_layer, layer_idx=i)
            for i in range(num_layers)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        # Embed
        pos = torch.arange(T, device=x.device)
        h = self.token_embed(x) + self.pos_embed(pos)

        # Process through cascade
        for layer in self.layers:
            h = layer(h)

        # Output
        h = self.norm_out(h)
        logits = self.head(h)

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return logits

    def total_neurons(self):
        return sum(layer.attention.num_neurons for layer in self.layers)

    def total_frozen(self):
        return sum(layer.attention.frozen.sum().item() for layer in self.layers)

    def layer_stats(self):
        """Get per-layer neuron/frozen counts"""
        stats = []
        for i, layer in enumerate(self.layers):
            att = layer.attention
            stats.append({
                'layer': i,
                'neurons': att.num_neurons,
                'frozen': att.frozen.sum().item(),
                'frozen_pct': 100 * att.frozen.sum().item() / att.num_neurons
            })
        return stats

    def accumulate_all_gradients(self):
        for layer in self.layers:
            layer.attention.accumulate_gradients()

    def zero_all_frozen_grads(self):
        for layer in self.layers:
            layer.attention.zero_frozen_grads()

    def grow_and_freeze(self, epoch, total_epochs, device):
        """Grow and freeze neurons in all layers"""
        total_grown = 0
        total_frozen = 0

        for i, layer in enumerate(self.layers):
            att = layer.attention
            att.current_epoch = epoch

            # Freeze with layer-specific multiplier
            freeze_mult = LAYER_FREEZE_MULTIPLIER[i] if i < len(LAYER_FREEZE_MULTIPLIER) else 1.0
            n_frozen = att.freeze_cold_neurons(epoch, total_epochs, freeze_mult)
            total_frozen += n_frozen

            # Grow with layer-specific rate
            max_neurons = MAX_NEURONS_PER_LAYER
            if att.num_neurons < max_neurons:
                growth_rate = LAYER_GROWTH_RATE[i] if i < len(LAYER_GROWTH_RATE) else 4
                progress = epoch / total_epochs
                if progress < 0.3:
                    grow = growth_rate
                elif progress < 0.6:
                    grow = growth_rate // 2
                else:
                    grow = growth_rate // 4

                grow = max(1, min(grow, max_neurons - att.num_neurons))
                att.grow_neurons(grow, device)
                total_grown += grow

        return total_grown, total_frozen


def load_shakespeare():
    """Load TinyShakespeare dataset"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_file = "data/tinyshakespeare.txt"

    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            text = f.read()
    else:
        # Try parent directory too
        parent_cache = "../data/tinyshakespeare.txt"
        if os.path.exists(parent_cache):
            with open(parent_cache, 'r') as f:
                text = f.read()
        else:
            print("Downloading TinyShakespeare...")
            response = requests.get(url)
            text = response.text
            with open(cache_file, 'w') as f:
                f.write(text)

    return text


def create_batches(tokens, batch_size, context_len, max_sequences=MAX_SEQUENCES):
    """Create training batches with random sampling for large corpora"""
    n_tokens = len(tokens)
    n_possible = n_tokens - context_len - 1

    if n_possible <= max_sequences:
        sequences = []
        for i in range(0, n_tokens - context_len - 1, context_len // 2):
            seq = tokens[i:i + context_len + 1]
            if len(seq) == context_len + 1:
                sequences.append(seq)
    else:
        starts = np.random.randint(0, n_possible, size=max_sequences)
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
    """Generate text from prompt"""
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


def visualize_cascade(model, epoch, run_dir, sample_text=""):
    """Visualize all crystal layers"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for layer_idx, layer in enumerate(model.layers):
        ax = axes[layer_idx // 2, layer_idx % 2]

        positions = layer.attention.positions.detach().cpu().numpy()
        frozen = layer.attention.frozen.cpu().numpy()
        activity = layer.attention.gradient_acc.cpu().numpy()

        # PCA to 2D
        if positions.shape[0] > 2:
            pca = PCA(n_components=2)
            pos_2d = pca.fit_transform(positions)
        else:
            pos_2d = positions[:, :2] if positions.shape[1] >= 2 else np.zeros((len(positions), 2))

        colors = ['blue' if not f else 'gray' for f in frozen]
        sizes = 30 + activity * 300
        sizes = np.clip(sizes, 15, 200)

        ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=sizes, alpha=0.6)

        n_frozen = frozen.sum()
        n_total = len(frozen)
        frozen_pct = 100 * n_frozen / n_total if n_total > 0 else 0

        layer_names = ['Syntax', 'Phrases', 'Semantics', 'Context']
        name = layer_names[layer_idx] if layer_idx < len(layer_names) else f'Layer {layer_idx}'
        ax.set_title(f'{name} Layer {layer_idx}\n{n_total} neurons, {n_frozen} frozen ({frozen_pct:.0f}%)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    plt.suptitle(f'Cascaded Crystal - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{run_dir}/cascade_epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def main():
    print("=" * 70)
    print("CASCADED CRYSTAL - Breaking Through the Loss Plateau!")
    print("=" * 70)
    print(f"Architecture: {NUM_LAYERS} layers, {NEURONS_PER_LAYER} neurons each")
    print(f"Layer names: Syntax -> Phrases -> Semantics -> Context")
    print("Early layers freeze aggressively, late layers stay flexible")
    print("=" * 70)

    # Load data
    text = load_shakespeare()
    print(f"Loaded {len(text):,} characters of Shakespeare")

    # BPE tokenization
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")

    # Create model
    model = CascadedCrystalLM(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        neurons_per_layer=NEURONS_PER_LAYER,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/cascade_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nTraining on {DEVICE}")
    print(f"Output: {run_dir}/")
    print("=" * 70)

    best_loss = float('inf')
    loss_history = []
    last_optimizer_update = 0

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

            # Track gradients
            model.accumulate_all_gradients()

            # Zero frozen gradients
            model.zero_all_frozen_grads()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(batches)
        loss_history.append(avg_loss)

        # Growth and freezing every interval
        if epoch % GROWTH_INTERVAL == 0:
            grown, frozen = model.grow_and_freeze(epoch, EPOCHS, DEVICE)

            # Update optimizer if neurons were added
            if grown > 0:
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
                last_optimizer_update = epoch

        # Generate sample
        sample = generate(model, tokenizer, "ROMEO:", max_tokens=80, temperature=0.8)

        # Stats
        stats = model.layer_stats()
        total_n = model.total_neurons()
        total_f = model.total_frozen()
        frozen_pct = 100 * total_f / total_n if total_n > 0 else 0

        # Per-layer summary
        layer_summary = " | ".join([f"L{s['layer']}:{s['neurons']}/{s['frozen']}" for s in stats])

        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Total: {total_n}N/{total_f}F ({frozen_pct:.0f}%) | {layer_summary}")
        print(f"         -> {sample[:80]}...")

        # Check if we're beating the plateau!
        if avg_loss < 5.19 and epoch > 100:
            print("         *** BREAKING THROUGH 5.19 PLATEAU! ***")

        # Visualize
        if epoch % 20 == 0 or epoch == 1:
            visualize_cascade(model, epoch, run_dir, sample)

        # Save history
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'total_neurons': total_n,
            'total_frozen': total_f,
            'layer_stats': stats,
            'sample': sample[:200]
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
                    'neurons_per_layer': NEURONS_PER_LAYER,
                    'num_layers': NUM_LAYERS
                },
                'epoch': epoch,
                'loss': avg_loss
            }, f'{run_dir}/best_model.pt')

    # Final
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Best loss: {best_loss:.4f}")
    print("\nPer-layer final stats:")
    for s in model.layer_stats():
        print(f"  Layer {s['layer']}: {s['neurons']} neurons, {s['frozen']} frozen ({s['frozen_pct']:.0f}%)")
    print("=" * 70)

    # Final generation
    sample = generate(model, tokenizer, "ROMEO:", max_tokens=200, temperature=0.8)
    print("\nFinal generation:")
    print(sample)


if __name__ == "__main__":
    main()
