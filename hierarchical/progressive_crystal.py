"""
Progressive Crystal - Layer-by-layer crystallization

Key insight: Train one layer until it crystallizes (75-80% frozen),
THEN add the next layer. Each layer learns the residual that
previous layers couldn't capture.

Layer 0: Learns easy patterns (syntax, common tokens)
Layer 1: Learns what Layer 0 missed (phrases, local context)
Layer 2: Learns what Layer 0+1 missed (longer dependencies)
...and so on

This avoids gradient dilution and ensures each layer is useful.
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
MAX_SEQUENCES = 10000

# Progressive config
INITIAL_NEURONS = 64
MAX_NEURONS_PER_LAYER = 256
CRYSTALLIZATION_THRESHOLD = 0.50  # Add new layer when this % frozen (was 0.75)
MAX_LAYERS = 6
GROWTH_INTERVAL = 5  # More frequent growth/freeze cycles (was 10)
MIN_EPOCHS_PER_LAYER = 40  # Minimum epochs before crystallizing (was 100)
MAX_TOTAL_EPOCHS = 1200  # Total training budget


class GeometricAttention(nn.Module):
    """Attention through geometric proximity"""

    def __init__(self, embed_dim, num_neurons, layer_idx=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons
        self.layer_idx = layer_idx

        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.5)
        self.values = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.02)
        self.temperature = nn.Parameter(torch.ones(num_neurons) * 0.5)

        self.register_buffer('gradient_acc', torch.zeros(num_neurons))
        self.register_buffer('gradient_history', torch.zeros(num_neurons, 10))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('frozen', torch.zeros(num_neurons, dtype=torch.bool))
        self.register_buffer('birth_epoch', torch.zeros(num_neurons))
        self.current_epoch = 0

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        dists = torch.cdist(x_flat, self.positions)
        weights = torch.exp(-dists / (self.temperature.abs() + 0.1))
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
        if num_new <= 0 or self.num_neurons >= MAX_NEURONS_PER_LAYER:
            return 0

        num_new = min(num_new, MAX_NEURONS_PER_LAYER - self.num_neurons)

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
            return n_new
        return 0

    def freeze_cold_neurons(self, min_age=15):
        """Freeze neurons that have stabilized - AGGRESSIVE version"""
        neuron_age = self.current_epoch - self.birth_epoch
        too_young = neuron_age < min_age

        if self.history_ptr < 5:  # Reduced from 10
            return 0

        grad_variance = self.gradient_history.var(dim=1)
        grad_mean = self.gradient_history.mean(dim=1)

        candidates = (~self.frozen) & (~too_young)
        if candidates.sum() < 2:
            return 0

        # Freeze neurons with low and stable gradients
        coldness = torch.zeros(self.num_neurons, device=self.positions.device)
        coldness[candidates] = -grad_mean[candidates] - grad_variance[candidates] * 5  # Reduced variance penalty
        coldness[~candidates] = float('-inf')

        # Freeze up to 25% of candidates per call (was 10%)
        max_freeze = max(2, int(candidates.sum().item() * 0.25))
        _, freeze_idx = torch.topk(coldness, min(max_freeze, candidates.sum().item()))

        # Freeze if below 75th percentile activity (was median)
        threshold = torch.quantile(grad_mean[candidates].float(), 0.75)
        actually_cold = grad_mean[freeze_idx] < threshold

        to_freeze = freeze_idx[actually_cold]
        self.frozen[to_freeze] = True

        return len(to_freeze)

    def frozen_ratio(self):
        return self.frozen.sum().item() / self.num_neurons if self.num_neurons > 0 else 0

    def zero_frozen_grads(self):
        if self.positions.grad is not None:
            self.positions.grad[self.frozen] = 0
            self.values.grad[self.frozen] = 0
            self.temperature.grad[self.frozen] = 0


class CrystalLayer(nn.Module):
    """Single crystal layer"""

    def __init__(self, embed_dim, num_neurons, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_crystallized = False  # Fully frozen flag

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = GeometricAttention(embed_dim, num_neurons, layer_idx)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x):
        h = x + self.attention(self.norm1(x))
        h = h + self.ffn(self.norm2(h))
        return h

    def freeze_all(self):
        """Completely freeze this layer"""
        self.is_crystallized = True
        for param in self.parameters():
            param.requires_grad = False
        self.attention.frozen.fill_(True)


class ProgressiveCrystalLM(nn.Module):
    """Language model that grows layers progressively"""

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Embeddings (always trainable)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(CONTEXT_LEN, embed_dim)

        # Start with one layer
        self.layers = nn.ModuleList([
            CrystalLayer(embed_dim, INITIAL_NEURONS, layer_idx=0)
        ])
        self.active_layer_idx = 0  # Which layer is currently training

        # Output
        self.norm_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        pos = torch.arange(T, device=x.device)
        h = self.token_embed(x) + self.pos_embed(pos)

        # Process through all layers
        for layer in self.layers:
            h = layer(h)

        h = self.norm_out(h)
        logits = self.head(h)

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss
        return logits

    def add_layer(self, device):
        """Add a new crystal layer"""
        new_idx = len(self.layers)
        if new_idx >= MAX_LAYERS:
            return False

        new_layer = CrystalLayer(self.embed_dim, INITIAL_NEURONS, layer_idx=new_idx)
        new_layer = new_layer.to(device)
        self.layers.append(new_layer)
        self.active_layer_idx = new_idx
        return True

    def get_active_layer(self):
        """Get the currently training layer"""
        if self.active_layer_idx < len(self.layers):
            return self.layers[self.active_layer_idx]
        return None

    def crystallize_active_layer(self):
        """Fully freeze the active layer and prep for next"""
        active = self.get_active_layer()
        if active:
            active.freeze_all()
            print(f"  >>> Layer {self.active_layer_idx} CRYSTALLIZED ({active.attention.num_neurons} neurons)")

    def total_neurons(self):
        return sum(layer.attention.num_neurons for layer in self.layers)

    def total_frozen(self):
        return sum(layer.attention.frozen.sum().item() for layer in self.layers)

    def status_string(self):
        """Get status of all layers"""
        parts = []
        for i, layer in enumerate(self.layers):
            att = layer.attention
            n = att.num_neurons
            f = att.frozen.sum().item()
            pct = 100 * f / n if n > 0 else 0
            marker = "*" if i == self.active_layer_idx else ""
            marker = "C" if layer.is_crystallized else marker
            parts.append(f"L{i}{marker}:{n}/{f}({pct:.0f}%)")
        return " | ".join(parts)


def load_shakespeare():
    """Load TinyShakespeare"""
    cache_paths = ["data/tinyshakespeare.txt", "../data/tinyshakespeare.txt"]

    for path in cache_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()

    # Download if not found
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


def train_epoch(model, batches, optimizer):
    """Train for one epoch, only updating active layer + embeddings + head"""
    model.train()
    total_loss = 0

    active_layer = model.get_active_layer()

    for batch in batches:
        batch = batch.to(DEVICE)
        x, y = batch[:, :-1], batch[:, 1:]

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()

        # Accumulate gradients for active layer
        if active_layer:
            active_layer.attention.accumulate_gradients()
            active_layer.attention.zero_frozen_grads()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(batches)


def main():
    print("=" * 70)
    print("PROGRESSIVE CRYSTAL - Layer-by-Layer Crystallization")
    print("=" * 70)
    print(f"Strategy: Train layer until {CRYSTALLIZATION_THRESHOLD*100:.0f}% frozen, then add next layer")
    print(f"Max layers: {MAX_LAYERS}, Initial neurons: {INITIAL_NEURONS}")
    print("=" * 70)

    # Load data
    text = load_shakespeare()
    print(f"Loaded {len(text):,} characters")

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")

    # Create model
    model = ProgressiveCrystalLM(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM
    ).to(DEVICE)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/progressive_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nTraining on {DEVICE}")
    print(f"Output: {run_dir}/")
    print("=" * 70)

    # Training state
    best_loss = float('inf')
    epoch = 0
    layer_start_epoch = 0  # When current layer started training

    # Get trainable params for optimizer
    def get_optimizer():
        # Only optimize: embeddings, active layer, output head
        params = list(model.token_embed.parameters()) + \
                 list(model.pos_embed.parameters()) + \
                 list(model.norm_out.parameters()) + \
                 list(model.head.parameters())

        active = model.get_active_layer()
        if active and not active.is_crystallized:
            params += list(active.parameters())

        return torch.optim.AdamW(params, lr=3e-4)

    optimizer = get_optimizer()

    while epoch < MAX_TOTAL_EPOCHS:
        epoch += 1
        batches = create_batches(tokens, BATCH_SIZE, CONTEXT_LEN)

        avg_loss = train_epoch(model, batches, optimizer)

        active_layer = model.get_active_layer()

        # Growth and freezing for active layer
        if epoch % GROWTH_INTERVAL == 0 and active_layer and not active_layer.is_crystallized:
            att = active_layer.attention
            att.current_epoch = epoch

            # Freeze cold neurons aggressively
            att.freeze_cold_neurons(min_age=15)

            # Grow if not at max - faster growth early
            if att.num_neurons < MAX_NEURONS_PER_LAYER:
                # Grow 12 neurons per cycle (was 8)
                grow_count = min(12, MAX_NEURONS_PER_LAYER - att.num_neurons)
                grown = att.grow_neurons(grow_count, DEVICE)
                if grown > 0:
                    optimizer = get_optimizer()

        # Check for layer crystallization
        frozen_ratio = active_layer.attention.frozen_ratio() if active_layer else 1.0

        # Generate sample periodically
        sample = ""
        if epoch % 20 == 0:
            sample = generate(model, tokenizer, "ROMEO:", max_tokens=60, temperature=0.8)

        # Status
        status = model.status_string()
        print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | {status}")
        if sample:
            print(f"         -> {sample[:70]}...")

        # Save history
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'num_layers': len(model.layers),
            'active_layer': model.active_layer_idx,
            'total_neurons': model.total_neurons(),
            'total_frozen': model.total_frozen(),
            'active_frozen_ratio': frozen_ratio,
            'status': status
        }
        with open(f'{run_dir}/history.jsonl', 'a') as f:
            f.write(json.dumps(epoch_data) + '\n')

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'num_layers': len(model.layers)
            }, f'{run_dir}/best_model.pt')

        # Check if we should crystallize and add new layer
        epochs_on_layer = epoch - layer_start_epoch
        if frozen_ratio >= CRYSTALLIZATION_THRESHOLD and epochs_on_layer >= MIN_EPOCHS_PER_LAYER:
            print(f"\n{'='*70}")
            print(f"Layer {model.active_layer_idx} reached {frozen_ratio*100:.0f}% frozen!")

            # Crystallize current layer
            model.crystallize_active_layer()

            # Add new layer if not at max
            if len(model.layers) < MAX_LAYERS:
                if model.add_layer(DEVICE):
                    print(f"Added Layer {model.active_layer_idx} with {INITIAL_NEURONS} neurons")
                    layer_start_epoch = epoch
                    optimizer = get_optimizer()
                else:
                    print("Max layers reached!")
                    break
            else:
                print("Max layers reached!")
                break

            print(f"{'='*70}\n")

    # Final
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Final: {len(model.layers)} layers, {model.total_neurons()} neurons, {model.total_frozen()} frozen")
    print(f"Best loss: {best_loss:.4f}")
    print("=" * 70)

    sample = generate(model, tokenizer, "ROMEO:", max_tokens=200, temperature=0.8)
    print("\nFinal generation:")
    print(sample)


if __name__ == "__main__":
    main()
