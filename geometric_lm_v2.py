"""
Geometric Language Model v2

Improvements:
1. LEARNABLE positions - neurons can move in semantic space
2. More aggressive growth
3. Better context encoding (simple attention)
4. Lower bandwidth for sharper activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeometricLMv2(nn.Module):
    """
    Geometric LM with learnable neuron positions.

    Neurons don't just tessellate - they MIGRATE to where they're needed.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 64,
                 context_length: int = 32,
                 max_neurons: int = 100,
                 n_seeds: int = 10,
                 bandwidth: float = 0.5):  # Tighter bandwidth
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.max_neurons = max_neurons
        self.bandwidth = bandwidth

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding for context
        self.pos_embed = nn.Embedding(context_length, embed_dim)

        # Context compression: attention-style pooling
        self.context_query = nn.Parameter(torch.randn(embed_dim) * 0.1)

        # LEARNABLE neuron positions in semantic space
        self.positions = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.5)

        # Each neuron votes on next token
        self.output_weights = nn.Parameter(torch.randn(max_neurons, vocab_size) * 0.02)

        # Alive mask
        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.alive_mask[:n_seeds] = True

        # Load tracking
        self.register_buffer('activation_load', torch.zeros(max_neurons))

        self.total_splits = 0
        self.split_threshold = 0.6

    @property
    def n_alive(self):
        return self.alive_mask.sum().item()

    def embed_context(self, token_ids):
        """
        Convert sequence of token IDs to a single context vector.
        Uses simple attention-style pooling.
        """
        batch_size, seq_len = token_ids.shape

        # Token + position embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        embeds = self.token_embed(token_ids) + self.pos_embed(positions).unsqueeze(0)

        # Attention weights: query dot product with each position
        # (batch, seq_len, embed_dim) @ (embed_dim,) -> (batch, seq_len)
        attn_scores = (embeds @ self.context_query) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum
        context = (embeds * attn_weights.unsqueeze(-1)).sum(dim=1)

        return context

    def forward(self, token_ids):
        """Predict next token distribution."""
        context = self.embed_context(token_ids)  # (batch, embed_dim)

        alive_idx = torch.where(self.alive_mask)[0]
        positions = self.positions[alive_idx]  # (n_alive, embed_dim)

        # Distances in semantic space
        diff = context.unsqueeze(1) - positions.unsqueeze(0)
        distances_sq = (diff ** 2).sum(dim=-1)

        # RBF activation
        activations = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))

        # Normalize activations (so they sum to 1 - like attention)
        activations = activations / (activations.sum(dim=1, keepdim=True) + 1e-8)

        # Track load
        with torch.no_grad():
            load = activations.mean(dim=0)
            self.activation_load[alive_idx] = (
                0.9 * self.activation_load[alive_idx] + 0.1 * load
            )

        # Vote for next token
        logits = activations @ self.output_weights[alive_idx]

        return logits

    def split_neuron(self, parent_idx):
        """Split overloaded neuron."""
        empty_slots = torch.where(~self.alive_mask)[0]
        if len(empty_slots) == 0:
            return False

        child_idx = empty_slots[0].item()
        parent_idx = parent_idx.item() if torch.is_tensor(parent_idx) else parent_idx

        with torch.no_grad():
            # Position: offset in random direction
            offset = torch.randn(self.embed_dim, device=self.positions.device) * 0.2
            self.positions.data[child_idx] = self.positions.data[parent_idx] + offset

            # Clone output weights
            self.output_weights.data[child_idx] = (
                self.output_weights.data[parent_idx] *
                (1 + torch.randn_like(self.output_weights.data[parent_idx]) * 0.01)
            )

            self.alive_mask[child_idx] = True
            self.activation_load[parent_idx] = 0
            self.activation_load[child_idx] = 0

        self.total_splits += 1
        return True

    def maybe_split(self, force=False):
        """Check for overloaded neurons."""
        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) == 0 or self.n_alive >= self.max_neurons:
            return 0

        loads = self.activation_load[alive_idx]
        max_load_local = loads.argmax()
        max_load_idx = alive_idx[max_load_local]
        max_load = loads[max_load_local]

        if max_load > self.split_threshold or force:
            if self.split_neuron(max_load_idx):
                return 1
        return 0

    def generate(self, start_tokens, max_new_tokens=100, temperature=0.8):
        """Generate text."""
        self.eval()
        tokens = start_tokens.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = tokens[:, -self.context_length:]
                logits = self(context)
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)

        self.train()
        return tokens


def load_shakespeare(path='data/shakespeare.txt', max_chars=200000):
    """Load Shakespeare text."""
    import os

    if not os.path.exists(path):
        os.makedirs('data', exist_ok=True)
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading Shakespeare...")
        urllib.request.urlretrieve(url, path)

    with open(path, 'r') as f:
        text = f.read()[:max_chars]

    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    return data, char_to_idx, idx_to_char


def train_geometric_lm_v2(epochs=50, context_length=64, batch_size=128):
    """Train the improved geometric LM."""

    print("=" * 60)
    print("  GEOMETRIC LM v2")
    print("  Learnable positions - neurons MIGRATE to meaning")
    print("=" * 60)

    data, char_to_idx, idx_to_char = load_shakespeare()
    vocab_size = len(char_to_idx)

    print(f"\nVocab: {vocab_size} chars | Data: {len(data)} chars")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = GeometricLMv2(
        vocab_size=vocab_size,
        embed_dim=64,
        context_length=context_length,
        max_neurons=100,
        n_seeds=10,
        bandwidth=0.5
    ).to(device)

    data = data.to(device)

    print(f"Starting with {model.n_alive} neurons\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n_batches_per_epoch = len(data) // (batch_size * context_length)
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for _ in range(n_batches_per_epoch):
            starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
            x = torch.stack([data[s:s+context_length] for s in starts])
            y = torch.stack([data[s+context_length] for s in starts])

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            # Gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Keep growing - more capacity!
        if (epoch + 1) % 2 == 0 and model.n_alive < 100:
            model.maybe_split(force=True)  # Always grow until 100

        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1

        # Quiet training - only sample every 50 epochs
        if (epoch + 1) % 50 == 0:
            start_text = "ROMEO:"
            start_tokens = torch.tensor([[char_to_idx.get(c, 0) for c in start_text]], device=device)
            generated = model.generate(start_tokens, max_new_tokens=120, temperature=0.5)
            gen_text = ''.join([idx_to_char[i.item()] for i in generated[0]])

            print(f"\n[Epoch {epoch+1:3d}] Loss: {avg_loss:.3f} (best: {best_loss:.3f} @ {best_epoch}) | Neurons: {model.n_alive}")
            print(f">>> {gen_text}\n")
        elif (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.3f} | Neurons: {model.n_alive}")

    print("\n" + "=" * 60)
    print(f"Final: {model.n_alive} neurons | Loss: {avg_loss:.3f}")
    print("=" * 60)

    # Final samples
    print("\n--- Final Generations ---")
    for prompt in ["ROMEO:", "To be or", "What light", "My lord,"]:
        if all(c in char_to_idx for c in prompt):
            start_tokens = torch.tensor([[char_to_idx[c] for c in prompt]], device=device)
            generated = model.generate(start_tokens, max_new_tokens=120, temperature=0.6)
            gen_text = ''.join([idx_to_char[i.item()] for i in generated[0]])
            print(f"\n{gen_text}")

    return model


if __name__ == "__main__":
    train_geometric_lm_v2(epochs=1000, context_length=64, batch_size=256)
