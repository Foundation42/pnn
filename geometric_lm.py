"""
Geometric Language Model

Neurons live in semantic space, not physical 3D.
Each neuron represents a region of meaning and predicts next tokens.
Sparse activation - only nearby neurons fire.
Dynamic growth - split when hitting unknown territory.

Start simple: character-level Shakespeare.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeometricLM(nn.Module):
    """
    A language model where neurons tessellate semantic space.

    Instead of positions in 3D, neurons live in embedding space.
    Context embeddings find nearby neurons, which vote on next token.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 64,
                 context_length: int = 32,
                 max_neurons: int = 100,
                 n_seeds: int = 10,
                 bandwidth: float = 1.0):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.max_neurons = max_neurons
        self.bandwidth = bandwidth

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Context compression: sequence of embeddings -> single vector
        # Simple: average of last N tokens weighted by position
        self.register_buffer('pos_weights',
            torch.exp(torch.linspace(-2, 0, context_length)))  # Recent tokens matter more

        # Neurons in semantic space
        # Position = where in embedding space this neuron lives
        self.register_buffer('positions', torch.randn(max_neurons, embed_dim) * 0.5)

        # Each neuron votes on next token
        self.output_weights = nn.Parameter(torch.randn(max_neurons, vocab_size) * 0.01)

        # Alive mask
        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.alive_mask[:n_seeds] = True

        # Load tracking for splitting
        self.register_buffer('activation_load', torch.zeros(max_neurons))

        # Track prediction errors per neuron (for gap detection)
        self.register_buffer('error_accum', torch.zeros(max_neurons))

        self.total_splits = 0
        self.split_threshold = 0.7

    @property
    def n_alive(self):
        return self.alive_mask.sum().item()

    def embed_context(self, token_ids):
        """
        Convert sequence of token IDs to a single context vector.

        token_ids: (batch, seq_len)
        returns: (batch, embed_dim)
        """
        # Get embeddings: (batch, seq_len, embed_dim)
        embeds = self.token_embed(token_ids)

        # Weight by position (recent tokens matter more)
        seq_len = token_ids.shape[1]
        weights = self.pos_weights[-seq_len:].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Weighted average
        context = (embeds * weights).sum(dim=1)  # (batch, embed_dim)

        return context

    def forward(self, token_ids):
        """
        Predict next token distribution.

        token_ids: (batch, seq_len) - context window
        returns: (batch, vocab_size) - logits for next token
        """
        # Get context vector
        context = self.embed_context(token_ids)  # (batch, embed_dim)

        # Get alive neuron positions
        alive_idx = torch.where(self.alive_mask)[0]
        positions = self.positions[alive_idx]  # (n_alive, embed_dim)

        # Compute distances from context to each neuron
        # context: (batch, embed_dim), positions: (n_alive, embed_dim)
        # distances: (batch, n_alive)
        diff = context.unsqueeze(1) - positions.unsqueeze(0)  # (batch, n_alive, embed_dim)
        distances_sq = (diff ** 2).sum(dim=-1)  # (batch, n_alive)

        # RBF activation: closer neurons fire more
        activations = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))  # (batch, n_alive)

        # Track load (for splitting decisions)
        with torch.no_grad():
            load = activations.mean(dim=0)  # Average activation per neuron
            self.activation_load[alive_idx] = (
                0.9 * self.activation_load[alive_idx] + 0.1 * load
            )

        # Weighted vote for next token
        # activations: (batch, n_alive), output_weights: (n_alive, vocab_size)
        logits = activations @ self.output_weights[alive_idx]  # (batch, vocab_size)

        return logits

    def split_neuron(self, parent_idx):
        """Split overloaded neuron."""
        empty_slots = torch.where(~self.alive_mask)[0]
        if len(empty_slots) == 0:
            return False

        child_idx = empty_slots[0].item()
        parent_idx = parent_idx.item() if torch.is_tensor(parent_idx) else parent_idx

        with torch.no_grad():
            # Position: offset in random direction in embedding space
            offset = torch.randn(self.embed_dim, device=self.positions.device) * 0.3
            self.positions[child_idx] = self.positions[parent_idx] + offset

            # Clone weights with small noise
            self.output_weights.data[child_idx] = (
                self.output_weights.data[parent_idx] *
                (1 + torch.randn_like(self.output_weights.data[parent_idx]) * 0.01)
            )

            self.alive_mask[child_idx] = True
            self.activation_load[parent_idx] = 0
            self.activation_load[child_idx] = 0

        self.total_splits += 1
        return True

    def maybe_split(self):
        """Check for overloaded neurons and split."""
        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) == 0 or self.n_alive >= self.max_neurons:
            return 0

        loads = self.activation_load[alive_idx]
        max_load_local = loads.argmax()
        max_load_idx = alive_idx[max_load_local]
        max_load = loads[max_load_local]

        if max_load > self.split_threshold:
            if self.split_neuron(max_load_idx):
                print(f"  Split! Neuron {max_load_idx.item()} → {self.n_alive} neurons total")
                return 1
        return 0

    def generate(self, start_tokens, max_new_tokens=100, temperature=0.8):
        """Generate text autoregressively."""
        self.eval()
        tokens = start_tokens.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get context (last context_length tokens)
                context = tokens[:, -self.context_length:]

                # Predict next token
                logits = self(context)

                # Sample with temperature
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                tokens = torch.cat([tokens, next_token], dim=1)

        self.train()
        return tokens


def load_shakespeare(path='data/shakespeare.txt', max_chars=100000):
    """Load and preprocess Shakespeare text."""
    import os

    # Download if needed
    if not os.path.exists(path):
        os.makedirs('data', exist_ok=True)
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading Shakespeare to {path}...")
        urllib.request.urlretrieve(url, path)

    with open(path, 'r') as f:
        text = f.read()[:max_chars]

    # Build vocab
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}

    # Encode
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    return data, char_to_idx, idx_to_char


def train_geometric_lm(epochs=20, context_length=32, batch_size=64):
    """Train the geometric language model."""

    print("=" * 60)
    print("  GEOMETRIC LANGUAGE MODEL")
    print("  Neurons tessellate semantic space")
    print("  Character-level Shakespeare")
    print("=" * 60)

    # Load data
    data, char_to_idx, idx_to_char = load_shakespeare()
    vocab_size = len(char_to_idx)

    print(f"\nVocab size: {vocab_size} characters")
    print(f"Data length: {len(data)} chars")
    print(f"Context length: {context_length}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    model = GeometricLM(
        vocab_size=vocab_size,
        embed_dim=64,
        context_length=context_length,
        max_neurons=50,
        n_seeds=10,
        bandwidth=1.0
    ).to(device)

    data = data.to(device)

    print(f"Starting with {model.n_alive} seed neurons")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    # Training loop
    n_batches_per_epoch = len(data) // (batch_size * context_length)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Random starting positions for each batch
        for batch_idx in range(n_batches_per_epoch):
            # Get random starting positions
            starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))

            # Build input and target
            x = torch.stack([data[s:s+context_length] for s in starts])
            y = torch.stack([data[s+context_length] for s in starts])

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Maybe split
        n_splits = 0
        if (epoch + 1) % 3 == 0:
            n_splits = model.maybe_split()

        # Generate sample
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Start with "ROMEO:"
            start_text = "ROMEO:"
            start_tokens = torch.tensor([[char_to_idx.get(c, 0) for c in start_text]], device=device)

            generated = model.generate(start_tokens, max_new_tokens=100, temperature=0.8)
            gen_text = ''.join([idx_to_char[i.item()] for i in generated[0]])

            print(f"\nEpoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Neurons: {model.n_alive} | Splits: {n_splits}")
            print(f"Sample: {gen_text[:80]}...")
            print()
        else:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Neurons: {model.n_alive} | Splits: {n_splits}")

    print()
    print("=" * 60)
    print(f"  Final: {model.n_alive} neurons")
    print(f"  Final loss: {avg_loss:.4f}")
    print("=" * 60)

    # Final generation
    print("\nFinal generation samples:")
    for prompt in ["ROMEO:", "To be", "What "]:
        start_tokens = torch.tensor([[char_to_idx.get(c, 0) for c in prompt]], device=device)
        generated = model.generate(start_tokens, max_new_tokens=150, temperature=0.7)
        gen_text = ''.join([idx_to_char[i.item()] for i in generated[0]])
        print(f"\n{gen_text}")

    return model


if __name__ == "__main__":
    train_geometric_lm(epochs=30, context_length=32, batch_size=128)
