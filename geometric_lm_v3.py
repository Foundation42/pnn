"""
Geometric LM v3 - Push for coherent text

Aggressive version:
1. Start with 20 seeds, grow to 100
2. Each neuron has a small MLP (more capacity per neuron)
3. Longer context, more training
4. Faster growth early on
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GeometricLMv3(nn.Module):
    """
    Each neuron is a small expert with its own MLP.
    Neurons tessellate semantic space and specialize.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_dim: int = 64,
                 context_length: int = 64,
                 max_neurons: int = 100,
                 n_seeds: int = 20,
                 bandwidth: float = 0.3):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_length = context_length
        self.max_neurons = max_neurons
        self.bandwidth = bandwidth

        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)

        # Context compression: small transformer-style attention
        self.context_proj = nn.Linear(embed_dim, embed_dim)
        self.context_query = nn.Parameter(torch.randn(embed_dim) * 0.1)

        # Learnable neuron positions
        self.positions = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.3)

        # Each neuron has its own small MLP: embed_dim -> hidden -> vocab
        # This gives each neuron more expressive power
        self.neuron_hidden = nn.Parameter(torch.randn(max_neurons, embed_dim, hidden_dim) * 0.05)
        self.neuron_output = nn.Parameter(torch.randn(max_neurons, hidden_dim, vocab_size) * 0.05)
        self.neuron_bias = nn.Parameter(torch.zeros(max_neurons, vocab_size))

        # Alive mask
        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.alive_mask[:n_seeds] = True

        # Load tracking
        self.register_buffer('activation_load', torch.zeros(max_neurons))

        self.total_splits = 0
        self.split_threshold = 0.5

    @property
    def n_alive(self):
        return self.alive_mask.sum().item()

    def embed_context(self, token_ids):
        """Attention-pooled context embedding."""
        batch_size, seq_len = token_ids.shape

        positions = torch.arange(seq_len, device=token_ids.device)
        embeds = self.token_embed(token_ids) + self.pos_embed(positions)

        # Project and attend
        projected = self.context_proj(embeds)
        attn_scores = (projected @ self.context_query) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(attn_scores, dim=1)

        context = (embeds * attn_weights.unsqueeze(-1)).sum(dim=1)
        return context

    def forward(self, token_ids):
        """Forward pass with neuron MLPs."""
        batch_size = token_ids.shape[0]
        context = self.embed_context(token_ids)  # (batch, embed_dim)

        alive_idx = torch.where(self.alive_mask)[0]
        n_alive = len(alive_idx)

        # Distances to neurons
        positions = self.positions[alive_idx]
        diff = context.unsqueeze(1) - positions.unsqueeze(0)
        distances_sq = (diff ** 2).sum(dim=-1)

        # RBF activation (sharper with small bandwidth)
        activations = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))

        # Track load
        with torch.no_grad():
            load = activations.mean(dim=0)
            self.activation_load[alive_idx] = 0.9 * self.activation_load[alive_idx] + 0.1 * load

        # Each neuron computes its own prediction via MLP
        # context: (batch, embed_dim)
        # neuron_hidden[alive]: (n_alive, embed_dim, hidden_dim)
        # neuron_output[alive]: (n_alive, hidden_dim, vocab_size)

        hidden_weights = self.neuron_hidden[alive_idx]  # (n_alive, embed_dim, hidden)
        output_weights = self.neuron_output[alive_idx]  # (n_alive, hidden, vocab)
        biases = self.neuron_bias[alive_idx]  # (n_alive, vocab)

        # Compute each neuron's hidden: (batch, n_alive, hidden)
        # context: (batch, 1, embed_dim) @ hidden_weights: (n_alive, embed_dim, hidden)
        hidden = torch.einsum('be,neh->bnh', context, hidden_weights)
        hidden = F.relu(hidden)

        # Compute each neuron's output: (batch, n_alive, vocab)
        neuron_logits = torch.einsum('bnh,nhv->bnv', hidden, output_weights) + biases

        # Weight by activation and sum
        # activations: (batch, n_alive, 1), neuron_logits: (batch, n_alive, vocab)
        weighted_logits = activations.unsqueeze(-1) * neuron_logits
        logits = weighted_logits.sum(dim=1)  # (batch, vocab)

        return logits

    def split_neuron(self, parent_idx):
        """Split with inherited MLP weights."""
        empty_slots = torch.where(~self.alive_mask)[0]
        if len(empty_slots) == 0:
            return False

        child_idx = empty_slots[0].item()
        parent_idx = parent_idx.item() if torch.is_tensor(parent_idx) else parent_idx

        with torch.no_grad():
            # Position offset
            offset = torch.randn(self.embed_dim, device=self.positions.device) * 0.15
            self.positions.data[child_idx] = self.positions.data[parent_idx] + offset

            # Clone MLP weights with small noise
            noise = 0.01
            self.neuron_hidden.data[child_idx] = self.neuron_hidden.data[parent_idx] * (
                1 + torch.randn_like(self.neuron_hidden.data[parent_idx]) * noise)
            self.neuron_output.data[child_idx] = self.neuron_output.data[parent_idx] * (
                1 + torch.randn_like(self.neuron_output.data[parent_idx]) * noise)
            self.neuron_bias.data[child_idx] = self.neuron_bias.data[parent_idx].clone()

            self.alive_mask[child_idx] = True
            self.activation_load[parent_idx] *= 0.5
            self.activation_load[child_idx] = 0

        self.total_splits += 1
        return True

    def maybe_split(self, force=False):
        """Split overloaded neurons."""
        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) == 0 or self.n_alive >= self.max_neurons:
            return 0

        loads = self.activation_load[alive_idx]
        max_idx = alive_idx[loads.argmax()]
        max_load = loads.max().item()

        if max_load > self.split_threshold or force:
            if self.split_neuron(max_idx):
                return 1
        return 0

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens=100, temperature=0.7, top_k=None):
        """Generate with optional top-k sampling."""
        self.eval()
        tokens = start_tokens.clone()

        for _ in range(max_new_tokens):
            context = tokens[:, -self.context_length:]
            logits = self(context)

            # Temperature
            logits = logits / temperature

            # Optional top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        self.train()
        return tokens


def load_shakespeare(path='data/shakespeare.txt'):
    """Load full Shakespeare."""
    import os

    if not os.path.exists(path):
        os.makedirs('data', exist_ok=True)
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print("Downloading Shakespeare...")
        urllib.request.urlretrieve(url, path)

    with open(path, 'r') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    return data, char_to_idx, idx_to_char, text


def train_v3(epochs=100, batch_size=128):
    """Train aggressive v3."""

    print("=" * 60)
    print("  GEOMETRIC LM v3 - PUSH FOR COHERENCE")
    print("  20 seeds → 100 neurons, each with MLP")
    print("=" * 60)

    data, char_to_idx, idx_to_char, raw_text = load_shakespeare()
    vocab_size = len(char_to_idx)
    context_length = 64

    print(f"\nVocab: {vocab_size} | Data: {len(data):,} chars")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = GeometricLMv3(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=64,
        context_length=context_length,
        max_neurons=100,
        n_seeds=20,
        bandwidth=0.3
    ).to(device)

    data = data.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Starting neurons: {model.n_alive}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n_batches = len(data) // (batch_size * context_length)

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for _ in range(n_batches):
            starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
            x = torch.stack([data[s:s+context_length] for s in starts])
            y = torch.stack([data[s+context_length] for s in starts])

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Aggressive growth early
        n_splits = 0
        if epoch < 50 and (epoch + 1) % 2 == 0:
            force = model.n_alive < 50
            n_splits = model.maybe_split(force=force)
            if n_splits:
                print(f"  +1 → {model.n_alive} neurons")

        # Sample periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            prompt = "ROMEO:"
            tokens = torch.tensor([[char_to_idx.get(c, 0) for c in prompt]], device=device)
            gen = model.generate(tokens, max_new_tokens=100, temperature=0.6, top_k=10)
            text = ''.join([idx_to_char[i.item()] for i in gen[0]])

            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.3f} | Best: {best_loss:.3f} | Neurons: {model.n_alive}")
            print(f"{'='*60}")
            print(text)
            print()
        else:
            status = f"Epoch {epoch+1:3d} | Loss: {avg_loss:.3f} | Neurons: {model.n_alive}"
            if n_splits:
                status += f" | +{n_splits}"
            print(status)

    # Final samples
    print("\n" + "=" * 60)
    print("FINAL GENERATIONS")
    print("=" * 60)

    prompts = [
        "ROMEO:\n",
        "To be, or not to be",
        "What light through yonder",
        "JULIET:\nO Romeo, Romeo",
        "Now is the winter of"
    ]

    for prompt in prompts:
        if all(c in char_to_idx for c in prompt):
            tokens = torch.tensor([[char_to_idx[c] for c in prompt]], device=device)
            gen = model.generate(tokens, max_new_tokens=200, temperature=0.5, top_k=8)
            text = ''.join([idx_to_char[i.item()] for i in gen[0]])
            print(f"\n{'-'*40}")
            print(text)

    return model


if __name__ == "__main__":
    train_v3(epochs=100, batch_size=128)
