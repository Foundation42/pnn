"""
Geometric Language Model - Word Level

Word-level is MUCH easier than character-level:
- Fewer tokens to predict
- Each token is meaningful
- Context window covers more semantic content

Let's see if 100 neurons can tessellate word-space!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter
import re


class GeometricWordLM(nn.Module):
    """Word-level geometric language model."""

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 128,
                 context_length: int = 32,
                 max_neurons: int = 200,
                 n_seeds: int = 20,
                 bandwidth: float = 0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.max_neurons = max_neurons
        self.bandwidth = bandwidth

        # Word embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)

        # Context pooling
        self.context_query = nn.Parameter(torch.randn(embed_dim) * 0.1)

        # Learnable neuron positions
        self.positions = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.3)

        # Each neuron votes on next word
        self.output_weights = nn.Parameter(torch.randn(max_neurons, vocab_size) * 0.02)

        # Alive mask
        self.register_buffer('alive_mask', torch.zeros(max_neurons, dtype=torch.bool))
        self.alive_mask[:n_seeds] = True

        # Load tracking
        self.register_buffer('activation_load', torch.zeros(max_neurons))

        self.split_threshold = 0.5

    @property
    def n_alive(self):
        return self.alive_mask.sum().item()

    def embed_context(self, token_ids):
        """Attention-pooled context."""
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        embeds = self.token_embed(token_ids) + self.pos_embed(positions)

        attn_scores = (embeds @ self.context_query) / math.sqrt(self.embed_dim)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = (embeds * attn_weights.unsqueeze(-1)).sum(dim=1)
        return context

    def forward(self, token_ids):
        context = self.embed_context(token_ids)
        alive_idx = torch.where(self.alive_mask)[0]
        positions = self.positions[alive_idx]

        diff = context.unsqueeze(1) - positions.unsqueeze(0)
        distances_sq = (diff ** 2).sum(dim=-1)
        activations = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))

        with torch.no_grad():
            load = activations.mean(dim=0)
            self.activation_load[alive_idx] = 0.9 * self.activation_load[alive_idx] + 0.1 * load

        logits = activations @ self.output_weights[alive_idx]
        return logits

    def split_neuron(self, parent_idx):
        empty_slots = torch.where(~self.alive_mask)[0]
        if len(empty_slots) == 0:
            return False

        child_idx = empty_slots[0].item()
        parent_idx = parent_idx.item() if torch.is_tensor(parent_idx) else parent_idx

        with torch.no_grad():
            offset = torch.randn(self.embed_dim, device=self.positions.device) * 0.2
            self.positions.data[child_idx] = self.positions.data[parent_idx] + offset
            self.output_weights.data[child_idx] = (
                self.output_weights.data[parent_idx] *
                (1 + torch.randn_like(self.output_weights.data[parent_idx]) * 0.01)
            )
            self.alive_mask[child_idx] = True
            self.activation_load[parent_idx] = 0
            self.activation_load[child_idx] = 0

        return True

    def maybe_split(self, force=False):
        alive_idx = torch.where(self.alive_mask)[0]
        if len(alive_idx) == 0 or self.n_alive >= self.max_neurons:
            return 0

        loads = self.activation_load[alive_idx]
        max_idx = alive_idx[loads.argmax()]

        if loads.max() > self.split_threshold or force:
            if self.split_neuron(max_idx):
                return 1
        return 0

    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens=50, temperature=0.7, top_k=50):
        self.eval()
        tokens = start_tokens.clone()

        for _ in range(max_new_tokens):
            context = tokens[:, -self.context_length:]
            logits = self(context)
            logits = logits / temperature

            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        self.train()
        return tokens


def load_shakespeare_words(path='data/shakespeare.txt', min_freq=3):
    """Load Shakespeare and tokenize by words."""
    import os

    if not os.path.exists(path):
        os.makedirs('data', exist_ok=True)
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print("Downloading Shakespeare...")
        urllib.request.urlretrieve(url, path)

    with open(path, 'r') as f:
        text = f.read()

    # Simple word tokenization - split on whitespace and punctuation
    # Keep punctuation as separate tokens
    tokens = re.findall(r"[\w']+|[.,!?;:\n]", text.lower())

    # Build vocab with frequency threshold
    counts = Counter(tokens)
    vocab = ['<unk>', '<pad>'] + [w for w, c in counts.most_common() if c >= min_freq]

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    # Encode
    unk_idx = word_to_idx['<unk>']
    data = torch.tensor([word_to_idx.get(t, unk_idx) for t in tokens], dtype=torch.long)

    return data, word_to_idx, idx_to_word, vocab


def train_word_lm(epochs=500, context_length=32, batch_size=128):
    """Train word-level geometric LM."""

    print("=" * 60)
    print("  GEOMETRIC LM - WORD LEVEL")
    print("  Each neuron covers a region of word-space")
    print("=" * 60)

    data, word_to_idx, idx_to_word, vocab = load_shakespeare_words()
    vocab_size = len(vocab)

    print(f"\nVocab: {vocab_size} words | Data: {len(data):,} tokens")
    print(f"Sample vocab: {vocab[2:20]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = GeometricWordLM(
        vocab_size=vocab_size,
        embed_dim=64,
        context_length=context_length,
        max_neurons=500,
        n_seeds=100,  # LOTS more seeds - independent explorers!
        bandwidth=1.0
    ).to(device)

    data = data.to(device)
    print(f"Starting with {model.n_alive} neurons\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n_batches = len(data) // (batch_size * context_length)
    best_loss = float('inf')
    best_epoch = 0

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
            best_epoch = epoch + 1

        # Grow neurons aggressively
        if (epoch + 1) % 2 == 0 and model.n_alive < 500:
            model.maybe_split(force=True)

        # Sample every 100 epochs
        if (epoch + 1) % 100 == 0:
            # Generate from "romeo :"
            start_words = ['romeo', ':']
            start_ids = [word_to_idx.get(w, 0) for w in start_words]
            start_tokens = torch.tensor([start_ids], device=device)

            generated = model.generate(start_tokens, max_new_tokens=40, temperature=0.7)
            gen_words = [idx_to_word[i.item()] for i in generated[0]]
            gen_text = ' '.join(gen_words)

            print(f"\n[Epoch {epoch+1:3d}] Loss: {avg_loss:.3f} (best: {best_loss:.3f} @ {best_epoch}) | Neurons: {model.n_alive}")
            print(f">>> {gen_text}\n")
        elif (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.3f} | Neurons: {model.n_alive}")

    print("\n" + "=" * 60)
    print(f"Final: {model.n_alive} neurons | Loss: {avg_loss:.3f}")
    print("=" * 60)

    # Final generations
    print("\n--- Final Generations ---")
    prompts = [
        ['romeo', ':'],
        ['to', 'be', 'or', 'not', 'to'],
        ['what', 'light', 'through'],
        ['my', 'lord', ','],
    ]

    for prompt in prompts:
        ids = [word_to_idx.get(w, 0) for w in prompt]
        tokens = torch.tensor([ids], device=device)
        gen = model.generate(tokens, max_new_tokens=30, temperature=0.6)
        words = [idx_to_word[i.item()] for i in gen[0]]
        print(f"\n{' '.join(words)}")

    return model


if __name__ == "__main__":
    train_word_lm(epochs=1000, context_length=32, batch_size=128)
