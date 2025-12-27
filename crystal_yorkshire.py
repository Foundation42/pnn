"""
Crystal Yorkshire - Growing geometric LM on 1865-1900 historical text
Uses GPT-2's BPE tokenizer and graceful freezing schedule
Overnight training run on large historical corpus (Yorkshire + main 1865-1900)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import re
import glob
import tiktoken
from datetime import datetime
import json

# Config - tuned for overnight run on large corpus
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 192  # Larger for richer corpus
CONTEXT_LEN = 128  # Longer context for historical prose
BATCH_SIZE = 24
EPOCHS = 800  # Long overnight run
INITIAL_NEURONS = 96
MAX_NEURONS = 2048  # More capacity for large corpus
GROWTH_INTERVAL = 10


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
        self.register_buffer('gradient_history', torch.zeros(num_neurons, 10))
        self.register_buffer('history_ptr', torch.tensor(0))
        self.register_buffer('frozen', torch.zeros(num_neurons, dtype=torch.bool))
        self.register_buffer('birth_epoch', torch.zeros(num_neurons))
        self.register_buffer('update_count', torch.tensor(0))
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
            self.update_count += 1

    def grow_neurons(self, num_new):
        """Add new neurons by splitting the hottest ones"""
        if num_new <= 0:
            return

        activity = self.gradient_acc.clone()
        activity[self.frozen] = -float('inf')
        _, hot_idx = torch.topk(activity, min(num_new, (~self.frozen).sum()))

        new_positions = []
        new_values = []
        new_temps = []

        for idx in hot_idx:
            offset = torch.randn(self.embed_dim, device=self.positions.device) * 0.1
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
            self.gradient_acc = torch.cat([self.gradient_acc, torch.zeros(n_new, device=DEVICE)])
            self.gradient_history = torch.cat([self.gradient_history, torch.zeros(n_new, 10, device=DEVICE)])
            self.frozen = torch.cat([self.frozen, torch.zeros(n_new, dtype=torch.bool, device=DEVICE)])
            self.birth_epoch = torch.cat([self.birth_epoch, torch.full((n_new,), self.current_epoch, device=DEVICE)])

            self.num_neurons = len(self.positions)

    def freeze_cold_neurons(self, epoch, total_epochs):
        """Graceful freezing schedule"""
        if epoch < total_epochs * 0.2:
            return 0

        progress = epoch / total_epochs
        if progress < 0.5:
            aggression = (progress - 0.2) / 0.3 * 0.3
        elif progress < 0.8:
            aggression = 0.3 + (progress - 0.5) / 0.3 * 0.3
        else:
            aggression = 0.6 + (progress - 0.8) / 0.2 * 0.4

        min_age = 50
        neuron_age = epoch - self.birth_epoch
        too_young = neuron_age < min_age

        if self.history_ptr < 10:
            return 0

        grad_variance = self.gradient_history.var(dim=1)
        grad_mean = self.gradient_history.mean(dim=1)

        candidates = (~self.frozen) & (~too_young)
        if candidates.sum() < 10:
            return 0

        coldness = torch.zeros(self.num_neurons, device=DEVICE)
        coldness[candidates] = -grad_mean[candidates] - grad_variance[candidates] * 10

        max_freeze = int(candidates.sum().item() * aggression * 0.1)
        max_freeze = max(1, min(max_freeze, 20))

        coldness[~candidates] = float('-inf')
        _, freeze_idx = torch.topk(coldness, min(max_freeze, candidates.sum().item()))

        median_activity = grad_mean[candidates].median()
        actually_cold = grad_mean[freeze_idx] < median_activity

        to_freeze = freeze_idx[actually_cold]
        self.frozen[to_freeze] = True

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


def clean_text(text):
    """Basic cleaning for historical text"""
    # Remove Google Books boilerplate
    markers = [
        "This is a digital copy of a book",
        "Google Book Search",
        "public domain",
        "Usage guidelines",
    ]
    for marker in markers:
        idx = text.lower().find(marker.lower())
        if idx != -1 and idx < 5000:  # Only if near start
            # Find end of boilerplate (look for double newline)
            end = text.find("\n\n", idx + len(marker))
            if end != -1 and end < 10000:
                text = text[end:]

    # Remove page numbers (standalone numbers on lines)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def load_yorkshire_corpus():
    """Load and clean the Yorkshire + 1865-1900 corpus"""

    print("Loading historical corpus...")
    all_text = []

    # Load from Yorkshire-specific corpus
    yorkshire_path = "../historyLLM/corpus/1865-1900-yorkshire"
    for subdir in ['internet_archive_yorkshire', 'gutenberg_yorkshire']:
        pattern = os.path.join(yorkshire_path, subdir, '*.txt')
        files = glob.glob(pattern)
        print(f"  Found {len(files)} files in yorkshire/{subdir}")

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if len(text) > 1000:  # Skip tiny files
                        cleaned = clean_text(text)
                        if len(cleaned) > 500:
                            all_text.append(cleaned)
            except Exception as e:
                continue

    # Also load from main 1865-1900 corpus
    main_path = "../historyLLM/corpus/1865-1900"
    for subdir in ['gutenberg', 'internet_archive']:
        pattern = os.path.join(main_path, subdir, '*.txt')
        files = glob.glob(pattern)
        print(f"  Found {len(files)} files in 1865-1900/{subdir}")

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    if len(text) > 1000:  # Skip tiny files
                        cleaned = clean_text(text)
                        if len(cleaned) > 500:
                            all_text.append(cleaned)
            except Exception as e:
                continue

    # Combine all text
    full_text = "\n\n".join(all_text)
    print(f"Total corpus: {len(full_text):,} characters from {len(all_text)} documents")

    return full_text


def create_batches(tokens, batch_size, context_len):
    """Create training batches"""
    sequences = []
    # Use strided sampling for large corpus
    stride = context_len // 2
    for i in range(0, len(tokens) - context_len - 1, stride):
        seq = tokens[i:i + context_len + 1]
        if len(seq) == context_len + 1:
            sequences.append(seq)

    # Limit sequences per epoch for reasonable epoch time
    max_sequences = 10000
    if len(sequences) > max_sequences:
        indices = np.random.choice(len(sequences), max_sequences, replace=False)
        sequences = [sequences[i] for i in indices]

    # Shuffle and batch
    indices = torch.randperm(len(sequences))
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        if len(batch_idx) == batch_size:
            batch = torch.stack([torch.tensor(sequences[j]) for j in batch_idx])
            batches.append(batch)

    return batches


def visualize_crystal(model, epoch, run_dir, sample_text="", pca_model=None):
    """Visualize the crystal structure"""
    positions = model.attention.positions.detach().cpu().numpy()
    frozen = model.attention.frozen.cpu().numpy()
    activity = model.attention.gradient_acc.cpu().numpy()

    if positions.shape[0] > 2:
        if pca_model is None:
            pca_model = PCA(n_components=2)
            pos_2d = pca_model.fit_transform(positions)
        else:
            if positions.shape[1] == pca_model.components_.shape[1]:
                pos_2d = pca_model.transform(positions)
            else:
                pca_model = PCA(n_components=2)
                pos_2d = pca_model.fit_transform(positions)
    else:
        pos_2d = positions[:, :2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = ['blue' if not f else 'gray' for f in frozen]
    sizes = 50 + activity * 500
    sizes = np.clip(sizes, 20, 300)

    ax.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=sizes, alpha=0.6)
    ax.set_title(f'Crystal Structure - Epoch {epoch}\n{model.num_neurons} neurons, {model.num_frozen} frozen ({100*model.num_frozen/model.num_neurons:.0f}%)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # Adaptive axis limits with 20% padding
    x_min, x_max = pos_2d[:, 0].min(), pos_2d[:, 0].max()
    y_min, y_max = pos_2d[:, 1].min(), pos_2d[:, 1].max()
    x_pad = max(0.5, (x_max - x_min) * 0.2)
    y_pad = max(0.5, (y_max - y_min) * 0.2)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax = axes[1]
    ax.text(0.02, 0.5, sample_text[:600], fontsize=8, family='monospace',
            verticalalignment='center', wrap=True, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Generated Sample')

    plt.tight_layout()
    plt.savefig(f'{run_dir}/epoch_{epoch:03d}.png', dpi=100)
    plt.close()

    return pca_model


def generate(model, tokenizer, prompt="In Yorkshire,", max_tokens=150, temperature=0.8):
    """Generate text from prompt"""
    model.eval()
    tokens = tokenizer.encode(prompt)[-CONTEXT_LEN:]

    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([tokens[-CONTEXT_LEN:]], device=DEVICE)
            logits = model(x)[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

    return tokenizer.decode(tokens)


def main():
    print("=" * 70)
    print("Crystal Yorkshire - 1865-1900 Historical Text")
    print("Graceful Freezing Schedule - Overnight Run")
    print("=" * 70)
    print("Phases: GROW (0-20%) -> grow+freeze (20-50%) -> FREEZE (50-80%) -> CRYSTALLIZE (80-100%)")

    # Load data
    text = load_yorkshire_corpus()

    # BPE tokenization
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(text)
    print(f"Tokenized to {len(tokens):,} BPE tokens")

    # Create model
    model = CrystalLM(
        vocab_size=tokenizer.n_vocab,
        embed_dim=EMBED_DIM,
        num_neurons=INITIAL_NEURONS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/crystal_yorkshire_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"\nTraining on {DEVICE}")
    print(f"Initial neurons: {INITIAL_NEURONS}, Max: {MAX_NEURONS}")
    print(f"Epochs: {EPOCHS}, Context: {CONTEXT_LEN}")
    print(f"Output: {run_dir}/")
    print("=" * 70)

    best_loss = float('inf')
    history = {
        'config': {
            'epochs': EPOCHS,
            'embed_dim': EMBED_DIM,
            'context_len': CONTEXT_LEN,
            'batch_size': BATCH_SIZE,
            'initial_neurons': INITIAL_NEURONS,
            'max_neurons': MAX_NEURONS,
            'corpus': '1865-1900-yorkshire + 1865-1900',
            'corpus_tokens': len(tokens)
        }
    }
    pca_model = None

    # Sample prompts for generation
    prompts = [
        "In Yorkshire,",
        "The village of",
        "It was a cold winter",
        "The farmer said",
    ]
    prompt_idx = 0

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

            model.attention.accumulate_gradients()

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
            model.attention.current_epoch = epoch
            model.attention.freeze_cold_neurons(epoch, EPOCHS)

            if model.num_neurons < MAX_NEURONS:
                active = model.num_neurons - model.num_frozen
                progress = epoch / EPOCHS

                if progress < 0.3:
                    base_grow = 32
                elif progress < 0.6:
                    base_grow = 16
                else:
                    base_grow = 8

                grow = min(base_grow, MAX_NEURONS - model.num_neurons, max(active // 2, 2))
                model.attention.grow_neurons(grow)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        # Generate sample (rotate through prompts)
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1
        sample = generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8)

        # Calculate stats
        active_neurons = model.num_neurons - model.num_frozen
        speedup = model.num_neurons / max(active_neurons, 1)

        progress = epoch / EPOCHS
        if progress < 0.2:
            phase = "GROW"
        elif progress < 0.5:
            phase = "grow+freeze"
        elif progress < 0.8:
            phase = "FREEZE"
        else:
            phase = "CRYSTALLIZE"

        frozen_pct = 100 * model.num_frozen / model.num_neurons
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | N: {model.num_neurons} | F: {model.num_frozen} ({frozen_pct:.0f}%) | S: {speedup:.1f}x | {phase}")
        print(f"         -> {sample[:90]}...")

        # Append to JSONL
        epoch_data = {
            'epoch': epoch,
            'loss': avg_loss,
            'neurons': model.num_neurons,
            'frozen': model.num_frozen,
            'sample': sample[:250],
            'phase': phase
        }
        with open(f'{run_dir}/history.jsonl', 'a') as f:
            f.write(json.dumps(epoch_data) + '\n')

        # Visualize every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            pca_model = visualize_crystal(model, epoch, run_dir, sample, pca_model)

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
    sample = generate(model, tokenizer, "In Yorkshire,", max_tokens=200, temperature=0.8)
    visualize_crystal(model, EPOCHS, run_dir, sample, pca_model)

    # Save final config
    config_data = {
        'config': history['config'],
        'final_neurons': model.num_neurons,
        'final_frozen': model.num_frozen,
        'best_loss': best_loss,
        'total_epochs': EPOCHS
    }
    with open(f'{run_dir}/config.json', 'w') as f:
        json.dump(config_data, f, indent=2)

    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"Final: {model.num_neurons} neurons, {model.num_frozen} frozen ({100*model.num_frozen/model.num_neurons:.0f}%)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Output: {run_dir}/")
    print("=" * 70)
    print("\nFinal generation:")
    print(sample)


if __name__ == "__main__":
    main()
