#!/usr/bin/env python3
"""
Modal: Crystal v9 - TinyStories From Scratch

Train a crystal from scratch on TinyStories dataset.
TinyStories has clean, coherent, simple stories - perfect for small LMs.

Run with:
    modal run modal_crystal_v9_tinystories.py
"""

import modal

app = modal.App("crystal-tinystories-v9")
volume = modal.Volume.from_name("crystal-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "matplotlib",
        "numpy",
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    volumes={"/outputs": volume},
)
def train_tinystories():
    """Train Crystal from scratch on TinyStories."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import GPT2Tokenizer
    from datasets import load_dataset
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
    import os
    import time
    from datetime import datetime

    RUN_NAME = f"crystal_tinystories_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(f"/outputs/{RUN_NAME}", exist_ok=True)

    print("=" * 70)
    print("CRYSTAL v9: TINYSTORIES FROM SCRATCH (H100)")
    print(f"Output: /outputs/{RUN_NAME}/")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ========================================================================
    # HYPERPARAMETERS
    # ========================================================================
    INITIAL_NEURONS = 16
    MAX_NEURONS = 512
    NUM_BLOCKS = 6
    EMBED_DIM = 768
    VOCAB_SIZE = 50257
    SEQ_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 600
    LR = 3e-4
    VIZ_EVERY = 10

    GROW_EVERY = 5
    FREEZE_THRESHOLD_PCT = 20
    MIN_COLD_EPOCHS = 3

    print(f"""
Configuration:
  - Initial neurons: {INITIAL_NEURONS} per block ({INITIAL_NEURONS * NUM_BLOCKS} total)
  - Max neurons: {MAX_NEURONS} per block ({MAX_NEURONS * NUM_BLOCKS} total)
  - Batch size: {BATCH_SIZE}
  - Sequence length: {SEQ_LEN}
  - Epochs: {EPOCHS}
  - Dataset: TinyStories (clean, coherent stories)
""")

    # ========================================================================
    # CRYSTAL ARCHITECTURE
    # ========================================================================

    class CrystalAttention(nn.Module):
        def __init__(self, embed_dim, initial_neurons, max_neurons):
            super().__init__()
            self.embed_dim = embed_dim
            self.max_neurons = max_neurons
            self.num_neurons = initial_neurons

            self.positions = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.02)
            self.scales = nn.Parameter(torch.ones(max_neurons) * 5.0)
            self.values = nn.Parameter(torch.randn(max_neurons, embed_dim) * 0.02)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

            self.register_buffer('frozen', torch.zeros(max_neurons, dtype=torch.bool))
            self.register_buffer('temperature', torch.ones(max_neurons))
            self.register_buffer('grad_ema', torch.zeros(max_neurons))
            self.register_buffer('cold_epochs', torch.zeros(max_neurons))

        def forward(self, x):
            B, T, D = x.shape
            N = self.num_neurons
            pos = self.positions[:N].unsqueeze(0).unsqueeze(0)
            x_exp = x.unsqueeze(2)
            dist = torch.norm(x_exp - pos, dim=-1)
            attn = F.softmax(self.scales[:N] / (dist + 0.1), dim=-1)
            out = torch.einsum('btn,nd->btd', attn, self.values[:N])
            return self.out_proj(out)

        def update_stats(self):
            if self.positions.grad is not None:
                N = self.num_neurons
                grad_norm = self.positions.grad[:N].norm(dim=-1).detach()
                self.grad_ema[:N] = 0.9 * self.grad_ema[:N] + 0.1 * grad_norm
                self.temperature[:N] = self.grad_ema[:N]

        def get_hot_neurons(self, top_k=2):
            N = self.num_neurons
            temps = self.temperature[:N].clone()
            temps[self.frozen[:N]] = -1
            if (temps > 0).sum() < top_k:
                return []
            _, indices = temps.topk(min(top_k, N))
            return [i.item() for i in indices if temps[i] > 0]

        def get_cold_neurons(self, threshold_pct=20, min_cold=3):
            N = self.num_neurons
            active = ~self.frozen[:N]
            if active.sum() < 5:
                return []
            temps = self.temperature[:N]
            threshold = torch.quantile(temps[active], threshold_pct/100)
            candidates = []
            for i in range(N):
                if not self.frozen[i]:
                    if temps[i] < threshold:
                        self.cold_epochs[i] += 1
                        if self.cold_epochs[i] >= min_cold:
                            candidates.append(i)
                    else:
                        self.cold_epochs[i] = 0
            return candidates

        def split(self, idx):
            if self.num_neurons >= self.max_neurons:
                return False
            new = self.num_neurons
            with torch.no_grad():
                noise = 0.01
                self.positions.data[new] = self.positions.data[idx] + \
                    torch.randn(self.embed_dim, device=self.positions.device) * noise
                self.scales.data[new] = self.scales.data[idx]
                self.values.data[new] = self.values.data[idx] + \
                    torch.randn(self.embed_dim, device=self.values.device) * noise
                self.temperature[new] = self.temperature[idx]
                self.cold_epochs[new] = 0
            self.num_neurons += 1
            return True

        def freeze(self, idx):
            self.frozen[idx] = True

    class CrystalBlock(nn.Module):
        def __init__(self, embed_dim, initial_neurons, max_neurons):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_dim)
            self.attn = CrystalAttention(embed_dim, initial_neurons, max_neurons)
            self.ln2 = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class CrystalGPT(nn.Module):
        def __init__(self, vocab_size, embed_dim, initial_neurons, num_blocks, max_neurons):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_blocks = num_blocks
            self.wte = nn.Embedding(vocab_size, embed_dim)
            self.wpe = nn.Embedding(1024, embed_dim)
            self.blocks = nn.ModuleList([
                CrystalBlock(embed_dim, initial_neurons, max_neurons)
                for _ in range(num_blocks)
            ])
            self.ln_f = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, vocab_size, bias=False)
            self.head.weight = self.wte.weight

        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.wte(idx)
            pos_emb = self.wpe(torch.arange(T, device=idx.device))
            x = tok_emb + pos_emb
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln_f(x))

            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                return logits, loss
            return logits

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=0.7, top_k=50):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -1024:]
                logits = self(idx_cond)[:, -1, :]
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
            return idx

        def get_stats(self):
            total = sum(b.attn.num_neurons for b in self.blocks)
            frozen = sum(b.attn.frozen[:b.attn.num_neurons].sum().item() for b in self.blocks)
            return total, frozen, total - frozen

        def grow_and_freeze(self):
            grown = 0
            frozen_count = 0
            for block in self.blocks:
                attn = block.attn
                attn.update_stats()
                cold = attn.get_cold_neurons(FREEZE_THRESHOLD_PCT, MIN_COLD_EPOCHS)
                for idx in cold:
                    attn.freeze(idx)
                    frozen_count += 1
                hot = attn.get_hot_neurons(top_k=2)
                for idx in hot:
                    if attn.split(idx):
                        grown += 1
            return grown, frozen_count

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def visualize_crystal(crystal, epoch, loss, pca, save_path):
        fig = plt.figure(figsize=(20, 12))

        all_positions = []
        all_frozen = []
        all_temps = []
        block_labels = []

        for block_idx, block in enumerate(crystal.blocks):
            attn = block.attn
            N = attn.num_neurons
            positions = attn.positions[:N].detach().cpu().numpy()
            frozen = attn.frozen[:N].detach().cpu().numpy()
            temps = attn.temperature[:N].detach().cpu().numpy()
            all_positions.append(positions)
            all_frozen.append(frozen)
            all_temps.append(temps)
            block_labels.extend([block_idx] * N)

        if len(all_positions) == 0:
            plt.close()
            return pca

        all_positions = np.vstack(all_positions)
        all_frozen = np.concatenate(all_frozen)
        all_temps = np.concatenate(all_temps)
        block_labels = np.array(block_labels)

        total = len(all_frozen)
        frozen_count = all_frozen.sum()
        frozen_pct = 100 * frozen_count / total if total > 0 else 0
        speedup = total / max(total - frozen_count, 1)

        if pca is None:
            pca = PCA(n_components=3)
            positions_3d = pca.fit_transform(all_positions)
        else:
            positions_3d = pca.transform(all_positions)

        ax1 = fig.add_subplot(2, 4, 1, projection='3d')
        colors = ['blue' if f else 'red' for f in all_frozen]
        ax1.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                   c=colors, s=[20 if f else 50 for f in all_frozen], alpha=0.6)
        ax1.set_title(f'Epoch {epoch}\nFrozen: {frozen_count} | Active: {total-frozen_count}')

        ax2 = fig.add_subplot(2, 4, 2, projection='3d')
        scatter = ax2.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                             c=all_temps, cmap='coolwarm', s=30, alpha=0.7)
        ax2.set_title('Temperature')

        ax3 = fig.add_subplot(2, 4, 3, projection='3d')
        ax3.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                   c=block_labels, cmap='tab10', s=30, alpha=0.7)
        ax3.set_title(f'By Layer ({total} neurons)')

        ax4 = fig.add_subplot(2, 4, 4)
        colors = ['blue' if f else 'red' for f in all_frozen]
        ax4.scatter(positions_3d[:, 0], positions_3d[:, 1], c=colors, s=20, alpha=0.6)
        ax4.set_title('2D Projection')

        ax5 = fig.add_subplot(2, 4, 5)
        ax5.hist(all_temps[all_frozen], bins=30, alpha=0.7, label='Frozen', color='blue')
        ax5.hist(all_temps[~all_frozen], bins=30, alpha=0.7, label='Active', color='red')
        ax5.set_title('Temperature')
        ax5.legend()

        ax6 = fig.add_subplot(2, 4, 6)
        block_totals = [b.attn.num_neurons for b in crystal.blocks]
        block_frozen = [b.attn.frozen[:b.attn.num_neurons].sum().item() for b in crystal.blocks]
        x = np.arange(len(block_totals))
        ax6.bar(x, block_totals, 0.35, label='Total', color='green', alpha=0.7)
        ax6.bar(x, block_frozen, 0.35, label='Frozen', color='blue', alpha=0.7)
        ax6.set_title('Neurons per Block')
        ax6.legend()

        ax7 = fig.add_subplot(2, 4, 7)
        ax7.axis('off')
        ax7.text(0.5, 0.5, f'Epoch {epoch}\nLoss: {loss:.4f}\nFrozen: {frozen_pct:.1f}%',
                transform=ax7.transAxes, fontsize=14, ha='center', va='center')

        ax8 = fig.add_subplot(2, 4, 8)
        ax8.axis('off')
        ax8.text(0.5, 0.5, 'TINYSTORIES\nFROM SCRATCH\n(H100)',
                transform=ax8.transAxes, fontsize=14, ha='center', va='center', color='purple')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close()
        return pca

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    print("\n[1] Loading TinyStories dataset...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load TinyStories - clean, simple stories
    dataset = load_dataset('roneneldan/TinyStories', split='train')

    # Get clean stories
    stories = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 100 and len(text) < 2000:  # Filter by length
            stories.append(text)
        if len(stories) >= 5000:
            break

    print(f"    Loaded {len(stories)} clean stories")

    # Show samples
    print("\nSample stories:")
    for i in range(3):
        print(f"  {i+1}. {stories[i][:100]}...")

    # Tokenize and create sequences
    print("\n[2] Tokenizing...")
    all_tokens = []
    for story in stories:
        tokens = tokenizer.encode(story, add_special_tokens=False)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)

    sequences = []
    for i in range(0, len(all_tokens) - SEQ_LEN - 1, SEQ_LEN // 2):
        seq = all_tokens[i:i + SEQ_LEN + 1]
        if len(seq) == SEQ_LEN + 1:
            sequences.append(seq)

    print(f"    Created {len(sequences)} training sequences")

    # ========================================================================
    # CREATE MODEL
    # ========================================================================

    print("\n[3] Creating Crystal...")
    crystal = CrystalGPT(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        initial_neurons=INITIAL_NEURONS,
        num_blocks=NUM_BLOCKS,
        max_neurons=MAX_NEURONS
    ).to(device)

    optimizer = torch.optim.AdamW(crystal.parameters(), lr=LR)

    # ========================================================================
    # TRAINING
    # ========================================================================

    print(f"\n[4] Training ({EPOCHS} epochs)...")
    print("=" * 70)

    history = {'loss': [], 'neurons': [], 'frozen': [], 'active': [], 'speedup': []}
    pca = None
    start_time = time.time()

    for epoch in range(EPOCHS):
        crystal.train()
        epoch_loss = 0
        num_batches = 0

        np.random.shuffle(sequences)

        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seqs = sequences[i:i + BATCH_SIZE]
            if len(batch_seqs) < BATCH_SIZE:
                continue

            batch = torch.tensor(batch_seqs, device=device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits, loss = crystal(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(crystal.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % GROW_EVERY == 0:
            crystal.grow_and_freeze()

        total, frozen, active = crystal.get_stats()
        speedup = total / max(active, 1)

        history['loss'].append(avg_loss)
        history['neurons'].append(total)
        history['frozen'].append(frozen)
        history['active'].append(active)
        history['speedup'].append(speedup)

        if (epoch + 1) % VIZ_EVERY == 0 or epoch == 0:
            pca = visualize_crystal(crystal, epoch + 1, avg_loss, pca,
                                   f"/outputs/{RUN_NAME}/epoch_{epoch+1:03d}.png")

        elapsed = (time.time() - start_time) / 60
        frozen_pct = 100 * frozen / total if total > 0 else 0

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | N: {total} | "
                  f"F: {frozen} ({frozen_pct:.0f}%) | S: {speedup:.1f}x | T: {elapsed:.1f}m")

            crystal.eval()
            prompt = tokenizer.encode("Once upon a time", return_tensors='pt').to(device)
            output = crystal.generate(prompt, max_new_tokens=40, temperature=0.7)
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"         -> \"{text[:80]}...\"")

    # ========================================================================
    # SAVE
    # ========================================================================

    print("\n[5] Saving...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].plot(history['loss'], 'b-')
    axes[0, 0].set_title('Loss')
    axes[0, 1].plot(history['neurons'], 'g-', label='Total')
    axes[0, 1].plot(history['frozen'], 'b-', label='Frozen')
    axes[0, 1].set_title('Growth')
    axes[0, 1].legend()
    axes[0, 2].plot([100*f/n for f,n in zip(history['frozen'], history['neurons'])], 'm-')
    axes[0, 2].set_title('Crystallization %')
    axes[1, 0].plot(history['speedup'], 'orange')
    axes[1, 0].set_title('Speedup')
    axes[1, 1].axis('off')
    total, frozen, active = crystal.get_stats()
    elapsed = (time.time() - start_time) / 60
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, f"TinyStories Training\n\nTime: {elapsed:.1f}m\nNeurons: {total}\nFrozen: {frozen} ({100*frozen/total:.1f}%)",
                   transform=axes[1, 2].transAxes, fontsize=11, ha='center', va='center')
    plt.tight_layout()
    plt.savefig(f"/outputs/{RUN_NAME}/summary.png", dpi=150)

    torch.save({
        'model': crystal.state_dict(),
        'history': history,
        'config': {'neurons': INITIAL_NEURONS, 'blocks': NUM_BLOCKS, 'max': MAX_NEURONS}
    }, f"/outputs/{RUN_NAME}/crystal_final.pt")

    # Generation test
    print("\n" + "=" * 70)
    print("[6] GENERATION TEST")
    print("=" * 70)

    crystal.eval()
    prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a boy",
        "There was a",
        "The cat and the dog",
    ]

    print("\nCrystal Generation (TinyStories trained):\n")
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output = crystal.generate(input_ids, max_new_tokens=60, temperature=0.7)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"  {text}")
        print()

    print("=" * 70)
    print(f"COMPLETE! Output: /outputs/{RUN_NAME}/")
    print(f"Time: {elapsed:.1f}m | Neurons: {total} | Frozen: {frozen}")
    print("=" * 70)

    volume.commit()

    return {'run_name': RUN_NAME, 'neurons': total, 'frozen': frozen, 'elapsed': elapsed}


@app.local_entrypoint()
def main():
    print("Starting TinyStories training on Modal H100...")
    result = train_tinystories.remote()
    print(f"\nComplete! Results: {result}")
    print(f"\nDownload: modal volume get crystal-outputs {result['run_name']} ./")
