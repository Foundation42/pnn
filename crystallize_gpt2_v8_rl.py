#!/usr/bin/env python3
"""
Crystallize GPT-2 v8: RL FROM GPT-2 WITH CLEAN DATA

Key innovations:
1. Generate clean training data FROM GPT-2 (no WikiText junk)
2. RL training: Crystal generates, GPT-2 scores
3. Policy gradient to optimize for sequence-level coherence

"Let GPT-2 teach through evaluation, not just imitation."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import os
import time
from datetime import datetime

# Create output directory
RUN_NAME = f"crystal_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(f"runs/{RUN_NAME}", exist_ok=True)

print("=" * 70)
print("CRYSTALLIZE GPT-2 v8: RL WITH CLEAN DATA")
print(f"Output: runs/{RUN_NAME}/")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
INITIAL_NEURONS = 16
MAX_NEURONS = 512
NUM_BLOCKS = 6
EMBED_DIM = 768
VOCAB_SIZE = 50257
SEQ_LEN = 64              # Shorter sequences for RL
BATCH_SIZE = 8
EPOCHS = 600
LR = 1e-4
VIZ_EVERY = 10

# RL parameters
GENERATION_LEN = 32       # How many tokens crystal generates
TEMPERATURE = 0.8         # Sampling temperature
BASELINE_DECAY = 0.99     # For variance reduction
KL_COEF = 0.1            # KL penalty to prevent divergence

# Growth/freeze parameters
GROW_EVERY = 5
FREEZE_THRESHOLD_PCT = 20
MIN_COLD_EPOCHS = 3

# Clean prompts for data generation
PROMPT_TEMPLATES = [
    "The scientist discovered that",
    "In the year 2050,",
    "The president announced",
    "According to recent studies,",
    "The technology revolution",
    "Scientists have found",
    "The research team",
    "A new discovery shows",
    "The experiment revealed",
    "Experts believe that",
    "The study concluded",
    "Recent findings indicate",
    "The breakthrough came when",
    "Researchers at the university",
    "The data suggests that",
    "Analysis of the results",
    "The theory proposes that",
    "Evidence shows that",
    "The investigation found",
    "New research demonstrates",
    "The ancient civilization",
    "Historians believe",
    "The manuscript reveals",
    "Archaeological evidence",
    "The discovery of",
    "Once upon a time,",
    "In a distant land,",
    "The young hero",
    "The kingdom was",
    "Long ago, there lived",
    "The neural network learned",
    "Machine learning models",
    "Artificial intelligence",
    "The algorithm discovered",
    "Deep learning systems",
]

print(f"""
Configuration:
  - Initial neurons: {INITIAL_NEURONS} per block
  - Max neurons: {MAX_NEURONS} per block
  - Generation length: {GENERATION_LEN} tokens
  - Temperature: {TEMPERATURE}
  - Epochs: {EPOCHS}
  - Training: RL with GPT-2 as reward model
""")

# ============================================================================
# CRYSTAL ARCHITECTURE (same as before)
# ============================================================================

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

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def generate_with_logprobs(self, idx, max_new_tokens, temperature=0.8):
        """Generate tokens and return log probabilities for RL."""
        generated_tokens = []
        log_probs = []

        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :]
            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Get log prob of selected token
            log_prob = F.log_softmax(logits, dim=-1)
            selected_log_prob = log_prob.gather(1, next_token)

            generated_tokens.append(next_token)
            log_probs.append(selected_log_prob)

            idx = torch.cat([idx, next_token], dim=1)

        generated_tokens = torch.cat(generated_tokens, dim=1)
        log_probs = torch.cat(log_probs, dim=1)

        return idx, generated_tokens, log_probs

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


# ============================================================================
# GPT-2 REWARD MODEL
# ============================================================================

class GPT2Reward(nn.Module):
    """GPT-2 as a reward model - scores sequence likelihood."""
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        for param in self.gpt2.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def score_sequence(self, input_ids):
        """Return average log probability of sequence according to GPT-2."""
        outputs = self.gpt2(input_ids, labels=input_ids)
        # Negative loss = log probability (higher is better)
        return -outputs.loss

    @torch.no_grad()
    def get_logprobs(self, input_ids):
        """Get per-token log probabilities."""
        outputs = self.gpt2(input_ids)
        logits = outputs.logits[:, :-1, :]  # Shift for next token prediction
        targets = input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        return token_log_probs


# ============================================================================
# VISUALIZATION (same as before)
# ============================================================================

def visualize_crystal(crystal, epoch, reward, pca, save_path):
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
    active_count = total - frozen_count
    frozen_pct = 100 * frozen_count / total if total > 0 else 0
    speedup = total / max(active_count, 1)

    if pca is None:
        pca = PCA(n_components=3)
        positions_3d = pca.fit_transform(all_positions)
    else:
        positions_3d = pca.transform(all_positions)

    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    colors = ['blue' if f else 'red' for f in all_frozen]
    sizes = [20 if f else 50 for f in all_frozen]
    ax1.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
               c=colors, s=sizes, alpha=0.6)
    ax1.set_title(f'Epoch {epoch}\nFrozen: {frozen_count} | Active: {active_count}')

    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    scatter = ax2.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
                         c=all_temps, cmap='coolwarm', s=30, alpha=0.7)
    ax2.set_title('Temperature')
    plt.colorbar(scatter, ax=ax2, shrink=0.5)

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
    ax5.set_title('Temperature Distribution')
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
    ax7.text(0.5, 0.5, f'Epoch {epoch}\nReward: {reward:.3f}\nFrozen: {frozen_pct:.1f}%\nSpeedup: {speedup:.1f}x',
            transform=ax7.transAxes, fontsize=14, ha='center', va='center')

    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    ax8.text(0.5, 0.5, 'RL TRAINING\n(GPT-2 Reward)',
            transform=ax8.transAxes, fontsize=14, ha='center', va='center', color='green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

    return pca


# ============================================================================
# GENERATE CLEAN TRAINING DATA FROM GPT-2
# ============================================================================

def generate_clean_prompts(gpt2_model, tokenizer, num_prompts=500):
    """Generate clean training prompts using GPT-2."""
    print("\n[*] Generating clean training data from GPT-2...")

    clean_data = []
    gpt2_model.eval()

    for i, template in enumerate(PROMPT_TEMPLATES * (num_prompts // len(PROMPT_TEMPLATES) + 1)):
        if len(clean_data) >= num_prompts:
            break

        input_ids = tokenizer.encode(template, return_tensors='pt').to(device)

        # Generate completion
        with torch.no_grad():
            output = gpt2_model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        clean_data.append(text)

        if (i + 1) % 100 == 0:
            print(f"    Generated {len(clean_data)}/{num_prompts} samples")

    print(f"    Total clean samples: {len(clean_data)}")
    return clean_data


# ============================================================================
# MAIN TRAINING
# ============================================================================

print("\n[1] Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

gpt2_for_generation = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_for_generation.eval()

reward_model = GPT2Reward().to(device)
reward_model.eval()

# Generate clean training data
clean_data = generate_clean_prompts(gpt2_for_generation, tokenizer, num_prompts=500)

# Show samples
print("\nClean data samples:")
for i in range(3):
    print(f"  {i+1}. {clean_data[i][:100]}...")

print("\n[2] Creating Crystal...")
crystal = CrystalGPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    initial_neurons=INITIAL_NEURONS,
    num_blocks=NUM_BLOCKS,
    max_neurons=MAX_NEURONS
).to(device)

print(f"    Initial neurons: {INITIAL_NEURONS * NUM_BLOCKS}")

optimizer = torch.optim.AdamW(crystal.parameters(), lr=LR)

# ============================================================================
# RL TRAINING LOOP
# ============================================================================

print(f"\n[3] RL Training ({EPOCHS} epochs)...")
print("=" * 70)

history = {'reward': [], 'policy_loss': [], 'neurons': [], 'frozen': [], 'active': [], 'speedup': []}
pca = None
start_time = time.time()
baseline = 0.0  # Running baseline for variance reduction

for epoch in range(EPOCHS):
    crystal.train()
    epoch_rewards = []
    epoch_policy_loss = []

    np.random.shuffle(clean_data)

    for i in range(0, len(clean_data), BATCH_SIZE):
        batch_texts = clean_data[i:i + BATCH_SIZE]
        if len(batch_texts) < BATCH_SIZE:
            continue

        # Get prompts (first part of each text)
        prompts = []
        for text in batch_texts:
            tokens = tokenizer.encode(text)[:20]  # First 20 tokens as prompt
            prompts.append(tokens)

        # Pad prompts
        max_len = max(len(p) for p in prompts)
        padded_prompts = [p + [tokenizer.eos_token_id] * (max_len - len(p)) for p in prompts]
        prompt_ids = torch.tensor(padded_prompts, device=device)

        optimizer.zero_grad()

        # Crystal generates continuation
        full_seq, generated_tokens, crystal_logprobs = crystal.generate_with_logprobs(
            prompt_ids, GENERATION_LEN, temperature=TEMPERATURE
        )

        # GPT-2 scores the full sequence
        with torch.no_grad():
            reward = reward_model.score_sequence(full_seq)

        # Compute advantage (reward - baseline)
        advantage = reward - baseline

        # Update baseline
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward.item()

        # Policy gradient loss: -log_prob * advantage
        policy_loss = -(crystal_logprobs.mean(dim=1) * advantage).mean()

        # Optional: Add supervised loss on clean data for stability
        clean_ids = tokenizer(batch_texts, return_tensors='pt', padding=True,
                             truncation=True, max_length=SEQ_LEN)['input_ids'].to(device)
        if clean_ids.shape[1] > 1:
            logits = crystal(clean_ids[:, :-1])
            supervised_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                clean_ids[:, 1:].reshape(-1)
            )
            total_loss = policy_loss + 0.5 * supervised_loss
        else:
            total_loss = policy_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(crystal.parameters(), 1.0)
        optimizer.step()

        epoch_rewards.append(reward.item())
        epoch_policy_loss.append(policy_loss.item())

    avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
    avg_policy_loss = np.mean(epoch_policy_loss) if epoch_policy_loss else 0

    # Growth and freezing
    if (epoch + 1) % GROW_EVERY == 0:
        crystal.grow_and_freeze()

    total, frozen, active = crystal.get_stats()
    speedup = total / max(active, 1)

    history['reward'].append(avg_reward)
    history['policy_loss'].append(avg_policy_loss)
    history['neurons'].append(total)
    history['frozen'].append(frozen)
    history['active'].append(active)
    history['speedup'].append(speedup)

    # Visualization
    if (epoch + 1) % VIZ_EVERY == 0 or epoch == 0:
        pca = visualize_crystal(crystal, epoch + 1, avg_reward, pca,
                               f"runs/{RUN_NAME}/epoch_{epoch+1:03d}.png")

    # Progress
    elapsed = (time.time() - start_time) / 60
    frozen_pct = 100 * frozen / total if total > 0 else 0

    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Reward: {avg_reward:.3f} | N: {total} | "
              f"F: {frozen} ({frozen_pct:.0f}%) | S: {speedup:.1f}x | T: {elapsed:.1f}m")

        # Generate sample
        crystal.eval()
        prompt = tokenizer.encode("The scientist discovered that", return_tensors='pt').to(device)
        output = crystal.generate(prompt, max_new_tokens=40, temperature=0.7)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"         -> \"{text[:80]}...\"")
        print()

# ============================================================================
# SAVE AND SUMMARIZE
# ============================================================================

print("\n[4] Saving results...")

# Summary plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].plot(history['reward'], 'g-', linewidth=2)
axes[0, 0].set_title('GPT-2 Reward (higher=better)')
axes[0, 0].set_xlabel('Epoch')

axes[0, 1].plot(history['neurons'], 'g-', label='Total')
axes[0, 1].plot(history['frozen'], 'b-', label='Frozen')
axes[0, 1].set_title('Crystal Growth')
axes[0, 1].legend()

axes[0, 2].plot([100*f/n for f,n in zip(history['frozen'], history['neurons'])], 'm-')
axes[0, 2].set_title('Crystallization %')

axes[1, 0].plot(history['speedup'], 'orange')
axes[1, 0].set_title('Speedup')

axes[1, 1].plot(history['policy_loss'], 'r-')
axes[1, 1].set_title('Policy Loss')

total, frozen, active = crystal.get_stats()
elapsed = (time.time() - start_time) / 60
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, f"RL Training Complete\n\nTime: {elapsed:.1f}m\nNeurons: {total}\nFrozen: {frozen} ({100*frozen/total:.1f}%)\nFinal Reward: {history['reward'][-1]:.3f}",
               transform=axes[1, 2].transAxes, fontsize=11, ha='center', va='center')

plt.tight_layout()
plt.savefig(f"runs/{RUN_NAME}/summary.png", dpi=150)

torch.save({
    'model': crystal.state_dict(),
    'history': history,
    'config': {
        'neurons': INITIAL_NEURONS,
        'blocks': NUM_BLOCKS,
        'max': MAX_NEURONS,
        'training': 'rl'
    }
}, f"runs/{RUN_NAME}/crystal_final.pt")

# Generation test
print("\n" + "=" * 70)
print("[5] GENERATION TEST")
print("=" * 70)

crystal.eval()
test_prompts = [
    "The scientist discovered that",
    "In the year 2050,",
    "The neural network learned",
    "According to recent studies,",
    "The breakthrough came when",
]

print("\nCrystal Generation (RL trained on clean data):\n")
for prompt in test_prompts:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = crystal.generate(input_ids, max_new_tokens=50, temperature=0.7)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"  {text}")
    print()

print("=" * 70)
print("RL TRAINING COMPLETE!")
print("=" * 70)
print(f"""
    Training time: {elapsed:.1f} minutes

    CRYSTAL STATS:
    - Total neurons: {total}
    - Frozen: {frozen} ({100*frozen/total:.1f}%)
    - Active: {active}
    - Speedup: {history['speedup'][-1]:.1f}x

    RL STATS:
    - Final reward: {history['reward'][-1]:.3f}
    - Baseline: {baseline:.3f}

    Output: runs/{RUN_NAME}/

    "Learned through evaluation, not just imitation."
""")
