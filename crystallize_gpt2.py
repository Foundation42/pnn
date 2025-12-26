#!/usr/bin/env python3
"""
Crystallize GPT-2: Distill a pre-trained transformer into geometric space.

The crazy idea:
1. Load GPT-2 (124M params)
2. Use its gradients as a "heat map" of where knowledge lives
3. Seed geometric neurons in high-gradient regions
4. Let them crystallize by distilling FROM the teacher
5. The crystal self-optimizes: freeze where stable, grow where needed

"Knowledge crystallization from pre-trained models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("=" * 60)
print("CRYSTALLIZE GPT-2: Knowledge → Geometry")
print("=" * 60)

# Load GPT-2
print("\n[1] Loading GPT-2...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
teacher = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
teacher.eval()

# Model stats
num_params = sum(p.numel() for p in teacher.parameters())
print(f"    Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
print(f"    Layers: {teacher.config.n_layer}")
print(f"    Heads: {teacher.config.n_head}")
print(f"    Embedding dim: {teacher.config.n_embd}")
print(f"    Vocab size: {teacher.config.vocab_size}")

# Test it works
print("\n[2] Testing GPT-2...")
test_text = "The crystal structure of intelligence"
inputs = tokenizer(test_text, return_tensors='pt').to(device)
with torch.no_grad():
    outputs = teacher(**inputs)
    next_token_logits = outputs.logits[0, -1, :]
    next_token = torch.argmax(next_token_logits).item()
    print(f"    Input: '{test_text}'")
    print(f"    Next token: '{tokenizer.decode([next_token])}'")


print("\n[3] Analyzing gradient flow...")
# Run some text through and collect gradients
sample_texts = [
    "The meaning of life is",
    "In the beginning there was",
    "The neural network learned to",
    "Mathematics is the language of",
    "Consciousness emerges from",
]

# Enable gradients for analysis
teacher.train()  # Enable dropout for gradient variance
for p in teacher.parameters():
    p.requires_grad_(True)

# Collect gradient statistics per layer
layer_gradients = defaultdict(list)
attention_patterns = []

for text in sample_texts:
    inputs = tokenizer(text, return_tensors='pt').to(device)

    # Forward pass with gradient tracking
    outputs = teacher(**inputs, output_attentions=True)

    # Get loss (predict next token)
    labels = inputs['input_ids'].clone()
    loss = F.cross_entropy(
        outputs.logits[:, :-1, :].reshape(-1, teacher.config.vocab_size),
        labels[:, 1:].reshape(-1)
    )

    # Backward pass
    loss.backward()

    # Collect attention patterns
    for layer_idx, attn in enumerate(outputs.attentions):
        attention_patterns.append({
            'layer': layer_idx,
            'pattern': attn.detach().cpu().numpy()
        })

    # Collect gradients per layer
    for name, param in teacher.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            layer_gradients[name].append(grad_norm)

    teacher.zero_grad()

# Analyze gradient distribution
print("\n    Gradient norms by component:")
component_grads = defaultdict(float)
for name, grads in layer_gradients.items():
    avg_grad = np.mean(grads)
    # Group by component type
    if 'attn' in name:
        component_grads['attention'] += avg_grad
    elif 'mlp' in name:
        component_grads['mlp'] += avg_grad
    elif 'wte' in name or 'wpe' in name:
        component_grads['embedding'] += avg_grad
    elif 'ln' in name:
        component_grads['layernorm'] += avg_grad

for comp, grad in sorted(component_grads.items(), key=lambda x: -x[1]):
    print(f"      {comp}: {grad:.4f}")


print("\n[4] Designing Crystal Architecture...")
print("""
    The key insight: GPT-2's knowledge lives in:
    - Attention patterns (what to focus on)
    - MLP weights (how to transform)
    - Embeddings (what words mean)

    Our crystal will have neurons that:
    - Live in embedding space (768-dim → project to 3D for viz)
    - Compute attention-like interactions (1/distance weighting)
    - Transform via learned weights (like MLP)

    Instead of 12 layers × 12 heads = 144 attention patterns,
    we'll have N neurons that interact geometrically.
""")


# ============================================================================
# THE CRYSTAL GPT-2 ARCHITECTURE
# ============================================================================

class CrystalAttention(nn.Module):
    """
    Geometric attention: neurons interact based on distance in embedding space.

    Instead of Q,K,V matrices, neurons have:
    - Positions in embedding space
    - Interaction strength based on distance
    - Value projections
    """
    def __init__(self, embed_dim, num_neurons, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Neuron positions in embedding space (768-dim for GPT-2)
        self.positions = nn.Parameter(torch.randn(num_neurons, embed_dim) * 0.1)

        # Interaction scale (learnable)
        self.interaction_scale = nn.Parameter(torch.ones(num_neurons) * 10.0)

        # Value projection per neuron
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Temperature tracking for freezing
        self.register_buffer('temperature', torch.ones(num_neurons))
        self.register_buffer('grad_history', torch.zeros(num_neurons, 10))
        self.grad_ptr = 0

    def forward(self, x, return_interactions=False):
        """
        x: (batch, seq_len, embed_dim)

        Each token attends to neurons based on distance in embedding space.
        """
        B, T, D = x.shape

        # Compute distances from each token to each neuron
        # x: (B, T, D), positions: (N, D)
        # distances: (B, T, N)
        x_expanded = x.unsqueeze(2)  # (B, T, 1, D)
        pos_expanded = self.positions.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

        distances = torch.norm(x_expanded - pos_expanded, dim=-1)  # (B, T, N)

        # Interaction strength: 1 / (distance + epsilon)
        # Scaled by learned interaction_scale
        interactions = self.interaction_scale / (distances + 0.1)  # (B, T, N)

        # Softmax to get attention weights
        attn_weights = F.softmax(interactions, dim=-1)  # (B, T, N)

        # Value projection of neuron positions
        values = self.value_proj(self.positions)  # (N, D)

        # Weighted sum of values
        # attn_weights: (B, T, N), values: (N, D)
        output = torch.einsum('btn,nd->btd', attn_weights, values)

        # Output projection
        output = self.out_proj(output)

        if return_interactions:
            return output, interactions
        return output

    def update_temperature(self):
        """Update temperature based on gradient history (for freezing)."""
        if self.positions.grad is not None:
            grad_norm = self.positions.grad.norm(dim=-1)  # (N,)
            self.grad_history[:, self.grad_ptr] = grad_norm.detach()
            self.grad_ptr = (self.grad_ptr + 1) % 10

            # Temperature = recent gradient variance
            self.temperature = self.grad_history.std(dim=-1) + 1e-6


class CrystalMLP(nn.Module):
    """
    Geometric MLP: transform based on neuron activations.
    """
    def __init__(self, embed_dim, num_neurons, expansion=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        # Neuron-specific transformations
        self.up_proj = nn.Linear(embed_dim, num_neurons * expansion)
        self.down_proj = nn.Linear(num_neurons * expansion, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class CrystalBlock(nn.Module):
    """One block of the crystal transformer."""
    def __init__(self, embed_dim, num_neurons, num_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CrystalAttention(embed_dim, num_neurons, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = CrystalMLP(embed_dim, num_neurons)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CrystalGPT2(nn.Module):
    """
    GPT-2 distilled into geometric crystal form.

    Instead of 12 layers × 12 heads with Q,K,V matrices,
    we have N neurons in embedding space that interact geometrically.
    """
    def __init__(self, vocab_size, embed_dim, num_neurons, num_blocks, max_seq_len=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neurons = num_neurons

        # Embeddings (shared with teacher initially)
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(max_seq_len, embed_dim)

        # Crystal blocks
        self.blocks = nn.ModuleList([
            CrystalBlock(embed_dim, num_neurons)
            for _ in range(num_blocks)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, return_hidden=False):
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # Through crystal blocks
        hiddens = [x]
        for block in self.blocks:
            x = block(x)
            hiddens.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_hidden:
            return logits, hiddens
        return logits

    def get_neuron_positions(self):
        """Get all neuron positions for visualization."""
        positions = []
        for block in self.blocks:
            positions.append(block.attn.positions.detach().cpu())
        return positions

    def get_temperatures(self):
        """Get temperature (activity level) of all neurons."""
        temps = []
        for block in self.blocks:
            temps.append(block.attn.temperature.detach().cpu())
        return temps


print("\n[5] Creating Crystal GPT-2...")

# Start with fewer neurons than GPT-2 has attention heads
# GPT-2: 12 layers × 12 heads = 144 "attention units"
# Crystal: Start with 32 neurons, let it grow
INITIAL_NEURONS = 32
NUM_BLOCKS = 4  # Start simpler than GPT-2's 12 layers

crystal = CrystalGPT2(
    vocab_size=teacher.config.vocab_size,
    embed_dim=teacher.config.n_embd,  # 768
    num_neurons=INITIAL_NEURONS,
    num_blocks=NUM_BLOCKS,
).to(device)

# Initialize embeddings from teacher!
print("    Copying embeddings from teacher...")
crystal.wte.weight.data = teacher.transformer.wte.weight.data.clone()
crystal.wpe.weight.data = teacher.transformer.wpe.weight.data.clone()

crystal_params = sum(p.numel() for p in crystal.parameters())
print(f"    Crystal parameters: {crystal_params:,} ({crystal_params/1e6:.1f}M)")
print(f"    Compression ratio: {num_params/crystal_params:.1f}x")
print(f"    Neurons: {INITIAL_NEURONS} (vs {12*12}=144 attention units in GPT-2)")
print(f"    Blocks: {NUM_BLOCKS} (vs 12 in GPT-2)")


print("\n[6] Knowledge Distillation: Teacher → Crystal")

# Training setup
optimizer = torch.optim.AdamW(crystal.parameters(), lr=1e-4)
teacher.eval()
crystal.train()

# Sample training data
training_texts = [
    "The meaning of life is to find purpose and happiness in our daily existence.",
    "Neural networks learn patterns from data through gradient descent optimization.",
    "In mathematics, a function maps inputs to outputs in a deterministic way.",
    "The universe began with the Big Bang approximately 13.8 billion years ago.",
    "Language models predict the next word based on the previous context.",
    "Consciousness remains one of the greatest mysteries in science.",
    "The crystal structure determines the physical properties of materials.",
    "Artificial intelligence aims to create machines that can think and learn.",
    "Quantum mechanics describes the behavior of particles at atomic scales.",
    "Evolution shaped all living organisms through natural selection.",
]

# Tokenize
train_tokens = []
for text in training_texts:
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    train_tokens.append(tokens['input_ids'].to(device))

print(f"    Training samples: {len(train_tokens)}")

# Distillation loop
NUM_EPOCHS = 100
TEMP = 2.0  # Softmax temperature for distillation

losses = []
print("\n    Training...")

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0

    for tokens in train_tokens:
        optimizer.zero_grad()

        # Teacher outputs (no grad)
        with torch.no_grad():
            teacher_outputs = teacher(tokens)
            teacher_logits = teacher_outputs.logits / TEMP
            teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Crystal outputs
        crystal_logits = crystal(tokens) / TEMP
        crystal_log_probs = F.log_softmax(crystal_logits, dim=-1)

        # KL divergence loss (distillation)
        kl_loss = F.kl_div(
            crystal_log_probs.view(-1, crystal_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='batchmean'
        ) * (TEMP ** 2)

        # Also add cross-entropy on actual next tokens
        labels = tokens[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            crystal_logits[:, :-1, :].contiguous().view(-1, crystal_logits.size(-1)),
            labels.view(-1)
        )

        # Combined loss
        loss = 0.5 * kl_loss + 0.5 * ce_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_tokens)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"      Epoch {epoch+1:3d}: loss = {avg_loss:.4f}")

print("\n[7] Testing Crystal GPT-2...")

crystal.eval()
test_prompts = [
    "The meaning of life is",
    "Neural networks learn",
    "The universe began",
]

print("\n    Generation comparison:")
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # Teacher generation
    with torch.no_grad():
        teacher_out = teacher.generate(
            inputs['input_ids'],
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        teacher_text = tokenizer.decode(teacher_out[0])

        # Crystal generation (greedy)
        crystal_ids = inputs['input_ids'].clone()
        for _ in range(10):
            logits = crystal(crystal_ids)
            next_id = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            crystal_ids = torch.cat([crystal_ids, next_id], dim=1)
        crystal_text = tokenizer.decode(crystal_ids[0])

    print(f"\n    Prompt: '{prompt}'")
    print(f"    Teacher: {teacher_text}")
    print(f"    Crystal: {crystal_text}")


print("\n[8] Visualizing Crystal Structure...")

# Get neuron positions and project to 3D for visualization
positions = crystal.get_neuron_positions()
temperatures = crystal.get_temperatures()

# Project 768-dim to 3D via PCA
from sklearn.decomposition import PCA

all_positions = torch.cat(positions, dim=0).numpy()
pca = PCA(n_components=3)
positions_3d = pca.fit_transform(all_positions)

# Create visualization
fig = plt.figure(figsize=(15, 5))

# 3D scatter of neurons
ax1 = fig.add_subplot(131, projection='3d')
colors = []
for i, temp in enumerate(temperatures):
    colors.extend(['red' if t > 0.5 else 'blue' for t in temp])
ax1.scatter(positions_3d[:, 0], positions_3d[:, 1], positions_3d[:, 2],
           c=colors, alpha=0.6, s=50)
ax1.set_title(f'Crystal Neurons (N={len(all_positions)})\nRed=Active, Blue=Stable')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')

# Loss curve
ax2 = fig.add_subplot(132)
ax2.plot(losses)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Distillation Loss')
ax2.grid(True)

# Neuron activity histogram
ax3 = fig.add_subplot(133)
all_temps = torch.cat(temperatures).numpy()
ax3.hist(all_temps, bins=30, edgecolor='black')
ax3.axvline(x=0.5, color='red', linestyle='--', label='Freeze threshold')
ax3.set_xlabel('Temperature (gradient variance)')
ax3.set_ylabel('Count')
ax3.set_title('Neuron Activity Distribution')
ax3.legend()

plt.tight_layout()
plt.savefig('crystal_gpt2_structure.png', dpi=150)
print(f"    Saved: crystal_gpt2_structure.png")

# Summary
print("\n" + "=" * 60)
print("CRYSTALLIZATION COMPLETE!")
print("=" * 60)
print(f"""
    Teacher (GPT-2):      {num_params:,} parameters
    Crystal:              {crystal_params:,} parameters
    Compression:          {num_params/crystal_params:.1f}x

    Architecture:
    - {INITIAL_NEURONS} neurons per block
    - {NUM_BLOCKS} blocks
    - Geometric attention (distance-based)

    This is just the SEED! Next steps:
    1. Train on more data
    2. Let neurons GROW where gradients are high
    3. FREEZE neurons where gradients are low
    4. The crystal will self-optimize!

    "Knowledge crystallizes into geometry."
""")

# Save the crystal
torch.save({
    'model_state_dict': crystal.state_dict(),
    'num_neurons': INITIAL_NEURONS,
    'num_blocks': NUM_BLOCKS,
    'losses': losses,
}, 'crystal_gpt2_seed.pt')
print("    Saved: crystal_gpt2_seed.pt")
