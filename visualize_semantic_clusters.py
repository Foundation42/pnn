"""
Visualize the semantic clusters that neurons have discovered.

Each neuron lives in 64D embedding space - let's see where they've migrated!
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from geometric_lm_words import GeometricWordLM, load_shakespeare_words


def visualize_neurons(model, word_to_idx, idx_to_word, device):
    """Visualize neuron positions and their semantic specializations."""

    model.eval()

    # Get alive neuron positions
    alive_idx = torch.where(model.alive_mask)[0]
    positions = model.positions[alive_idx].detach().cpu().numpy()
    output_weights = model.output_weights[alive_idx].detach().cpu().numpy()

    print(f"Visualizing {len(alive_idx)} neurons")

    # For each neuron, find the top words it predicts
    neuron_top_words = []
    for i in range(len(alive_idx)):
        top_word_ids = np.argsort(output_weights[i])[-5:][::-1]
        top_words = [idx_to_word[j] for j in top_word_ids]
        neuron_top_words.append(top_words)

    # t-SNE to 2D
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(positions)-1), random_state=42)
    positions_2d = tsne.fit_transform(positions)

    # Define semantic categories and their keywords
    categories = {
        'body': ['eye', 'eyes', 'hand', 'hands', 'head', 'heart', 'blood', 'tongue', 'arm', 'arms',
                 'bosom', 'face', 'ear', 'ears', 'spleen', 'breast', 'cheeks', 'lips', 'tears'],
        'royalty': ['king', 'queen', 'crown', 'throne', 'lord', 'duke', 'prince', 'royal', 'majesty'],
        'family': ['son', 'daughter', 'father', 'mother', 'wife', 'husband', 'brother', 'sister', 'child'],
        'emotion': ['love', 'hate', 'fear', 'joy', 'grief', 'rage', 'pride', 'shame', 'pity'],
        'death': ['death', 'dead', 'die', 'grave', 'kill', 'murder', 'blood', 'skull', 'skulls', 'tomb'],
        'nature': ['sun', 'moon', 'stars', 'night', 'day', 'light', 'heaven', 'earth', 'sea', 'wind'],
        'time': ['time', 'day', 'night', 'hour', 'tomorrow', 'yesterday', 'now', 'ever', 'never'],
    }

    # Assign each neuron to a category based on its top words
    neuron_categories = []
    for top_words in neuron_top_words:
        best_cat = 'other'
        best_count = 0
        for cat, keywords in categories.items():
            count = sum(1 for w in top_words if w in keywords)
            if count > best_count:
                best_count = count
                best_cat = cat
        neuron_categories.append(best_cat)

    # Color map
    cat_colors = {
        'body': 'red',
        'royalty': 'gold',
        'family': 'green',
        'emotion': 'pink',
        'death': 'black',
        'nature': 'skyblue',
        'time': 'orange',
        'other': 'gray'
    }

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: All neurons colored by category
    ax1 = axes[0]
    for cat in cat_colors:
        mask = [c == cat for c in neuron_categories]
        if any(mask):
            pts = positions_2d[mask]
            ax1.scatter(pts[:, 0], pts[:, 1], c=cat_colors[cat], label=cat, alpha=0.7, s=50)

    ax1.legend()
    ax1.set_title('Neuron Positions in Semantic Space\n(colored by discovered category)')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')

    # Right: Annotated with top words for some neurons
    ax2 = axes[1]
    ax2.scatter(positions_2d[:, 0], positions_2d[:, 1], c='lightblue', alpha=0.5, s=30)

    # Annotate every 10th neuron with its top word
    for i in range(0, len(positions_2d), max(1, len(positions_2d)//30)):
        top_word = neuron_top_words[i][0]
        if top_word not in ['\n', ',', '.', ':', ';', '?', '!', '<unk>', '<pad>']:
            ax2.annotate(top_word, (positions_2d[i, 0], positions_2d[i, 1]),
                        fontsize=8, alpha=0.8)

    ax2.set_title('Neuron Positions with Top Predicted Words')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig('semantic_clusters.png', dpi=150)
    print("Saved to semantic_clusters.png")
    plt.close()

    # Print cluster statistics
    print("\n=== SEMANTIC CLUSTER ANALYSIS ===")
    for cat in categories:
        count = sum(1 for c in neuron_categories if c == cat)
        if count > 0:
            print(f"\n{cat.upper()} ({count} neurons):")
            # Find neurons in this category and show their top words
            for i, (words, c) in enumerate(zip(neuron_top_words, neuron_categories)):
                if c == cat and i < 100:  # Limit output
                    print(f"  Neuron {i}: {', '.join(words[:3])}")

    other_count = sum(1 for c in neuron_categories if c == 'other')
    print(f"\nOTHER ({other_count} neurons) - unclassified")

    return positions_2d, neuron_categories, neuron_top_words


def main():
    # Load data
    data, word_to_idx, idx_to_word, vocab = load_shakespeare_words()
    vocab_size = len(vocab)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create and load model (we need to train it first or load weights)
    # For now, let's just train briefly to get some structure
    print("Training a quick model to visualize...")

    model = GeometricWordLM(
        vocab_size=vocab_size,
        embed_dim=64,
        context_length=32,
        max_neurons=200,
        n_seeds=30,
        bandwidth=1.0
    ).to(device)

    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    n_batches = len(data) // (128 * 32)

    # Quick training - 200 epochs
    for epoch in range(200):
        model.train()
        total_loss = 0

        for _ in range(n_batches):
            starts = torch.randint(0, len(data) - 33, (128,))
            x = torch.stack([data[s:s+32] for s in starts])
            y = torch.stack([data[s+32] for s in starts])

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Grow
        if (epoch + 1) % 2 == 0 and model.n_alive < 200:
            model.maybe_split(force=True)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss/n_batches:.3f} | Neurons: {model.n_alive}")

    # Visualize
    visualize_neurons(model, word_to_idx, idx_to_word, device)


if __name__ == "__main__":
    main()
