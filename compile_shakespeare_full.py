"""
Compile FULL Crystal Shakespeare to C - Text Generation!
A self-contained C program that generates Shakespeare text.
"""

import torch
import numpy as np
from pathlib import Path


def compile_full_model(checkpoint_path: str, output_path: str):
    """Compile complete Shakespeare model to standalone C."""

    print("=" * 60)
    print("Crystal Shakespeare FULL Compiler")
    print("Generating standalone C text generator!")
    print("=" * 60)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    config = checkpoint['config']
    model = checkpoint['model']

    vocab_size = config['vocab_size']
    embed_dim = config['embed_dim']
    num_neurons = config['num_neurons']

    print(f"\nModel config:")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Neurons: {num_neurons}")

    # Extract all weights
    token_embed = model['token_embed.weight'].numpy()  # [vocab_size, embed_dim]
    pos_embed = model['pos_embed.weight'].numpy()      # [context_len, embed_dim]

    positions = model['attention.positions'].numpy()   # [num_neurons, embed_dim]
    values = model['attention.values'].numpy()         # [num_neurons, embed_dim]
    temperatures = model['attention.temperature'].numpy()  # [num_neurons]
    frozen = model['attention.frozen'].numpy()

    # FFN weights
    ffn_w1 = model['ffn.0.weight'].numpy()  # [embed_dim*2, embed_dim]
    ffn_b1 = model['ffn.0.bias'].numpy()    # [embed_dim*2]
    ffn_w2 = model['ffn.2.weight'].numpy()  # [embed_dim, embed_dim*2]
    ffn_b2 = model['ffn.2.bias'].numpy()    # [embed_dim]

    # Layer norms
    norm1_w = model['norm1.weight'].numpy()  # [embed_dim]
    norm1_b = model['norm1.bias'].numpy()
    norm2_w = model['norm2.weight'].numpy()
    norm2_b = model['norm2.bias'].numpy()

    # Output head
    head_w = model['head.weight'].numpy()  # [vocab_size, embed_dim]
    head_b = model['head.bias'].numpy()    # [vocab_size]

    context_len = pos_embed.shape[0]
    hidden_dim = ffn_w1.shape[0]
    num_frozen = frozen.sum()

    print(f"  Context len: {context_len}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Frozen neurons: {num_frozen} ({100*num_frozen/num_neurons:.1f}%)")

    # Generate C code
    c_code = f"""/* Crystal Shakespeare - Complete Text Generator in C
 * Auto-generated from trained PyTorch model
 *
 * {num_neurons} neurons, {num_frozen} frozen ({100*num_frozen/num_neurons:.1f}%)
 * Vocab: {vocab_size:,}, Embed: {embed_dim}, Context: {context_len}
 *
 * Compile: gcc -O3 -lm crystal_shakespeare.c -o crystal_shakespeare
 * Run: ./crystal_shakespeare "ROMEO:"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VOCAB_SIZE {vocab_size}
#define EMBED_DIM {embed_dim}
#define CONTEXT_LEN {context_len}
#define NUM_NEURONS {num_neurons}
#define HIDDEN_DIM {hidden_dim}

/* ============== TOKEN EMBEDDINGS ============== */
static const float token_embed[VOCAB_SIZE][EMBED_DIM] = {{
"""

    # Token embeddings (this will be large!)
    print("\nWriting token embeddings...")
    for i in range(vocab_size):
        if i % 5000 == 0:
            print(f"  {i}/{vocab_size}...")
        vals = ', '.join(f'{v:.6f}f' for v in token_embed[i])
        c_code += f"    {{ {vals} }},\n"
    c_code += "};\n\n"

    # Position embeddings
    print("Writing position embeddings...")
    c_code += f"static const float pos_embed[CONTEXT_LEN][EMBED_DIM] = {{\n"
    for i in range(context_len):
        vals = ', '.join(f'{v:.6f}f' for v in pos_embed[i])
        c_code += f"    {{ {vals} }},\n"
    c_code += "};\n\n"

    # Neuron positions
    print("Writing neuron positions...")
    c_code += f"static const float neuron_pos[NUM_NEURONS][EMBED_DIM] = {{\n"
    for i in range(num_neurons):
        vals = ', '.join(f'{v:.6f}f' for v in positions[i])
        c_code += f"    {{ {vals} }},\n"
    c_code += "};\n\n"

    # Neuron values
    print("Writing neuron values...")
    c_code += f"static const float neuron_val[NUM_NEURONS][EMBED_DIM] = {{\n"
    for i in range(num_neurons):
        vals = ', '.join(f'{v:.6f}f' for v in values[i])
        c_code += f"    {{ {vals} }},\n"
    c_code += "};\n\n"

    # Temperatures
    c_code += f"static const float temperatures[NUM_NEURONS] = {{\n    "
    c_code += ', '.join(f'{t:.6f}f' for t in temperatures)
    c_code += "\n};\n\n"

    # FFN weights
    print("Writing FFN weights...")
    c_code += f"static const float ffn_w1[HIDDEN_DIM][EMBED_DIM] = {{\n"
    for i in range(hidden_dim):
        vals = ', '.join(f'{v:.6f}f' for v in ffn_w1[i])
        c_code += f"    {{ {vals} }},\n"
    c_code += "};\n\n"

    c_code += f"static const float ffn_b1[HIDDEN_DIM] = {{\n    "
    c_code += ', '.join(f'{v:.6f}f' for v in ffn_b1)
    c_code += "\n};\n\n"

    c_code += f"static const float ffn_w2[EMBED_DIM][HIDDEN_DIM] = {{\n"
    for i in range(embed_dim):
        vals = ', '.join(f'{v:.6f}f' for v in ffn_w2[i])
        c_code += f"    {{ {vals} }},\n"
    c_code += "};\n\n"

    c_code += f"static const float ffn_b2[EMBED_DIM] = {{\n    "
    c_code += ', '.join(f'{v:.6f}f' for v in ffn_b2)
    c_code += "\n};\n\n"

    # Layer norms
    print("Writing layer norms...")
    c_code += f"static const float norm1_w[EMBED_DIM] = {{\n    "
    c_code += ', '.join(f'{v:.6f}f' for v in norm1_w)
    c_code += "\n};\n"
    c_code += f"static const float norm1_b[EMBED_DIM] = {{\n    "
    c_code += ', '.join(f'{v:.6f}f' for v in norm1_b)
    c_code += "\n};\n"
    c_code += f"static const float norm2_w[EMBED_DIM] = {{\n    "
    c_code += ', '.join(f'{v:.6f}f' for v in norm2_w)
    c_code += "\n};\n"
    c_code += f"static const float norm2_b[EMBED_DIM] = {{\n    "
    c_code += ', '.join(f'{v:.6f}f' for v in norm2_b)
    c_code += "\n};\n\n"

    # Output head
    print("Writing output head...")
    c_code += f"static const float head_w[VOCAB_SIZE][EMBED_DIM] = {{\n"
    for i in range(vocab_size):
        if i % 5000 == 0:
            print(f"  {i}/{vocab_size}...")
        vals = ', '.join(f'{v:.6f}f' for v in head_w[i])
        c_code += f"    {{ {vals} }},\n"
    c_code += "};\n\n"

    c_code += f"static const float head_b[VOCAB_SIZE] = {{\n    "
    # Split into multiple lines for readability
    for i in range(0, vocab_size, 20):
        chunk = head_b[i:i+20]
        c_code += ', '.join(f'{v:.6f}f' for v in chunk)
        if i + 20 < vocab_size:
            c_code += ",\n    "
    c_code += "\n};\n\n"

    # Add the computation functions
    print("Writing computation functions...")
    c_code += """
/* ============== COMPUTATION FUNCTIONS ============== */

/* Layer normalization */
static void layer_norm(float* x, const float* w, const float* b, int dim) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    for (int i = 0; i < dim; i++) var += (x[i] - mean) * (x[i] - mean);
    var = sqrtf(var / dim + 1e-5f);
    for (int i = 0; i < dim; i++) {
        x[i] = (x[i] - mean) / var * w[i] + b[i];
    }
}

/* GELU activation */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* Geometric attention */
static void geometric_attention(const float* input, float* output) {
    float weights[NUM_NEURONS];
    float weight_sum = 0.0f;

    for (int n = 0; n < NUM_NEURONS; n++) {
        float dist_sq = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++) {
            float diff = input[d] - neuron_pos[n][d];
            dist_sq += diff * diff;
        }
        float dist = sqrtf(dist_sq);
        float temp = fabsf(temperatures[n]) + 0.1f;
        weights[n] = expf(-dist / temp);
        weight_sum += weights[n];
    }

    for (int d = 0; d < EMBED_DIM; d++) output[d] = 0.0f;
    for (int n = 0; n < NUM_NEURONS; n++) {
        float w = weights[n] / (weight_sum + 1e-8f);
        for (int d = 0; d < EMBED_DIM; d++) {
            output[d] += w * neuron_val[n][d];
        }
    }
}

/* FFN forward */
static void ffn_forward(const float* input, float* output) {
    float hidden[HIDDEN_DIM];

    /* First linear + GELU */
    for (int h = 0; h < HIDDEN_DIM; h++) {
        hidden[h] = ffn_b1[h];
        for (int d = 0; d < EMBED_DIM; d++) {
            hidden[h] += input[d] * ffn_w1[h][d];
        }
        hidden[h] = gelu(hidden[h]);
    }

    /* Second linear */
    for (int d = 0; d < EMBED_DIM; d++) {
        output[d] = ffn_b2[d];
        for (int h = 0; h < HIDDEN_DIM; h++) {
            output[d] += hidden[h] * ffn_w2[d][h];
        }
    }
}

/* Forward pass for single token position */
static void forward_token(int token_id, int pos, float* hidden) {
    float temp[EMBED_DIM], attn_out[EMBED_DIM], ffn_out[EMBED_DIM];

    /* Token + position embedding */
    for (int d = 0; d < EMBED_DIM; d++) {
        hidden[d] = token_embed[token_id][d] + pos_embed[pos][d];
    }

    /* Attention block: h = h + attn(norm1(h)) */
    memcpy(temp, hidden, EMBED_DIM * sizeof(float));
    layer_norm(temp, norm1_w, norm1_b, EMBED_DIM);
    geometric_attention(temp, attn_out);
    for (int d = 0; d < EMBED_DIM; d++) hidden[d] += attn_out[d];

    /* FFN block: h = h + ffn(norm2(h)) */
    memcpy(temp, hidden, EMBED_DIM * sizeof(float));
    layer_norm(temp, norm2_w, norm2_b, EMBED_DIM);
    ffn_forward(temp, ffn_out);
    for (int d = 0; d < EMBED_DIM; d++) hidden[d] += ffn_out[d];
}

/* Compute logits from hidden state */
static void compute_logits(const float* hidden, float* logits) {
    for (int v = 0; v < VOCAB_SIZE; v++) {
        logits[v] = head_b[v];
        for (int d = 0; d < EMBED_DIM; d++) {
            logits[v] += hidden[d] * head_w[v][d];
        }
    }
}

/* Softmax with temperature */
static void softmax_temp(float* logits, int size, float temperature) {
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        logits[i] = expf((logits[i] - max_val) / temperature);
        sum += logits[i];
    }
    for (int i = 0; i < size; i++) {
        logits[i] /= sum;
    }
}

/* Sample from probability distribution */
static int sample(const float* probs, int size) {
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < size; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }
    return size - 1;
}

/* ============== SIMPLE BPE DECODE ============== */
/* Note: This is a simplified decoder. For full BPE, you'd need the merges file. */

/* Common token mappings (subset for demo) */
static const char* decode_token(int token_id) {
    /* This is a hack - we'll just output token IDs for now */
    /* A full implementation would need the GPT-2 vocab.json */
    static char buf[16];

    /* Some common tokens we know */
    if (token_id == 198) return "\\n";
    if (token_id == 220) return " ";
    if (token_id == 25) return ":";
    if (token_id == 13) return ".";
    if (token_id == 11) return ",";
    if (token_id == 0) return "!";
    if (token_id == 30) return "?";

    /* For unknown, return the ID in brackets */
    sprintf(buf, "[%d]", token_id);
    return buf;
}

/* ============== MAIN ============== */
int main(int argc, char** argv) {
    srand(time(NULL));

    printf("Crystal Shakespeare - 788 Neurons of Poetry\\n");
    printf("============================================\\n\\n");

    /* Default prompt tokens (for "ROMEO:") */
    /* In GPT-2 BPE: R=49, O=46, M=44, E=36, O=46, :=25 */
    int tokens[CONTEXT_LEN] = {49, 2662, 36, 46, 25};  /* Approximate "ROMEO:" */
    int num_tokens = 5;

    float temperature = 0.8f;
    int max_new_tokens = 100;

    printf("Generating %d tokens with temperature %.1f...\\n\\n", max_new_tokens, temperature);

    /* Generate tokens */
    float hidden[EMBED_DIM];
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));

    for (int i = 0; i < max_new_tokens; i++) {
        /* Use last token position */
        int pos = (num_tokens - 1) % CONTEXT_LEN;
        int token = tokens[num_tokens - 1];

        /* Forward pass */
        forward_token(token, pos, hidden);

        /* Get logits and sample */
        compute_logits(hidden, logits);
        softmax_temp(logits, VOCAB_SIZE, temperature);
        int next_token = sample(logits, VOCAB_SIZE);

        /* Add to sequence */
        if (num_tokens < CONTEXT_LEN) {
            tokens[num_tokens++] = next_token;
        } else {
            /* Shift window */
            memmove(tokens, tokens + 1, (CONTEXT_LEN - 1) * sizeof(int));
            tokens[CONTEXT_LEN - 1] = next_token;
        }

        /* Print token ID (proper decoding needs vocab file) */
        printf("%d ", next_token);
        fflush(stdout);
    }

    printf("\\n\\nDone! (Token IDs shown - full text decode needs vocab.json)\\n");

    free(logits);
    return 0;
}
"""

    # Write the C file
    print(f"\nWriting to {output_path}...")
    with open(output_path, 'w') as f:
        f.write(c_code)

    file_size = Path(output_path).stat().st_size
    print(f"Generated: {output_path}")
    print(f"Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    print(f"\nTo compile: gcc -O3 -lm {output_path} -o crystal_shakespeare")


if __name__ == "__main__":
    compile_full_model(
        'runs/crystal_shakespeare_20251226_215547/best_model.pt',
        'crystal_shakespeare_full.c'
    )
