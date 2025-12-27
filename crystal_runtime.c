/*
 * Crystal Runtime v1.0
 *
 * A tiny, universal loader for .crystal model files.
 * Load once, run any crystallized neural network!
 *
 * Compile: gcc -O3 -o crystal_runtime crystal_runtime.c -lm
 * Usage: ./crystal_runtime model.crystal "PROMPT:" [max_tokens] [temperature]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

/* Crystal format constants */
#define CRYSTAL_MAGIC 0x53595243  /* "CRYS" */
#define FLAG_FLOAT32 0x0008       /* Weights stored as float32 */
#define FLAG_INT16 0x0010         /* Weights stored as int16 */
#define FLAG_MIXED 0x0020         /* Mixed: f32 embed + int16 rest */
#define FLAG_MIXED8 0x0040        /* Mixed: f32 embed + int8 rest */

/* Header structure (must match Python compiler) */
typedef struct __attribute__((packed)) {
    uint32_t magic;           /* 4 bytes */
    uint16_t version;         /* 2 bytes */
    uint16_t flags;           /* 2 bytes */
    uint32_t vocab_size;      /* 4 bytes */
    uint16_t embed_dim;       /* 2 bytes */
    uint16_t context_len;     /* 2 bytes */
    uint16_t hidden_dim;      /* 2 bytes */
    uint16_t num_neurons;     /* 2 bytes */
    uint16_t num_frozen;      /* 2 bytes */
    uint16_t num_active;      /* 2 bytes */
    uint16_t num_patterns;    /* 2 bytes */
    uint16_t pattern_dim;     /* 2 bytes */
    uint16_t reserved1;       /* 2 bytes */
    uint16_t reserved2;       /* 2 bytes */
    uint32_t quant_offset;    /* 4 bytes */
    uint32_t pattern_offset;  /* 4 bytes */
    uint32_t mapping_offset;  /* 4 bytes */
    uint32_t active_offset;   /* 4 bytes */
    uint32_t embed_offset;    /* 4 bytes */
    uint32_t pos_offset;      /* 4 bytes */
    uint32_t ffn_offset;      /* 4 bytes */
    uint32_t head_offset;     /* 4 bytes */
    /* Total: 64 bytes */
} CrystalHeader;

/* Quantization parameters (32 bytes) */
typedef struct __attribute__((packed)) {
    float embed_scale;
    float embed_offset;
    float weight_scale;
    float weight_offset;
    float head_scale;
    float head_offset;
    float head_bias_scale;
    float head_bias_offset;
} CrystalQuant;

/* Runtime model */
typedef struct {
    CrystalHeader header;
    CrystalQuant quant;
    int is_float32;             /* Whether weights are float32 (no quantization) */
    int is_int16;               /* Whether weights are int16 */
    int is_mixed;               /* Mixed: f32 embed + int16 rest */
    int is_mixed8;              /* Mixed: f32 embed + int8 rest */

    /* Pointers into mmap'd data (int8 version) */
    int8_t* patterns;           /* [num_patterns][pattern_dim] */
    uint16_t* frozen_mapping;   /* [num_frozen] -> pattern_id */
    int8_t* active_weights;     /* [num_active][pattern_dim] */
    int8_t* token_embed;        /* [vocab_size][embed_dim] */
    int8_t* pos_embed;          /* [context_len][embed_dim] */
    int8_t* temperatures;       /* [num_neurons] */

    /* Int16 pointers (used when is_int16=1) */
    int16_t* patterns_i16;      /* [num_patterns][pattern_dim] */
    int16_t* active_weights_i16;/* [num_active][pattern_dim] */
    int16_t* token_embed_i16;   /* [vocab_size][embed_dim] */
    int16_t* pos_embed_i16;     /* [context_len][embed_dim] */
    int16_t* temperatures_i16;  /* [num_neurons] */
    int16_t* head_weight_i16;   /* [vocab_size][embed_dim] */
    int16_t* head_bias_i16;     /* [vocab_size] */

    /* Float32 pointers (used when is_float32=1) */
    float* patterns_f32;        /* [num_patterns][pattern_dim] */
    float* active_weights_f32;  /* [num_active][pattern_dim] */
    float* token_embed_f32;     /* [vocab_size][embed_dim] */
    float* pos_embed_f32;       /* [context_len][embed_dim] */
    float* temperatures_f32;    /* [num_neurons] */
    float* head_weight_f32;     /* [vocab_size][embed_dim] */
    float* head_bias_f32;       /* [vocab_size] */

    float* ffn_w1;              /* [hidden_dim][embed_dim] - always float32 */
    float* ffn_b1;              /* [hidden_dim] - always float32 */
    float* ffn_w2;              /* [embed_dim][hidden_dim] - always float32 */
    float* ffn_b2;              /* [embed_dim] - always float32 */

    int8_t* head_weight;        /* [vocab_size][embed_dim] */
    int8_t* head_bias;          /* [vocab_size] */

    float* norm1_weight;        /* [embed_dim] */
    float* norm1_bias;          /* [embed_dim] */
    float* norm2_weight;        /* [embed_dim] */
    float* norm2_bias;          /* [embed_dim] */

    /* Memory mapping */
    void* mmap_base;
    size_t mmap_size;

    /* Workspace (allocated) */
    float* hidden;              /* [context_len][embed_dim] */
    float* attn_out;            /* [context_len][embed_dim] */
    float* ffn_hidden;          /* [hidden_dim] */
    float* logits;              /* [vocab_size] */

    /* Dequantized neuron data (computed once) */
    float* neuron_positions;    /* [num_neurons][embed_dim] */
    float* neuron_values;       /* [num_neurons][embed_dim] */
    float* neuron_temps;        /* [num_neurons] */

} CrystalModel;

/* Forward declarations */
CrystalModel* crystal_load(const char* path);
void crystal_free(CrystalModel* model);
void crystal_generate(CrystalModel* model, const char* prompt, int max_tokens, float temperature);

/* Dequantization */
static inline float dequant(int8_t val, float scale, float offset) {
    return (float)val * scale + offset;
}

static inline float dequant16(int16_t val, float scale, float offset) {
    return (float)val * scale + offset;
}

/* GELU activation */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* Layer normalization */
static void layer_norm(float* out, const float* in, const float* weight, const float* bias, int dim) {
    float mean = 0.0f, var = 0.0f;

    for (int i = 0; i < dim; i++) mean += in[i];
    mean /= dim;

    for (int i = 0; i < dim; i++) {
        float d = in[i] - mean;
        var += d * d;
    }
    var = 1.0f / sqrtf(var / dim + 1e-5f);

    for (int i = 0; i < dim; i++) {
        out[i] = (in[i] - mean) * var * weight[i] + bias[i];
    }
}

/* Softmax */
static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* Sample from distribution */
static int sample(const float* probs, int n, float temperature) {
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;

    for (int i = 0; i < n; i++) {
        cumsum += probs[i];
        if (r < cumsum) return i;
    }

    return n - 1;
}

/* Load .crystal file */
CrystalModel* crystal_load(const char* path) {
    CrystalModel* model = calloc(1, sizeof(CrystalModel));
    if (!model) return NULL;

    /* Memory-map the file */
#ifdef _WIN32
    HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) { free(model); return NULL; }

    DWORD size_high;
    DWORD size_low = GetFileSize(file, &size_high);
    model->mmap_size = ((size_t)size_high << 32) | size_low;

    HANDLE mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) { CloseHandle(file); free(model); return NULL; }

    model->mmap_base = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    CloseHandle(file);
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) { free(model); return NULL; }

    struct stat st;
    fstat(fd, &st);
    model->mmap_size = st.st_size;

    model->mmap_base = mmap(NULL, model->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (model->mmap_base == MAP_FAILED) { free(model); return NULL; }
#endif

    uint8_t* base = (uint8_t*)model->mmap_base;

    /* Read header */
    memcpy(&model->header, base, sizeof(CrystalHeader));

    if (model->header.magic != CRYSTAL_MAGIC) {
        fprintf(stderr, "Invalid crystal file (bad magic)\n");
        crystal_free(model);
        return NULL;
    }

    /* Check weight format */
    model->is_float32 = (model->header.flags & FLAG_FLOAT32) != 0;
    model->is_int16 = (model->header.flags & FLAG_INT16) != 0;
    model->is_mixed = (model->header.flags & FLAG_MIXED) != 0;
    model->is_mixed8 = (model->header.flags & FLAG_MIXED8) != 0;

    const char* format_str;
    if (model->is_float32) format_str = "float32";
    else if (model->is_mixed) format_str = "mixed (f32 embed + int16)";
    else if (model->is_mixed8) format_str = "mixed (f32 embed + int8)";
    else if (model->is_int16) format_str = "int16";
    else format_str = "int8";
    printf("Crystal Model v%d (%s)\n", model->header.version, format_str);
    printf("  Vocab: %u, Embed: %u, Context: %u\n",
           model->header.vocab_size, model->header.embed_dim, model->header.context_len);
    printf("  Neurons: %u (%u frozen, %u active)\n",
           model->header.num_neurons, model->header.num_frozen, model->header.num_active);
    printf("  Patterns: %u (dim=%u)\n", model->header.num_patterns, model->header.pattern_dim);

    /* Read quantization parameters */
    memcpy(&model->quant, base + model->header.quant_offset, sizeof(CrystalQuant));

    /* Element size depends on quantization - for neuron weights */
    size_t neuron_elem_size = model->is_float32 ? sizeof(float) :
                              (model->is_int16 || model->is_mixed) ? sizeof(int16_t) : sizeof(int8_t);
    /* Mixed modes use float32 for embeddings */
    size_t embed_elem_size = (model->is_float32 || model->is_mixed || model->is_mixed8) ? sizeof(float) : neuron_elem_size;
    size_t elem_size = neuron_elem_size;  /* For backward compat */

    /* Set up pointers into mmap'd data */
    model->frozen_mapping = (uint16_t*)(base + model->header.mapping_offset);

    /* Neuron patterns - use neuron_elem_size */
    if (model->is_float32) {
        model->patterns_f32 = (float*)(base + model->header.pattern_offset + 4);
        model->active_weights_f32 = (float*)(base + model->header.active_offset + 4);
    } else if (model->is_int16 || model->is_mixed) {
        model->patterns_i16 = (int16_t*)(base + model->header.pattern_offset + 4);
        model->active_weights_i16 = (int16_t*)(base + model->header.active_offset + 4);
    } else {
        /* int8 or mixed8 */
        model->patterns = (int8_t*)(base + model->header.pattern_offset + 4);
        model->active_weights = (int8_t*)(base + model->header.active_offset + 4);
    }

    /* Embeddings - float32 in mixed modes */
    if (model->is_float32 || model->is_mixed || model->is_mixed8) {
        model->token_embed_f32 = (float*)(base + model->header.embed_offset);
        model->pos_embed_f32 = (float*)(base + model->header.pos_offset);
    } else if (model->is_int16) {
        model->token_embed_i16 = (int16_t*)(base + model->header.embed_offset);
        model->pos_embed_i16 = (int16_t*)(base + model->header.pos_offset);
    } else {
        model->token_embed = (int8_t*)(base + model->header.embed_offset);
        model->pos_embed = (int8_t*)(base + model->header.pos_offset);
    }

    /* Compute intermediate offsets from header offsets */
    /* Layout after pos_offset: pos_data | temp_data | ffn_data | head_data | norm_data */
    size_t pos_size = model->header.context_len * model->header.embed_dim * embed_elem_size;
    size_t temp_offset = model->header.pos_offset + pos_size;

    if (model->is_float32) {
        model->temperatures_f32 = (float*)(base + temp_offset);
    } else if (model->is_int16 || model->is_mixed) {
        model->temperatures_i16 = (int16_t*)(base + temp_offset);
    } else {
        /* int8 or mixed8 */
        model->temperatures = (int8_t*)(base + temp_offset);
    }

    /* FFN weights - use ffn_offset from header (always float32) */
    uint8_t* ffn_ptr = base + model->header.ffn_offset;
    size_t w1_size = model->header.hidden_dim * model->header.embed_dim * sizeof(float);
    size_t b1_size = model->header.hidden_dim * sizeof(float);
    size_t w2_size = model->header.embed_dim * model->header.hidden_dim * sizeof(float);
    size_t b2_size = model->header.embed_dim * sizeof(float);

    model->ffn_w1 = (float*)ffn_ptr;
    model->ffn_b1 = (float*)(ffn_ptr + w1_size);
    model->ffn_w2 = (float*)(ffn_ptr + w1_size + b1_size);
    model->ffn_b2 = (float*)(ffn_ptr + w1_size + b1_size + w2_size);

    /* Head weights - use head_offset from header */
    /* In mixed modes, head uses int16/int8 (only embeddings are float32) */
    uint8_t* head_ptr = base + model->header.head_offset;
    size_t head_elem_size = model->is_float32 ? sizeof(float) :
                            (model->is_int16 || model->is_mixed) ? sizeof(int16_t) : sizeof(int8_t);
    size_t head_w_size = model->header.vocab_size * model->header.embed_dim * head_elem_size;
    size_t head_b_size = model->header.vocab_size * head_elem_size;

    if (model->is_float32) {
        model->head_weight_f32 = (float*)head_ptr;
        model->head_bias_f32 = (float*)(head_ptr + head_w_size);
    } else if (model->is_int16 || model->is_mixed) {
        model->head_weight_i16 = (int16_t*)head_ptr;
        model->head_bias_i16 = (int16_t*)(head_ptr + head_w_size);
    } else {
        /* int8 or mixed8 */
        model->head_weight = (int8_t*)head_ptr;
        model->head_bias = (int8_t*)(head_ptr + head_w_size);
    }

    /* Layer norms (always float32) - right after head data */
    uint8_t* norm_ptr = head_ptr + head_w_size + head_b_size;
    model->norm1_weight = (float*)norm_ptr;
    model->norm1_bias = (float*)(norm_ptr + model->header.embed_dim * 4);
    model->norm2_weight = (float*)(norm_ptr + model->header.embed_dim * 8);
    model->norm2_bias = (float*)(norm_ptr + model->header.embed_dim * 12);

    printf("  Offsets: pattern=%u, embed=%u, ffn=%u, head=%u\n",
           model->header.pattern_offset, model->header.embed_offset,
           model->header.ffn_offset, model->header.head_offset);

    /* Allocate workspace */
    int D = model->header.embed_dim;
    int T = model->header.context_len;
    int H = model->header.hidden_dim;
    int V = model->header.vocab_size;
    int N = model->header.num_neurons;

    model->hidden = calloc(T * D, sizeof(float));
    model->attn_out = calloc(T * D, sizeof(float));
    model->ffn_hidden = calloc(H, sizeof(float));
    model->logits = calloc(V, sizeof(float));

    /* Dequantize neuron data */
    model->neuron_positions = calloc(N * D, sizeof(float));
    model->neuron_values = calloc(N * D, sizeof(float));
    model->neuron_temps = calloc(N, sizeof(float));

    int pattern_half = model->header.pattern_dim / 2;  /* positions + values */

    if (model->is_float32) {
        /* Float32 - copy directly without dequantization */
        for (int i = 0; i < model->header.num_frozen; i++) {
            uint16_t pid = model->frozen_mapping[i];
            float* pattern = model->patterns_f32 + pid * model->header.pattern_dim;

            for (int j = 0; j < D; j++) {
                model->neuron_positions[i * D + j] = pattern[j];
                model->neuron_values[i * D + j] = pattern[pattern_half + j];
            }
        }

        for (int i = 0; i < model->header.num_active; i++) {
            int idx = model->header.num_frozen + i;
            float* weights = model->active_weights_f32 + i * model->header.pattern_dim;

            for (int j = 0; j < D; j++) {
                model->neuron_positions[idx * D + j] = weights[j];
                model->neuron_values[idx * D + j] = weights[pattern_half + j];
            }
        }

        for (int i = 0; i < N; i++) {
            model->neuron_temps[i] = model->temperatures_f32[i];
        }
    } else if (model->is_int16 || model->is_mixed) {
        /* Int16 - dequantize frozen neurons from patterns */
        for (int i = 0; i < model->header.num_frozen; i++) {
            uint16_t pid = model->frozen_mapping[i];
            int16_t* pattern = model->patterns_i16 + pid * model->header.pattern_dim;

            for (int j = 0; j < D; j++) {
                model->neuron_positions[i * D + j] = dequant16(pattern[j], model->quant.weight_scale, model->quant.weight_offset);
                model->neuron_values[i * D + j] = dequant16(pattern[pattern_half + j], model->quant.weight_scale, model->quant.weight_offset);
            }
        }

        /* Dequantize active neurons */
        for (int i = 0; i < model->header.num_active; i++) {
            int idx = model->header.num_frozen + i;
            int16_t* weights = model->active_weights_i16 + i * model->header.pattern_dim;

            for (int j = 0; j < D; j++) {
                model->neuron_positions[idx * D + j] = dequant16(weights[j], model->quant.weight_scale, model->quant.weight_offset);
                model->neuron_values[idx * D + j] = dequant16(weights[pattern_half + j], model->quant.weight_scale, model->quant.weight_offset);
            }
        }

        /* Dequantize temperatures */
        for (int i = 0; i < N; i++) {
            model->neuron_temps[i] = dequant16(model->temperatures_i16[i], model->quant.weight_scale, model->quant.weight_offset);
        }
    } else {
        /* Int8 or mixed8 - dequantize frozen neurons from patterns */
        for (int i = 0; i < model->header.num_frozen; i++) {
            uint16_t pid = model->frozen_mapping[i];
            int8_t* pattern = model->patterns + pid * model->header.pattern_dim;

            for (int j = 0; j < D; j++) {
                model->neuron_positions[i * D + j] = dequant(pattern[j], model->quant.weight_scale, model->quant.weight_offset);
                model->neuron_values[i * D + j] = dequant(pattern[pattern_half + j], model->quant.weight_scale, model->quant.weight_offset);
            }
        }

        /* Dequantize active neurons */
        for (int i = 0; i < model->header.num_active; i++) {
            int idx = model->header.num_frozen + i;  /* Active neurons come after frozen */
            int8_t* weights = model->active_weights + i * model->header.pattern_dim;

            for (int j = 0; j < D; j++) {
                model->neuron_positions[idx * D + j] = dequant(weights[j], model->quant.weight_scale, model->quant.weight_offset);
                model->neuron_values[idx * D + j] = dequant(weights[pattern_half + j], model->quant.weight_scale, model->quant.weight_offset);
            }
        }

        /* Dequantize temperatures */
        for (int i = 0; i < N; i++) {
            model->neuron_temps[i] = dequant(model->temperatures[i], model->quant.weight_scale, model->quant.weight_offset);
        }
    }

    printf("  Loaded successfully!\n");
    return model;
}

void crystal_free(CrystalModel* model) {
    if (!model) return;

#ifdef _WIN32
    if (model->mmap_base) UnmapViewOfFile(model->mmap_base);
#else
    if (model->mmap_base && model->mmap_base != MAP_FAILED) {
        munmap(model->mmap_base, model->mmap_size);
    }
#endif

    free(model->hidden);
    free(model->attn_out);
    free(model->ffn_hidden);
    free(model->logits);
    free(model->neuron_positions);
    free(model->neuron_values);
    free(model->neuron_temps);
    free(model);
}

/* Geometric attention forward pass */
static void geometric_attention(CrystalModel* model, float* out, const float* in, int seq_len) {
    int D = model->header.embed_dim;
    int N = model->header.num_neurons;

    /* For each position in sequence */
    for (int t = 0; t < seq_len; t++) {
        const float* x = in + t * D;
        float* y = out + t * D;

        /* Compute distances to all neurons */
        float weights[N];
        float weight_sum = 0.0f;

        for (int n = 0; n < N; n++) {
            float dist = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = x[d] - model->neuron_positions[n * D + d];
                dist += diff * diff;
            }
            dist = sqrtf(dist);

            float temp = fabsf(model->neuron_temps[n]) + 0.1f;
            weights[n] = expf(-dist / temp);
            weight_sum += weights[n];
        }

        /* Normalize and gather values */
        memset(y, 0, D * sizeof(float));
        for (int n = 0; n < N; n++) {
            float w = weights[n] / (weight_sum + 1e-8f);
            for (int d = 0; d < D; d++) {
                y[d] += w * model->neuron_values[n * D + d];
            }
        }
    }
}

/* FFN forward pass - weights are float32, no dequantization needed */
static void ffn_forward(CrystalModel* model, float* out, const float* in) {
    int D = model->header.embed_dim;
    int H = model->header.hidden_dim;

    /* First linear + GELU */
    for (int h = 0; h < H; h++) {
        float sum = model->ffn_b1[h];
        for (int d = 0; d < D; d++) {
            sum += in[d] * model->ffn_w1[h * D + d];
        }
        model->ffn_hidden[h] = gelu(sum);
    }

    /* Second linear */
    for (int d = 0; d < D; d++) {
        float sum = model->ffn_b2[d];
        for (int h = 0; h < H; h++) {
            sum += model->ffn_hidden[h] * model->ffn_w2[d * H + h];
        }
        out[d] = sum;
    }
}

/* Compute logits */
static void compute_logits(CrystalModel* model, const float* hidden) {
    int D = model->header.embed_dim;
    int V = model->header.vocab_size;

    if (model->is_float32) {
        /* Float32 head - direct access */
        for (int v = 0; v < V; v++) {
            float sum = model->head_bias_f32[v];
            for (int d = 0; d < D; d++) {
                sum += hidden[d] * model->head_weight_f32[v * D + d];
            }
            model->logits[v] = sum;
        }
    } else if (model->is_int16 || model->is_mixed) {
        /* Int16 - dequantize (mixed mode uses int16 for head) */
        for (int v = 0; v < V; v++) {
            float sum = dequant16(model->head_bias_i16[v], model->quant.head_bias_scale, model->quant.head_bias_offset);
            for (int d = 0; d < D; d++) {
                float w = dequant16(model->head_weight_i16[v * D + d], model->quant.head_scale, model->quant.head_offset);
                sum += hidden[d] * w;
            }
            model->logits[v] = sum;
        }
    } else {
        /* Int8 or mixed8 - dequantize */
        for (int v = 0; v < V; v++) {
            float sum = dequant(model->head_bias[v], model->quant.head_bias_scale, model->quant.head_bias_offset);
            for (int d = 0; d < D; d++) {
                float w = dequant(model->head_weight[v * D + d], model->quant.head_scale, model->quant.head_offset);
                sum += hidden[d] * w;
            }
            model->logits[v] = sum;
        }
    }
}

/* Simple BPE tokenizer (GPT-2 style) - just use token IDs directly for now */
/* In production, you'd load the vocabulary and do proper BPE encoding */

/* Generate text */
void crystal_generate(CrystalModel* model, const char* prompt, int max_tokens, float temperature) {
    int D = model->header.embed_dim;
    int T = model->header.context_len;
    int V = model->header.vocab_size;

    /* Start with "\n\nFirst" as prompt (Shakespeare character intro) */
    /* Pruned vocab: \n\n=409, "First"=3541 */
    int tokens[T * 2];
    int num_tokens = 2;
    tokens[0] = 409;   /* \n\n */
    tokens[1] = 3541;  /* First */

    /* Debug: print some quant params and first embedding values */
    if (model->is_float32) {
        printf("Float32 mode - no quantization\n");
    } else {
        printf("Quant (%s): embed_scale=%.6f, embed_offset=%.6f\n",
               model->is_int16 ? "int16" : "int8",
               model->quant.embed_scale, model->quant.embed_offset);
        printf("Quant: weight_scale=%.6f, weight_offset=%.6f\n", model->quant.weight_scale, model->quant.weight_offset);
    }

    /* Check first token embedding */
    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        float v;
        if (model->is_float32 || model->is_mixed || model->is_mixed8) {
            v = model->token_embed_f32[0 * D + d];
        } else if (model->is_int16) {
            v = dequant16(model->token_embed_i16[0 * D + d], model->quant.embed_scale, model->quant.embed_offset);
        } else {
            v = dequant(model->token_embed[0 * D + d], model->quant.embed_scale, model->quant.embed_offset);
        }
        sum += v;
    }
    printf("First token embed sum: %.4f\n", sum);

    /* Check first neuron position */
    sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum += model->neuron_positions[0 * D + d];
    }
    printf("First neuron pos sum: %.4f\n", sum);

    printf("Generating %d tokens...\n", max_tokens);

    for (int step = 0; step < max_tokens; step++) {
        int seq_len = num_tokens < T ? num_tokens : T;
        int start = num_tokens > T ? num_tokens - T : 0;

        /* Embed tokens + positions */
        for (int t = 0; t < seq_len; t++) {
            int tok = tokens[start + t];
            for (int d = 0; d < D; d++) {
                float te, pe;
                if (model->is_float32 || model->is_mixed || model->is_mixed8) {
                    te = model->token_embed_f32[tok * D + d];
                    pe = model->pos_embed_f32[t * D + d];
                } else if (model->is_int16) {
                    te = dequant16(model->token_embed_i16[tok * D + d], model->quant.embed_scale, model->quant.embed_offset);
                    pe = dequant16(model->pos_embed_i16[t * D + d], model->quant.embed_scale, model->quant.embed_offset);
                } else {
                    te = dequant(model->token_embed[tok * D + d], model->quant.embed_scale, model->quant.embed_offset);
                    pe = dequant(model->pos_embed[t * D + d], model->quant.embed_scale, model->quant.embed_offset);
                }
                model->hidden[t * D + d] = te + pe;
            }
        }

        /* Layer norm 1 */
        float normed[D];
        layer_norm(normed, model->hidden + (seq_len - 1) * D, model->norm1_weight, model->norm1_bias, D);

        /* Geometric attention */
        float attn_out[D];
        geometric_attention(model, attn_out, normed, 1);

        /* Residual */
        for (int d = 0; d < D; d++) {
            model->hidden[(seq_len - 1) * D + d] += attn_out[d];
        }

        /* Layer norm 2 */
        layer_norm(normed, model->hidden + (seq_len - 1) * D, model->norm2_weight, model->norm2_bias, D);

        /* FFN */
        float ffn_out[D];
        ffn_forward(model, ffn_out, normed);

        /* Residual */
        for (int d = 0; d < D; d++) {
            model->hidden[(seq_len - 1) * D + d] += ffn_out[d];
        }

        /* Compute logits */
        compute_logits(model, model->hidden + (seq_len - 1) * D);

        /* Debug: print top 5 logits BEFORE softmax */
        if (step == 0) {
            printf("\nPre-softmax logits (top 5): ");
            float top_vals[5] = {-1e9, -1e9, -1e9, -1e9, -1e9};
            int top_ids[5] = {0, 0, 0, 0, 0};
            for (int v = 0; v < V; v++) {
                for (int k = 0; k < 5; k++) {
                    if (model->logits[v] > top_vals[k]) {
                        for (int j = 4; j > k; j--) {
                            top_vals[j] = top_vals[j-1];
                            top_ids[j] = top_ids[j-1];
                        }
                        top_vals[k] = model->logits[v];
                        top_ids[k] = v;
                        break;
                    }
                }
            }
            for (int k = 0; k < 5; k++) {
                printf("%d:%.1f ", top_ids[k], top_vals[k]);
            }
            printf("\nLogit range: min=%.1f max=%.1f\n",
                   model->logits[0], top_vals[0]);
        }

        /* Temperature scaling */
        for (int v = 0; v < V; v++) {
            model->logits[v] /= temperature;
        }

        /* Softmax */
        softmax(model->logits, V);

        /* Debug: print top 5 logits on first step */
        if (step == 0) {
            printf("\nTop 5 logits: ");
            float top_vals[5] = {-1e9, -1e9, -1e9, -1e9, -1e9};
            int top_ids[5] = {0, 0, 0, 0, 0};
            for (int v = 0; v < V; v++) {
                for (int k = 0; k < 5; k++) {
                    if (model->logits[v] > top_vals[k]) {
                        for (int j = 4; j > k; j--) {
                            top_vals[j] = top_vals[j-1];
                            top_ids[j] = top_ids[j-1];
                        }
                        top_vals[k] = model->logits[v];
                        top_ids[k] = v;
                        break;
                    }
                }
            }
            for (int k = 0; k < 5; k++) {
                printf("%d:%.2f ", top_ids[k], top_vals[k]);
            }
            printf("\n");
        }

        /* Sample next token */
        int next = sample(model->logits, V, temperature);

        if (num_tokens < T * 2) {  /* Prevent overflow */
            tokens[num_tokens++] = next;
        }

        /* Print token ID (in production, you'd decode to text) */
        printf("%d ", next);
        fflush(stdout);
    }

    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Crystal Runtime v1.0\n");
        printf("Usage: %s model.crystal [max_tokens] [temperature]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int max_tokens = argc > 2 ? atoi(argv[2]) : 50;
    float temperature = argc > 3 ? atof(argv[3]) : 0.8f;

    srand(time(NULL));

    printf("Loading %s...\n", model_path);
    CrystalModel* model = crystal_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    printf("\nGenerating with temperature=%.2f...\n", temperature);
    crystal_generate(model, "", max_tokens, temperature);

    crystal_free(model);
    return 0;
}
