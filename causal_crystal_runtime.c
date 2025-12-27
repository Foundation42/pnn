/*
 * Causal Crystal Runtime v2.0
 *
 * Runtime for .crystal files with causal self-attention support.
 * The breakthrough architecture: tokens can finally see each other!
 *
 * Compile: gcc -O3 -o causal_crystal_runtime causal_crystal_runtime.c -lm
 * Usage: ./causal_crystal_runtime model.crystal "PROMPT:" [max_tokens] [temperature]
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
#define FLAG_FLOAT32 0x0008
#define FLAG_PRUNED_VOCAB 0x0004
#define FLAG_CAUSAL 0x0080

/* Header structure v2 (64 bytes) */
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
    uint16_t num_heads;       /* 2 bytes */
    uint16_t num_norms;       /* 2 bytes */
    uint16_t reserved1;       /* 2 bytes - padding */
    uint16_t reserved2;       /* 2 bytes - padding */
    /* Offsets (8 x 4 = 32 bytes) */
    uint32_t embed_offset;
    uint32_t pos_offset;
    uint32_t causal_offset;
    uint32_t geo_offset;
    uint32_t ffn_offset;
    uint32_t head_offset;
    uint32_t norm_offset;
    uint32_t total_size;
    /* Total: 64 bytes */
} CrystalHeader;

/* Runtime model */
typedef struct {
    CrystalHeader header;
    int has_causal;

    /* Memory mapped file */
    void* mmap_base;
    size_t mmap_size;

    /* Pointers to data sections (float32) */
    float* token_embed;       /* [vocab_size][embed_dim] */
    float* pos_embed;         /* [context_len][embed_dim] */

    /* Causal attention weights */
    float* q_proj_w;          /* [embed_dim][embed_dim] */
    float* q_proj_b;          /* [embed_dim] */
    float* k_proj_w;          /* [embed_dim][embed_dim] */
    float* k_proj_b;          /* [embed_dim] */
    float* v_proj_w;          /* [embed_dim][embed_dim] */
    float* v_proj_b;          /* [embed_dim] */
    float* out_proj_w;        /* [embed_dim][embed_dim] */
    float* out_proj_b;        /* [embed_dim] */

    /* Geometric attention */
    float* geo_positions;     /* [num_neurons][embed_dim] */
    float* geo_values;        /* [num_neurons][embed_dim] */
    float* geo_temps;         /* [num_neurons] */

    /* FFN */
    float* ffn_w1;            /* [hidden_dim][embed_dim] */
    float* ffn_b1;            /* [hidden_dim] */
    float* ffn_w2;            /* [embed_dim][hidden_dim] */
    float* ffn_b2;            /* [embed_dim] */

    /* Output head */
    float* head_w;            /* [vocab_size][embed_dim] */
    float* head_b;            /* [vocab_size] */

    /* Layer norms */
    float* norm_causal_w;     /* [embed_dim] */
    float* norm_causal_b;
    float* norm_geo_w;
    float* norm_geo_b;
    float* norm_ffn_w;
    float* norm_ffn_b;
} CrystalModel;

/* Allocate working memory */
typedef struct {
    float* h;                 /* [context_len][embed_dim] */
    float* q;                 /* [context_len][embed_dim] */
    float* k;                 /* [context_len][embed_dim] */
    float* v;                 /* [context_len][embed_dim] */
    float* attn;              /* [context_len][context_len] */
    float* attn_out;          /* [context_len][embed_dim] */
    float* geo_out;           /* [context_len][embed_dim] */
    float* ffn_hidden;        /* [context_len][hidden_dim] */
    float* logits;            /* [vocab_size] */
    float* temp;              /* [embed_dim] - scratch */
} WorkingMemory;

/* Helper functions */
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void layer_norm(float* out, const float* in, const float* w, const float* b, int dim) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < dim; i++) mean += in[i];
    mean /= dim;
    for (int i = 0; i < dim; i++) var += (in[i] - mean) * (in[i] - mean);
    var = 1.0f / sqrtf(var / dim + 1e-5f);
    for (int i = 0; i < dim; i++) out[i] = (in[i] - mean) * var * w[i] + b[i];
}

static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void matmul(float* out, const float* a, const float* b, int m, int k, int n) {
    /* out[m][n] = a[m][k] @ b[k][n] */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

static void linear(float* out, const float* in, const float* w, const float* b, int in_dim, int out_dim) {
    /* out = in @ w.T + b, where w is [out_dim][in_dim] */
    for (int i = 0; i < out_dim; i++) {
        float sum = b ? b[i] : 0.0f;
        for (int j = 0; j < in_dim; j++) {
            sum += in[j] * w[i * in_dim + j];
        }
        out[i] = sum;
    }
}

/* Causal self-attention */
static void causal_attention(CrystalModel* m, WorkingMemory* mem, int seq_len) {
    int D = m->header.embed_dim;
    int H = m->header.num_heads;
    int head_dim = D / H;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Project Q, K, V for all positions */
    for (int t = 0; t < seq_len; t++) {
        float* h_t = mem->h + t * D;
        float* q_t = mem->q + t * D;
        float* k_t = mem->k + t * D;
        float* v_t = mem->v + t * D;

        linear(q_t, h_t, m->q_proj_w, m->q_proj_b, D, D);
        linear(k_t, h_t, m->k_proj_w, m->k_proj_b, D, D);
        linear(v_t, h_t, m->v_proj_w, m->v_proj_b, D, D);
    }

    /* Compute attention for each head */
    for (int h = 0; h < H; h++) {
        /* Compute attention scores: Q @ K.T with causal mask */
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                if (j > i) {
                    /* Causal mask: can't attend to future */
                    mem->attn[i * seq_len + j] = -1e9f;
                } else {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        int idx = h * head_dim + d;
                        score += mem->q[i * D + idx] * mem->k[j * D + idx];
                    }
                    mem->attn[i * seq_len + j] = score * scale;
                }
            }
            /* Softmax over the row */
            softmax(mem->attn + i * seq_len, seq_len);
        }

        /* Attention output: attn @ V */
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    sum += mem->attn[i * seq_len + j] * mem->v[j * D + h * head_dim + d];
                }
                mem->attn_out[i * D + h * head_dim + d] = sum;
            }
        }
    }

    /* Output projection and residual */
    for (int t = 0; t < seq_len; t++) {
        float* attn_t = mem->attn_out + t * D;
        float* h_t = mem->h + t * D;
        linear(mem->temp, attn_t, m->out_proj_w, m->out_proj_b, D, D);
        for (int i = 0; i < D; i++) {
            h_t[i] += mem->temp[i];  /* Residual connection */
        }
    }
}

/* Geometric attention */
static void geometric_attention(CrystalModel* m, WorkingMemory* mem, int seq_len) {
    int D = m->header.embed_dim;
    int N = m->header.num_neurons;

    for (int t = 0; t < seq_len; t++) {
        float* h_t = mem->h + t * D;
        float* out_t = mem->geo_out + t * D;

        /* Compute distances to all neurons */
        float weights[N];
        float weight_sum = 0.0f;

        for (int n = 0; n < N; n++) {
            float dist = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = h_t[d] - m->geo_positions[n * D + d];
                dist += diff * diff;
            }
            dist = sqrtf(dist);

            /* RBF weight */
            float temp = fabsf(m->geo_temps[n]) + 0.1f;
            weights[n] = expf(-dist / temp);
            weight_sum += weights[n];
        }

        /* Normalize and gather values */
        for (int d = 0; d < D; d++) out_t[d] = 0.0f;
        for (int n = 0; n < N; n++) {
            float w = weights[n] / (weight_sum + 1e-8f);
            for (int d = 0; d < D; d++) {
                out_t[d] += w * m->geo_values[n * D + d];
            }
        }

        /* Residual */
        for (int d = 0; d < D; d++) h_t[d] += out_t[d];
    }
}

/* FFN */
static void ffn_forward(CrystalModel* m, WorkingMemory* mem, int seq_len) {
    int D = m->header.embed_dim;
    int H = m->header.hidden_dim;

    for (int t = 0; t < seq_len; t++) {
        float* h_t = mem->h + t * D;
        float* hidden = mem->ffn_hidden + t * H;

        /* First linear + GELU */
        linear(hidden, h_t, m->ffn_w1, m->ffn_b1, D, H);
        for (int i = 0; i < H; i++) hidden[i] = gelu(hidden[i]);

        /* Second linear + residual */
        linear(mem->temp, hidden, m->ffn_w2, m->ffn_b2, H, D);
        for (int i = 0; i < D; i++) h_t[i] += mem->temp[i];
    }
}

/* Load model from file */
CrystalModel* crystal_load(const char* path) {
    CrystalModel* m = calloc(1, sizeof(CrystalModel));
    if (!m) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) { free(m); return NULL; }

    fseek(f, 0, SEEK_END);
    m->mmap_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    m->mmap_base = malloc(m->mmap_size);
    if (!m->mmap_base) { fclose(f); free(m); return NULL; }

    fread(m->mmap_base, 1, m->mmap_size, f);
    fclose(f);

    /* Parse header */
    memcpy(&m->header, m->mmap_base, sizeof(CrystalHeader));

    if (m->header.magic != CRYSTAL_MAGIC) {
        fprintf(stderr, "Invalid crystal file\n");
        free(m->mmap_base); free(m); return NULL;
    }

    if (m->header.version < 2) {
        fprintf(stderr, "Crystal version %d not supported (need v2+)\n", m->header.version);
        free(m->mmap_base); free(m); return NULL;
    }

    m->has_causal = (m->header.flags & FLAG_CAUSAL) != 0;

    /* Set up pointers */
    uint8_t* base = (uint8_t*)m->mmap_base;
    int D = m->header.embed_dim;
    int V = m->header.vocab_size;
    int T = m->header.context_len;
    int H = m->header.hidden_dim;
    int N = m->header.num_neurons;

    m->token_embed = (float*)(base + m->header.embed_offset);
    m->pos_embed = (float*)(base + m->header.pos_offset);

    if (m->has_causal) {
        float* causal = (float*)(base + m->header.causal_offset);
        m->q_proj_w = causal; causal += D * D;
        m->q_proj_b = causal; causal += D;
        m->k_proj_w = causal; causal += D * D;
        m->k_proj_b = causal; causal += D;
        m->v_proj_w = causal; causal += D * D;
        m->v_proj_b = causal; causal += D;
        m->out_proj_w = causal; causal += D * D;
        m->out_proj_b = causal;
    }

    float* geo = (float*)(base + m->header.geo_offset);
    m->geo_positions = geo; geo += N * D;
    m->geo_values = geo; geo += N * D;
    m->geo_temps = geo;

    float* ffn = (float*)(base + m->header.ffn_offset);
    m->ffn_w1 = ffn; ffn += H * D;
    m->ffn_b1 = ffn; ffn += H;
    m->ffn_w2 = ffn; ffn += D * H;
    m->ffn_b2 = ffn;

    float* head = (float*)(base + m->header.head_offset);
    m->head_w = head; head += V * D;
    m->head_b = head;

    float* norm = (float*)(base + m->header.norm_offset);
    if (m->header.num_norms == 3) {
        m->norm_causal_w = norm; norm += D;
        m->norm_causal_b = norm; norm += D;
        m->norm_geo_w = norm; norm += D;
        m->norm_geo_b = norm; norm += D;
        m->norm_ffn_w = norm; norm += D;
        m->norm_ffn_b = norm;
    } else {
        /* 2 norms - use geo for first, ffn for second */
        m->norm_geo_w = norm; norm += D;
        m->norm_geo_b = norm; norm += D;
        m->norm_ffn_w = norm; norm += D;
        m->norm_ffn_b = norm;
        m->norm_causal_w = m->norm_geo_w;  /* Reuse */
        m->norm_causal_b = m->norm_geo_b;
    }

    printf("Loaded crystal: V=%d D=%d T=%d H=%d N=%d heads=%d causal=%d\n",
           V, D, T, H, N, m->header.num_heads, m->has_causal);

    return m;
}

/* Allocate working memory */
WorkingMemory* alloc_memory(CrystalModel* m) {
    WorkingMemory* mem = calloc(1, sizeof(WorkingMemory));
    int D = m->header.embed_dim;
    int T = m->header.context_len;
    int H = m->header.hidden_dim;
    int V = m->header.vocab_size;

    mem->h = calloc(T * D, sizeof(float));
    mem->q = calloc(T * D, sizeof(float));
    mem->k = calloc(T * D, sizeof(float));
    mem->v = calloc(T * D, sizeof(float));
    mem->attn = calloc(T * T, sizeof(float));
    mem->attn_out = calloc(T * D, sizeof(float));
    mem->geo_out = calloc(T * D, sizeof(float));
    mem->ffn_hidden = calloc(T * H, sizeof(float));
    mem->logits = calloc(V, sizeof(float));
    mem->temp = calloc(D, sizeof(float));

    return mem;
}

/* Generate text */
void generate(CrystalModel* m, WorkingMemory* mem, int* tokens, int num_tokens,
              int max_new, float temperature) {
    int D = m->header.embed_dim;
    int V = m->header.vocab_size;
    int T = m->header.context_len;

    for (int step = 0; step < max_new; step++) {
        int seq_len = num_tokens < T ? num_tokens : T;
        int start = num_tokens > T ? num_tokens - T : 0;

        /* Build input: token_embed + pos_embed */
        for (int t = 0; t < seq_len; t++) {
            int tok = tokens[start + t];
            for (int d = 0; d < D; d++) {
                mem->h[t * D + d] = m->token_embed[tok * D + d] + m->pos_embed[t * D + d];
            }
        }

        /* Forward pass */
        if (m->has_causal) {
            /* Layer norm before causal attention */
            for (int t = 0; t < seq_len; t++) {
                layer_norm(mem->temp, mem->h + t * D, m->norm_causal_w, m->norm_causal_b, D);
                memcpy(mem->h + t * D, mem->temp, D * sizeof(float));
            }
            /* Save for residual, then causal attention */
            float* residual = malloc(seq_len * D * sizeof(float));
            memcpy(residual, mem->h, seq_len * D * sizeof(float));
            causal_attention(m, mem, seq_len);
            /* Residual was added in causal_attention */
            free(residual);
        }

        /* Layer norm before geometric attention */
        for (int t = 0; t < seq_len; t++) {
            layer_norm(mem->temp, mem->h + t * D, m->norm_geo_w, m->norm_geo_b, D);
            memcpy(mem->h + t * D, mem->temp, D * sizeof(float));
        }
        geometric_attention(m, mem, seq_len);

        /* Layer norm before FFN */
        for (int t = 0; t < seq_len; t++) {
            layer_norm(mem->temp, mem->h + t * D, m->norm_ffn_w, m->norm_ffn_b, D);
            memcpy(mem->h + t * D, mem->temp, D * sizeof(float));
        }
        ffn_forward(m, mem, seq_len);

        /* Compute logits for last position */
        float* h_last = mem->h + (seq_len - 1) * D;
        for (int i = 0; i < V; i++) {
            float sum = m->head_b[i];
            for (int d = 0; d < D; d++) {
                sum += h_last[d] * m->head_w[i * D + d];
            }
            mem->logits[i] = sum / temperature;
        }

        /* Sample */
        softmax(mem->logits, V);
        float r = (float)rand() / RAND_MAX;
        float cumsum = 0.0f;
        int next_token = 0;
        for (int i = 0; i < V; i++) {
            cumsum += mem->logits[i];
            if (r < cumsum) { next_token = i; break; }
        }

        tokens[num_tokens++] = next_token;

        /* Print token (simple - just print the ID for now) */
        printf("[%d]", next_token);
        fflush(stdout);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s model.crystal \"PROMPT\" [max_tokens] [temperature]\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    const char* model_path = argv[1];
    const char* prompt = argv[2];
    int max_tokens = argc > 3 ? atoi(argv[3]) : 100;
    float temperature = argc > 4 ? atof(argv[4]) : 0.8f;

    printf("Loading %s...\n", model_path);
    CrystalModel* m = crystal_load(model_path);
    if (!m) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    WorkingMemory* mem = alloc_memory(m);

    /* For now, just use token IDs directly (no tokenizer in C) */
    printf("Note: C runtime doesn't have tokenizer - using token IDs\n");
    printf("Use Python for proper text generation\n");

    /* Simple test: generate from token 0 */
    int tokens[1024] = {0};  /* Start with token 0 */
    int num_tokens = 1;

    printf("Generating %d tokens at temperature %.2f...\n", max_tokens, temperature);
    generate(m, mem, tokens, num_tokens, max_tokens, temperature);

    return 0;
}
