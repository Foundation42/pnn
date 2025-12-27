/*
 * Crystal Model Format v1.0
 *
 * A compact, native model format for crystallized neural networks.
 * Uses quantization + radix tree deduplication for 10-50x compression.
 *
 * File Layout:
 * ┌─────────────────────────────────────┐
 * │ Header (64 bytes)                   │
 * ├─────────────────────────────────────┤
 * │ Quantization Parameters (16 bytes)  │
 * ├─────────────────────────────────────┤
 * │ Pattern Table                       │
 * │ ├─ Num patterns (4 bytes)           │
 * │ ├─ Pattern 0 (embed_dim bytes)      │
 * │ ├─ Pattern 1 ...                    │
 * │ └─ Pattern N-1                      │
 * ├─────────────────────────────────────┤
 * │ Neuron→Pattern Mapping              │
 * │ (num_neurons * 2 bytes)             │
 * ├─────────────────────────────────────┤
 * │ Active Weights (unfrozen neurons)   │
 * │ (num_active * embed_dim bytes)      │
 * ├─────────────────────────────────────┤
 * │ Token Embeddings                    │
 * │ (vocab_size * embed_dim bytes)      │
 * ├─────────────────────────────────────┤
 * │ Position Embeddings                 │
 * │ (context_len * embed_dim bytes)     │
 * ├─────────────────────────────────────┤
 * │ FFN Weights (quantized)             │
 * ├─────────────────────────────────────┤
 * │ Output Head (quantized)             │
 * ├─────────────────────────────────────┤
 * │ Vocabulary (optional, for tokens)   │
 * └─────────────────────────────────────┘
 */

#ifndef CRYSTAL_FORMAT_H
#define CRYSTAL_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Magic bytes: "CRYS" */
#define CRYSTAL_MAGIC 0x53595243

/* Current format version */
#define CRYSTAL_VERSION 1

/* Maximum values */
#define CRYSTAL_MAX_PATTERNS 65535
#define CRYSTAL_MAX_NEURONS 65535
#define CRYSTAL_MAX_VOCAB 65535

/* Header structure (64 bytes, aligned) */
typedef struct __attribute__((packed)) {
    uint32_t magic;           /* 0x53595243 = "CRYS" */
    uint16_t version;         /* Format version */
    uint16_t flags;           /* Feature flags */

    /* Model dimensions */
    uint32_t vocab_size;      /* Vocabulary size */
    uint16_t embed_dim;       /* Embedding dimension */
    uint16_t context_len;     /* Maximum context length */
    uint16_t hidden_dim;      /* FFN hidden dimension */

    /* Neuron counts */
    uint16_t num_neurons;     /* Total neurons */
    uint16_t num_frozen;      /* Frozen neurons */
    uint16_t num_active;      /* Active neurons */

    /* Pattern table info */
    uint16_t num_patterns;    /* Unique weight patterns */

    /* Section offsets (relative to file start) */
    uint32_t quant_offset;    /* Quantization parameters */
    uint32_t pattern_offset;  /* Pattern table */
    uint32_t mapping_offset;  /* Neuron→pattern mapping */
    uint32_t active_offset;   /* Active weights */
    uint32_t embed_offset;    /* Token embeddings */
    uint32_t pos_offset;      /* Position embeddings */
    uint32_t ffn_offset;      /* FFN weights */
    uint32_t head_offset;     /* Output head */
    uint32_t vocab_offset;    /* Vocabulary (optional) */

    /* Checksums */
    uint32_t crc32;           /* CRC32 of entire file */

} CrystalHeader;

/* Quantization parameters (16 bytes) */
typedef struct __attribute__((packed)) {
    float embed_scale;        /* Token embedding scale */
    float embed_offset;       /* Token embedding offset */
    float weight_scale;       /* Weight scale */
    float weight_offset;      /* Weight offset */
} CrystalQuant;

/* Feature flags */
#define CRYSTAL_FLAG_HAS_VOCAB     0x0001  /* File contains vocabulary */
#define CRYSTAL_FLAG_DELTA_ENCODED 0x0002  /* Patterns are delta-encoded */
#define CRYSTAL_FLAG_PRUNED_VOCAB  0x0004  /* Vocabulary is pruned */

/* Pattern table entry (just the raw int8 pattern) */
/* Size: embed_dim bytes each */

/* Neuron mapping entry */
typedef struct __attribute__((packed)) {
    uint16_t pattern_id;      /* Index into pattern table */
} NeuronMapping;

/* Active neuron entry */
typedef struct __attribute__((packed)) {
    uint16_t neuron_id;       /* Original neuron index */
    /* Followed by embed_dim bytes of int8 weights */
} ActiveNeuron;

/* Runtime model structure (loaded in memory) */
typedef struct {
    CrystalHeader header;
    CrystalQuant quant;

    /* Pattern table */
    int8_t* patterns;         /* [num_patterns][embed_dim] */

    /* Neuron mappings */
    uint16_t* frozen_mapping; /* [num_frozen] → pattern_id */
    uint16_t* active_indices; /* [num_active] → original neuron id */
    int8_t* active_weights;   /* [num_active][embed_dim] */

    /* Embeddings */
    int8_t* token_embed;      /* [vocab_size][embed_dim] */
    int8_t* pos_embed;        /* [context_len][embed_dim] */

    /* Neural network layers */
    int8_t* neuron_positions; /* [num_neurons][embed_dim] */
    int8_t* neuron_values;    /* [num_neurons][embed_dim] */
    int8_t* neuron_temps;     /* [num_neurons] */

    /* FFN */
    int8_t* ffn_w1;           /* [embed_dim][hidden_dim] */
    int8_t* ffn_w2;           /* [hidden_dim][embed_dim] */
    int8_t* ffn_b1;           /* [hidden_dim] */
    int8_t* ffn_b2;           /* [embed_dim] */

    /* Output head */
    int8_t* head_weight;      /* [embed_dim][vocab_size] */
    int8_t* head_bias;        /* [vocab_size] */

    /* Layer norms (store as float, small) */
    float* norm1_weight;      /* [embed_dim] */
    float* norm1_bias;        /* [embed_dim] */
    float* norm2_weight;      /* [embed_dim] */
    float* norm2_bias;        /* [embed_dim] */

    /* Memory-mapped file data (optional) */
    void* mmap_base;
    size_t mmap_size;

} CrystalModel;

/* API functions */
CrystalModel* crystal_load(const char* path);
void crystal_free(CrystalModel* model);

/* Inference */
int* crystal_tokenize(CrystalModel* model, const char* text, int* num_tokens);
char* crystal_detokenize(CrystalModel* model, int* tokens, int num_tokens);
char* crystal_generate(CrystalModel* model, const char* prompt, int max_tokens, float temperature);

/* Dequantization helpers */
static inline float dequantize(int8_t val, float scale, float offset) {
    return (float)val * scale + offset;
}

static inline int8_t quantize(float val, float scale, float offset) {
    float scaled = (val - offset) / scale;
    if (scaled < -127.0f) scaled = -127.0f;
    if (scaled > 127.0f) scaled = 127.0f;
    return (int8_t)(scaled + 0.5f);
}

#ifdef __cplusplus
}
#endif

#endif /* CRYSTAL_FORMAT_H */
