#include "ggml-quants.h"
#include "ggml-cpu/quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static volatile float sink;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static void * alloc64(size_t n) {
    void * p = NULL;
    if (posix_memalign(&p, 64, n) != 0) abort();
    return p;
}

static void fill(float * x, int n, int seed) {
    for (int i = 0; i < n; ++i) {
        x[i] = sinf((float) (i + seed * 17) * 0.071f) * (1.0f + (float) ((i + seed) % 11));
    }
}

static double bench_quant(const float * x, block_q8_K * q8, int k, int iters) {
    uint64_t start = now_ns();
    for (int i = 0; i < iters; ++i) {
        quantize_row_q8_K(x, q8, k);
    }
    return (double) (now_ns() - start) / iters;
}

static double bench_dot(const block_tq1_0 * w, const block_q8_K * q8, int k, int iters) {
    float sum = 0;
    uint64_t start = now_ns();
    for (int i = 0; i < iters; ++i) {
        ggml_vec_dot_tq1_0_q8_K(k, &sum, 0, w, 0, q8, 0, 1);
        sink = sum;
    }
    return (double) (now_ns() - start) / iters;
}

static double bench_leaf_total(const float * x, const block_tq1_0 * w, block_q8_K * q8, int k, int iters) {
    float sum = 0;
    uint64_t start = now_ns();
    for (int i = 0; i < iters; ++i) {
        quantize_row_q8_K(x, q8, k);
        ggml_vec_dot_tq1_0_q8_K(k, &sum, 0, w, 0, q8, 0, 1);
        sink = sum;
    }
    return (double) (now_ns() - start) / iters;
}

static double bench_n64_total(const float * x, const block_tq1_0 * w, block_q8_K * q8, int k, int iters) {
    const int blocks = k / 256;
    float sum = 0;
    uint64_t start = now_ns();
    for (int i = 0; i < iters; ++i) {
        quantize_row_q8_K(x, q8, k);
        for (int row = 0; row < 64; ++row) {
            ggml_vec_dot_tq1_0_q8_K(k, &sum, 0, w + row * blocks, 0, q8, 0, 1);
            sink = sum;
        }
    }
    return (double) (now_ns() - start) / iters;
}

int main(void) {
    const int k_leaf = 4096;
    const int k_n64 = 1024;
    float * x_leaf = alloc64((size_t) k_leaf * sizeof(float));
    float * x_n64 = alloc64((size_t) k_n64 * sizeof(float));
    float * source = alloc64((size_t) 64 * k_leaf * sizeof(float));
    block_tq1_0 * w_leaf = alloc64((size_t) (k_leaf / 256) * sizeof(block_tq1_0));
    block_tq1_0 * w_n64 = alloc64((size_t) 64 * (k_n64 / 256) * sizeof(block_tq1_0));
    block_q8_K * q8_leaf = alloc64((size_t) (k_leaf / 256) * sizeof(block_q8_K));
    block_q8_K * q8_n64 = alloc64((size_t) (k_n64 / 256) * sizeof(block_q8_K));

    fill(x_leaf, k_leaf, 1);
    fill(x_n64, k_n64, 2);
    fill(source, 64 * k_leaf, 3);
    quantize_row_tq1_0_ref(source, w_leaf, k_leaf);
    for (int row = 0; row < 64; ++row) {
        quantize_row_tq1_0_ref(source + row * k_leaf, w_n64 + row * (k_n64 / 256), k_n64);
    }
    quantize_row_q8_K(x_leaf, q8_leaf, k_leaf);
    quantize_row_q8_K(x_n64, q8_n64, k_n64);

    for (int warm = 0; warm < 1000; ++warm) {
        sink += (float) bench_dot(w_leaf, q8_leaf, k_leaf, 1);
    }

    printf("llama_tq1_q8k_quant_K4096 %.3f ns/op\n", bench_quant(x_leaf, q8_leaf, k_leaf, 200000));
    printf("llama_tq1_dot_only_K4096 %.3f ns/op\n", bench_dot(w_leaf, q8_leaf, k_leaf, 500000));
    printf("llama_tq1_f32_boundary_K4096 %.3f ns/op\n", bench_leaf_total(x_leaf, w_leaf, q8_leaf, k_leaf, 200000));
    printf("llama_tq1_f32_boundary_N64_K1024 %.3f ns/op\n", bench_n64_total(x_n64, w_n64, q8_n64, k_n64, 30000));

    free(q8_n64);
    free(q8_leaf);
    free(w_n64);
    free(w_leaf);
    free(source);
    free(x_n64);
    free(x_leaf);
    return sink == 1234567.0f;
}
