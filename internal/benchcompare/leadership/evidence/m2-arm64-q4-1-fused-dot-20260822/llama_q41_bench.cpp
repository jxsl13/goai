#include "quants.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kWidth = 4096;
constexpr int kWarmup = 2000;
constexpr int kIterations = 200000;

template <typename F>
double nanoseconds_per_call(F &&fn) {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        fn();
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(end - begin).count() /
           static_cast<double>(kIterations);
}

}  // namespace

int main() {
    static_assert(kWidth % QK4_1 == 0);
    static_assert(kWidth % QK8_1 == 0);

    std::vector<float> weights(kWidth);
    std::vector<float> activations(kWidth);
    for (int i = 0; i < kWidth; ++i) {
        const float fi = static_cast<float>(i);
        weights[i] = 0.45f * std::sin(fi * 0.013f) +
                     0.31f * std::cos(fi * 0.031f) +
                     static_cast<float>((i % 17) - 8) * 0.007f;
        activations[i] = 0.72f * std::sin(fi * 0.019f) -
                         0.27f * std::cos(fi * 0.043f) +
                         static_cast<float>((i % 11) - 5) * 0.011f;
    }

    std::vector<block_q4_1> q4(kWidth / QK4_1);
    std::vector<block_q8_1> q8(kWidth / QK8_1);
    quantize_row_q4_1(weights.data(), q4.data(), kWidth);
    quantize_row_q8_1(activations.data(), q8.data(), kWidth);

    float result = 0.0f;
    for (int i = 0; i < kWarmup; ++i) {
        ggml_vec_dot_q4_1_q8_1(kWidth, &result, 0, q4.data(), 0,
                               q8.data(), 0, 1);
    }

    volatile float checksum = result;
    const auto run_dot = [&] {
        return nanoseconds_per_call([&] {
            ggml_vec_dot_q4_1_q8_1(kWidth, &result, 0, q4.data(), 0,
                                   q8.data(), 0, 1);
            checksum = checksum + result * 1.0e-30f;
        });
    };
    const auto run_quantize_dot = [&] {
        return nanoseconds_per_call([&] {
            quantize_row_q8_1(activations.data(), q8.data(), kWidth);
            ggml_vec_dot_q4_1_q8_1(kWidth, &result, 0, q4.data(), 0,
                                   q8.data(), 0, 1);
            checksum = checksum + result * 1.0e-30f;
        });
    };

    double dot_ns = 0.0;
    double quantize_dot_ns = 0.0;
    if (std::getenv("LLAMA_Q41_QUANT_FIRST") != nullptr) {
        quantize_dot_ns = run_quantize_dot();
        dot_ns = run_dot();
    } else {
        dot_ns = run_dot();
        quantize_dot_ns = run_quantize_dot();
    }

    if (!std::isfinite(checksum) || checksum == 0.0f) {
        std::fprintf(stderr, "invalid checksum: %.9g\n", checksum);
        return EXIT_FAILURE;
    }

    std::printf("commit=3af988fabcf79fd81f8720505e684d2aa5bfc786 "
                "k=%d iterations=%d dot_q4_1_q8_1_ns=%.3f "
                "quantize_q8_1_plus_dot_ns=%.3f checksum=%.9g\n",
                kWidth, kIterations, dot_ns, quantize_dot_ns, checksum);
    return EXIT_SUCCESS;
}
