// Vulkan compute bridge for the optional vulkan backend (§T43). Compiled only
// under `-tags vulkan` with cgo. Synchronous API: each call uploads inputs, runs
// the compute shader, and vkQueueWaitIdle before reading the result back
// (async/device-resident tensors are a later optimization; §V14 keeps the
// interface stable so that lands without an API break).
#ifndef GOAI_VK_BRIDGE_H
#define GOAI_VK_BRIDGE_H

#include <stdint.h>

// vk_available returns 1 if a Vulkan instance with a compute-capable physical
// device is present.
int vk_available(void);

// vk_matmul_f32 computes C[M,N] = A[M,K]·B[K,N], all row-major float32, using the
// SPIR-V compute module in `spv` (spvLen bytes). Returns 0 on success, nonzero on
// failure (see vk_bridge.c for codes).
int vk_matmul_f32(const uint32_t* spv, int spvLen,
                  const float* A, const float* B, float* C,
                  int M, int K, int N);

#endif
