//go:build vulkan && cgo

package vulkan

/*
#include "vk_bridge.h"
*/
import "C"

import (
	_ "embed"
	"fmt"
	"unsafe"
)

// Tw-COOPMAT: tensor-core GEMM driven from GLSL via VK_KHR_cooperative_matrix — the
// mechanism behind llama.cpp-Vulkan's prefill lead on NVIDIA, available to this backend
// with nothing but glslc (no CUDA toolchain). This first slice is a probe: f16×f16→f32
// GEMM, one subgroup per 16×64 C tile, loads straight from global memory. Shared-memory
// tiling (llama.cpp's BM/BN=64) and dequant-in-tile are the follow-up slices.

//go:embed shaders/coopmat_gemm.spv
var coopmatGemmSpv []byte

// HasCoopMat reports whether the active device enabled VK_KHR_cooperative_matrix
// (NVIDIA/AMD/Intel current drivers; false on MoltenVK and pre-1.2 stacks).
func HasCoopMat() bool { return C.vk_coopmat() == 1 }

// CoopmatGemmF16 computes C[M,N]f32 = A[M,K]f16 · B[K,N]f16 on the cooperative-matrix
// pipeline. A and B are raw little-endian IEEE binary16 element slices (len M*K and K*N).
// Probe constraints: M%16==0, K%16==0, N%64==0.
func CoopmatGemmF16(aHalf, bHalf []uint16, c []float32, m, k, n int) error {
	if !HasCoopMat() {
		return fmt.Errorf("vulkan: cooperative matrix not available on this device")
	}
	if m%16 != 0 || k%16 != 0 || n%64 != 0 {
		return fmt.Errorf("vulkan: coopmat probe needs M%%16==0, K%%16==0, N%%64==0 (got %d,%d,%d)", m, k, n)
	}
	if len(aHalf) != m*k || len(bHalf) != k*n || len(c) != m*n {
		return fmt.Errorf("vulkan: coopmat gemm buffer sizes A=%d B=%d C=%d for [%d,%d,%d]", len(aHalf), len(bHalf), len(c), m, k, n)
	}
	rc := C.vk_coopmat_gemm_f16(
		(*C.uint32_t)(unsafe.Pointer(&coopmatGemmSpv[0])), C.int(len(coopmatGemmSpv)),
		unsafe.Pointer(&aHalf[0]), unsafe.Pointer(&bHalf[0]),
		(*C.float)(&c[0]), C.int(m), C.int(k), C.int(n),
	)
	if rc != 0 {
		return fmt.Errorf("vulkan: coopmat gemm failed (code %d)", int(rc))
	}
	return nil
}
