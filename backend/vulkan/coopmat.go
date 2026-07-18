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

//go:embed shaders/coopmat_gemm_tiled.spv
var coopmatGemmTiledSpv []byte

//go:embed shaders/coopmat_gemm_v2.spv
var coopmatGemmV2Spv []byte

//go:embed shaders/coopmat_gemm_v3.spv
var coopmatGemmV3Spv []byte

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

// CoopmatResident holds device-resident A/B/C buffers for the kernel-rate benchmark path:
// upload once, dispatch many, download C on demand — isolates the tensor-core kernel from
// the ~25 MB/call host round-trip the pooled path pays.
type CoopmatResident struct {
	ah, bh, ch unsafe.Pointer
	m, k, n    int
}

// NewCoopmatResident uploads A[M,K]f16 and B[K,N]f16 and allocates a resident C[M,N]f32.
func NewCoopmatResident(aHalf, bHalf []uint16, m, k, n int) (*CoopmatResident, error) {
	if !HasCoopMat() {
		return nil, fmt.Errorf("vulkan: cooperative matrix not available")
	}
	if m%16 != 0 || k%16 != 0 || n%64 != 0 {
		return nil, fmt.Errorf("vulkan: coopmat probe needs M%%16==0, K%%16==0, N%%64==0")
	}
	r := &CoopmatResident{m: m, k: k, n: n}
	r.ah = unsafe.Pointer(C.vk_devbuf_upload(unsafe.Pointer(&aHalf[0]), C.int(m*k*2)))
	r.bh = unsafe.Pointer(C.vk_devbuf_upload(unsafe.Pointer(&bHalf[0]), C.int(k*n*2)))
	zero := make([]float32, m*n)
	r.ch = unsafe.Pointer(C.vk_devbuf_upload(unsafe.Pointer(&zero[0]), C.int(m*n*4)))
	if r.ah == nil || r.bh == nil || r.ch == nil {
		r.Free()
		return nil, fmt.Errorf("vulkan: coopmat resident upload failed")
	}
	return r, nil
}

// Gemm dispatches one resident GEMM (device buffers only — no host transfer).
func (r *CoopmatResident) Gemm() error {
	rc := C.vk_coopmat_gemm_f16_res(
		(*C.uint32_t)(unsafe.Pointer(&coopmatGemmSpv[0])), C.int(len(coopmatGemmSpv)),
		r.ah, r.bh, r.ch, C.int(r.m), C.int(r.k), C.int(r.n),
	)
	if rc != 0 {
		return fmt.Errorf("vulkan: resident coopmat gemm failed (code %d)", int(rc))
	}
	return nil
}

// GemmTiled dispatches the slice-4 tiled kernel (BM=BN=64, shared-staged, 4x B reuse).
// Needs M%64==0 in addition to the probe constraints.
func (r *CoopmatResident) GemmTiled() error {
	if r.m%64 != 0 {
		return fmt.Errorf("vulkan: tiled coopmat needs M%%64==0 (got %d)", r.m)
	}
	rc := C.vk_coopmat_gemm_f16_res_tiled(
		(*C.uint32_t)(unsafe.Pointer(&coopmatGemmTiledSpv[0])), C.int(len(coopmatGemmTiledSpv)),
		r.ah, r.bh, r.ch, C.int(r.m), C.int(r.k), C.int(r.n),
	)
	if rc != 0 {
		return fmt.Errorf("vulkan: tiled coopmat gemm failed (code %d)", int(rc))
	}
	return nil
}

// GemmV2 dispatches the 32x64-per-subgroup variant (8 accumulators, 2x A reuse).
// Needs M%32==0 in addition to the probe constraints.
func (r *CoopmatResident) GemmV2() error {
	if r.m%32 != 0 {
		return fmt.Errorf("vulkan: coopmat v2 needs M%%32==0 (got %d)", r.m)
	}
	rc := C.vk_coopmat_gemm_f16_res_v2(
		(*C.uint32_t)(unsafe.Pointer(&coopmatGemmV2Spv[0])), C.int(len(coopmatGemmV2Spv)),
		r.ah, r.bh, r.ch, C.int(r.m), C.int(r.k), C.int(r.n),
	)
	if rc != 0 {
		return fmt.Errorf("vulkan: coopmat v2 gemm failed (code %d)", int(rc))
	}
	return nil
}

// GemmV3 dispatches the 64x64-per-subgroup variant (16 accumulators, 4x A and B reuse).
// Needs M%64==0.
func (r *CoopmatResident) GemmV3() error {
	if r.m%64 != 0 {
		return fmt.Errorf("vulkan: coopmat v3 needs M%%64==0 (got %d)", r.m)
	}
	rc := C.vk_coopmat_gemm_f16_res_v3(
		(*C.uint32_t)(unsafe.Pointer(&coopmatGemmV3Spv[0])), C.int(len(coopmatGemmV3Spv)),
		r.ah, r.bh, r.ch, C.int(r.m), C.int(r.k), C.int(r.n),
	)
	if rc != 0 {
		return fmt.Errorf("vulkan: coopmat v3 gemm failed (code %d)", int(rc))
	}
	return nil
}

// ReadC downloads the resident C into dst (len M*N).
func (r *CoopmatResident) ReadC(dst []float32) error {
	if len(dst) != r.m*r.n {
		return fmt.Errorf("vulkan: ReadC needs len %d", r.m*r.n)
	}
	if C.vk_devbuf_download(r.ch, unsafe.Pointer(&dst[0]), C.int(r.m*r.n*4)) != 0 {
		return fmt.Errorf("vulkan: ReadC download failed")
	}
	return nil
}

// Free releases the resident buffers.
func (r *CoopmatResident) Free() {
	for _, h := range []unsafe.Pointer{r.ah, r.bh, r.ch} {
		if h != nil {
			C.vk_devbuf_free(h)
		}
	}
	r.ah, r.bh, r.ch = nil, nil, nil
}
