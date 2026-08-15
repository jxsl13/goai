//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// TestQ4KMatrixUnitHasNoCrossover records a NEGATIVE result: the direct matrix-unit Q4_K matmul
// (qmatmul_q4k_mm, bit-exact but disabled) never beats dequant+GEMM, at any M.
//
// The hypothesis was structural and looked sound: dequant+GEMM pays a fixed [K][N] expansion
// regardless of M, so it should lose at small M where that cost cannot amortize. It does not.
// Measured on an M2 Pro, K=2048 N=5632, interleaved arms:
//
//	M= 16  dqgemm  492.6us  mmunit  640.6us  0.77x
//	M= 32  dqgemm  533.1us  mmunit  727.5us  0.73x
//	M= 48  dqgemm  582.6us  mmunit 1133.1us  0.51x
//	M= 64  dqgemm  740.4us  mmunit 1203.5us  0.62x
//	M=128  dqgemm 1020.9us  mmunit 2222.8us  0.46x
//	M=256  dqgemm 1549.7us  mmunit 4295.3us  0.36x
//
// dqgemm wins even at M=16, so the direction is closed for this kernel design — the gap widens
// with M rather than closing, meaning mmunit's cost is per-row work, not a fixed term.
//
// Two facts worth keeping, because they point at the design that SHOULD win:
//
//  1. Fitting the ends gives dqgemm ~= 450us fixed + 4.3us/row. At M=64 the fixed expansion is
//     61% of the total, and a TinyLlama forward pass runs 154 such matmuls. That is why short
//     prefill (pp64 ~0.46x of llama.cpp) is so much weaker than long prefill (pp1024 ~0.92x).
//  2. mmunit reads ~6.5 MB of quantized weight where dqgemm writes and re-reads ~46 MB of f32 —
//     7x less traffic — and still loses. So it is not bandwidth-bound; it is the dequant-in-inner-
//     loop instruction cost (~240 scalar instructions per matrix instruction), the same
//     scalar-prologue rule that governed the attention kernels.
//
// Neither arm implements the third design: dequantize each K-tile ONCE into threadgroup memory,
// then run simdgroup matrix ops against that tile. That pays dequant once per tile instead of once
// per row (fixing 2) without materializing [K][N] in device memory (fixing 1).
//
// Reported, not asserted on absolute timings; the assertion is only the qualitative ordering.
func TestQ4KMatrixUnitHasNoCrossover(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 5632
	nb := K / 256
	raw := make([]byte, N*nb*144)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	rw, err := Backend{}.UploadQuant(raw, 12, N, K)
	if err != nil {
		t.Skip(err)
	}
	rq := rw.(*ResidentQWeight)
	defer rq.Close()

	for _, M := range []int{16, 32, 48, 64, 96, 128, 256} {
		x, _ := NewDeviceBufferF32(make([]float32, M*K))
		o, _ := NewDeviceBufferF32(make([]float32, M*N))
		res := map[string]float64{}
		for _, arm := range []string{"dqgemm", "mmunit"} {
			SetQ4KDequantGemm(arm == "dqgemm")
			SetQ4KMatrixUnit(arm == "mmunit")
			best := 1e18
			for range 7 {
				r, _ := NewRecorder()
				for range 3 {
					if err := r.QMatMulResident(x, rq, o, M); err != nil {
						t.Fatal(err)
					}
				}
				r.Commit()
				r.Wait()
				if d := LastGPUSeconds() / 3; d < best {
					best = d
				}
				r.Free()
			}
			res[arm] = best
		}
		if res["mmunit"] < res["dqgemm"] {
			t.Errorf("M=%d: mmunit (%.1fus) beat dqgemm (%.1fus) — the negative result above no "+
				"longer holds; re-evaluate whether qmatmul_q4k_mm should be enabled",
				M, res["mmunit"]*1e6, res["dqgemm"]*1e6)
		}
		fmt.Printf("XO M=%4d dqgemm=%8.1fus mmunit=%8.1fus  dq/mm=%.2fx\n",
			M, res["dqgemm"]*1e6, res["mmunit"]*1e6, res["dqgemm"]/res["mmunit"])
		x.Release()
		o.Release()
	}
	SetQ4KDequantGemm(true)
	SetQ4KMatrixUnit(false)
}
