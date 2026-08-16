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
//	M= 16  dqgemm  492.6us  mmunit  640.6us  0.77x   (no longer measured; see the note in the body)
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
//     7x less traffic — and still loses, so it is not bandwidth-bound.
//
// WHY it loses, measured rather than assumed. qmatmul_q4k_mm already stages each dequantized weight
// tile in threadgroup memory and consumes it with simdgroup matrix ops, so "dequantize once per
// tile instead of once per row" is not the missing idea — it is what the kernel does. What it does
// NOT do is share that tile across M: the weight tile is rebuilt once per M-TILE, so its dequant
// work scales with ceil(M/BM) where dqgemm's is O(1) in M. That is exactly the shape of the table
// above, the gap widening from 0.77x to 0.36x as M grows.
//
// Tested directly by doubling the M tile to 64 rows (8 simdgroups, sO aliased onto sX to stay
// inside 32 KB), which halves the number of rebuilds. Bit-exact at every shape, and the prediction
// held at large M — but only there:
//
//	M= 16  BM=32  640.6us   BM=64  874.2us   0.73x (worse)
//	M= 32  BM=32  727.5us   BM=64 1019.8us   0.71x (worse)
//	M=128  BM=32 2222.8us   BM=64 1840.1us   1.21x (better)
//	M=256  BM=32 4295.3us   BM=64 3517.6us   1.22x (better)
//
// Confirmed but insufficient: 1.22x against a 2.75x deficit. A taller tile also wastes work on
// partial tiles (at M=16, 48 of 64 rows are padding), which is why it costs more than it saves
// below M~64. Reverted to BM=32; the kernel stays disabled.
//
// The remaining lever is not this kernel. dqgemm's ~450us fixed term moves ~52 MB (46 MB of f32
// out, 6.5 MB of quant in) which is ~65% of this machine's sustained bandwidth — close enough to
// the ceiling that expanding to f16 instead of f32, halving the write, is the honest next
// candidate for short-prompt prefill.
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

	// The f16 short-prompt path is checked BEFORE both arms in the resident dispatch, and with the
	// weight cache on (the default) its M cap becomes 1<<20 — so it intercepts every shape measured
	// here and NEITHER toggle selects anything. That made this guard vacuous: both arms landed
	// within 1% at every M (the signature of a path compared with itself) and it failed only on
	// thermal noise. Disable it for the duration so the arms are the two paths this test names.
	SetQ4KDequantGemmF16(false)
	defer SetQ4KDequantGemmF16(true)

	// WARM UP BOTH ARMS at the widest shape before timing anything. Each path allocates its scratch
	// and fills the weight cache once, globally — and whichever arm is measured first at the first
	// shape pays all of it. Untrimmed, that put dqgemm at M=16 at 1302us against 586us at M=32,
	// non-monotonic in M and ~2.6x its own recorded 492.6us, which reads as a crossover that is
	// really a cold start. Same rule as FIRST-BENCHMARK-SAMPLE-IS-NOT-COMPARABLE-001, applied to
	// the first SHAPE rather than the first sample.
	{
		wx, _ := NewDeviceBufferF32(make([]float32, 256*K))
		wo, _ := NewDeviceBufferF32(make([]float32, 256*N))
		for _, arm := range []string{"dqgemm", "mmunit"} {
			SetQ4KDequantGemm(arm == "dqgemm")
			SetQ4KMatrixUnit(arm == "mmunit")
			r, _ := NewRecorder()
			if err := r.QMatMulResident(wx, rq, wo, 256); err != nil {
				t.Fatal(err)
			}
			r.Commit()
			r.Wait()
			r.Free()
		}
		wx.Release()
		wo.Release()
	}

	// M=16 IS NOT A VALID COMPARISON POINT and is no longer measured. q4k_dq_gemm_eligible gates on
	// M >= 24, so at M=16 the "dqgemm" arm is not dqgemm at all — it falls through to the plain
	// quantized kernel. That used to be the cooperative kernel and landed near dqgemm's real cost,
	// which hid the mislabelling; once the cooperative kernels became M==1-only the fallback became
	// the scalar kernel and the arm jumped to ~1338us against 569us at M=32. Non-monotonic in M, and
	// read literally it says "mmunit wins at M=16" about a comparison that never involved dqgemm.
	vacuous := 0
	for _, M := range []int{32, 48, 64, 96, 128, 256} {
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
		// 5% margin: the two paths land within a fraction of a percent of each other at some M on a
		// thermally noisy machine, and a 0.26% inversion is not the crossover this guards against.
		if res["mmunit"] < 0.95*res["dqgemm"] {
			t.Errorf("M=%d: mmunit (%.1fus) beat dqgemm (%.1fus) — the negative result above no "+
				"longer holds; re-evaluate whether qmatmul_q4k_mm should be enabled",
				M, res["mmunit"]*1e6, res["dqgemm"]*1e6)
		}
		// NON-VACUITY, keyed to the WIDEST shape. At M=256 the recorded arms are 0.36x apart —
		// mmunit takes ~2.7x as long — so anything close to parity there means the toggles stopped
		// selecting different code, not that the paths converged. Checked at M=256 alone because
		// that is where the true gap is largest and noise is least able to imitate it; a
		// majority-of-shapes rule tolerated the real vacuous case, which cleared 2% at one M on
		// noise alone.
		if M == 256 {
			if d := res["dqgemm"] - res["mmunit"]; d < 0.10*res["dqgemm"] && -d < 0.10*res["dqgemm"] {
				vacuous++
			}
		}
		fmt.Printf("XO M=%4d dqgemm=%8.1fus mmunit=%8.1fus  dq/mm=%.2fx\n",
			M, res["dqgemm"]*1e6, res["mmunit"]*1e6, res["dqgemm"]/res["mmunit"])
		x.Release()
		o.Release()
	}
	SetQ4KDequantGemm(true)
	SetQ4KMatrixUnit(false)
	if vacuous > 0 {
		t.Error("at M=256 the two arms measured within 10% of each other, where the recorded gap is " +
			"2.7x — the toggles no longer select different paths, so this guard proves nothing; " +
			"find what intercepts the resident dispatch ahead of them (the f16 short-prompt path did)")
	}
}
