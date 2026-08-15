//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// TestPrefillLeaveOneOut decomposes prefill the way TestDecodeLeaveOneOut does decode: build a
// realistic per-layer chain at the true shapes, then time it with one op class removed.
//
// Measured on an M2 Pro (22 layers, dim 2048, hidden 5632, 32 heads over 4 KV heads, M=512):
//
//	full chain                240.54 ms   (2129 tok/s)
//	without matmul             21.49 ms   contribution 219.05 ms (91.1%)
//	without attention         227.16 ms   contribution  13.38 ms ( 5.6%)
//	without elementwise       233.47 ms   contribution   7.07 ms ( 2.9%)
//	without rmsnorm           238.96 ms   contribution   1.58 ms ( 0.7%)
//
// Two things follow, and the second is the important one.
//
//  1. Prefill is 91% matmul. At 992 GFLOP for the chain that is ~4530 GFLOP/s, about 67% of this
//     machine's f32 peak. Attention, which dominated DECODE at 31.5%, is 5.6% here — the two
//     regimes have almost nothing in common and optimising one says little about the other.
//
//  2. THE CHAIN IS AT PARITY. 2129 tok/s against llama.cpp's pp512 of 2143.92 measured in the same
//     session. Our real prefill is 1899.9. So the kernels are NOT the gap: something in the real
//     decoder costs ~12% that this chain does not model. Candidates it deliberately omits are RoPE,
//     the KV cache writes, the LM head, and the 21 Q6_K tensors of a real Q4_K_M file (this chain is
//     uniformly Q4_K).
//
// That reframes the remaining prefill work. Tuning the GEMM further chases 67% -> higher against
// Apple's MPS, while the 12% is our own code and is not yet attributed. Same shape as the decode
// investigation, where the synthetic chain was 4.4x faster than the real path and the difference
// turned out to be measurement error rather than real work — so the first step is to confirm the
// 12% is real before optimising it.
//
// Reported, not asserted on absolute timings; the assertion is only that matmul dominates.
func TestPrefillLeaveOneOut(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const (
		layers = 22
		dim    = 2048
		hidden = 5632
		heads  = 32
		kvh    = 4
		dk     = 64
		M      = 512
	)
	kvDim := kvh * dk
	up := func(k, n int) *ResidentQWeight {
		raw := make([]byte, n*(k/256)*144)
		for i := range raw {
			raw[i] = byte(i*31 + 7)
		}
		rw, err := Backend{}.UploadQuant(raw, 12, n, k)
		if err != nil {
			t.Skip(err)
		}
		return rw.(*ResidentQWeight)
	}
	type layer struct{ wq, wk, wv, wo, wg, wu, wd *ResidentQWeight }
	ls := make([]layer, layers)
	for i := range ls {
		ls[i] = layer{up(dim, dim), up(dim, kvDim), up(dim, kvDim), up(dim, dim),
			up(dim, hidden), up(dim, hidden), up(hidden, dim)}
	}
	defer func() {
		for _, l := range ls {
			for _, w := range []*ResidentQWeight{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Close()
			}
		}
	}()
	nb := func(n int) *DeviceBuffer { b, _ := NewDeviceBufferF32(make([]float32, n)); return b }
	x, xn, q, att, ob := nb(M*dim), nb(M*dim), nb(M*dim), nb(M*dim), nb(M*dim)
	kk, vv := nb(M*kvDim), nb(M*kvDim)
	g, u := nb(M*hidden), nb(M*hidden)
	gam := nb(dim)
	defer func() {
		for _, b := range []*DeviceBuffer{x, xn, q, att, ob, kk, vv, g, u, gam} {
			b.Release()
		}
	}()

	chain := func(r *Recorder, skip string) {
		for _, l := range ls {
			if skip != "rmsnorm" {
				_ = r.RMSNorm(x, gam, xn, M, dim, 1e-5)
			}
			if skip != "matmul" {
				_ = r.QMatMulResident(xn, l.wq, q, M)
				_ = r.QMatMulResident(xn, l.wk, kk, M)
				_ = r.QMatMulResident(xn, l.wv, vv, M)
			}
			if skip != "attention" {
				_ = r.MHAAt(q, kk, vv, att, 0, M, M, dim, heads, kvh, dk, 1, 0, 0.125)
			}
			if skip != "matmul" {
				_ = r.QMatMulResident(att, l.wo, ob, M)
			}
			if skip != "elementwise" {
				_ = r.BinaryN(x, ob, x, 0, M*dim)
			}
			if skip != "rmsnorm" {
				_ = r.RMSNorm(x, gam, xn, M, dim, 1e-5)
			}
			if skip != "matmul" {
				_ = r.QMatMulResident(xn, l.wg, g, M)
				_ = r.QMatMulResident(xn, l.wu, u, M)
			}
			if skip != "elementwise" {
				_ = r.Unary(g, g, 6)
				_ = r.BinaryN(g, u, g, 2, M*hidden)
			}
			if skip != "matmul" {
				_ = r.QMatMulResident(g, l.wd, ob, M)
			}
			if skip != "elementwise" {
				_ = r.BinaryN(x, ob, x, 0, M*dim)
			}
		}
	}
	timeIt := func(skip string) float64 {
		best := 1e18
		for range 7 {
			r, _ := NewRecorder()
			chain(r, skip)
			r.Commit()
			r.Wait()
			if d := LastGPUSeconds(); d < best {
				best = d
			}
			r.Free()
		}
		return best * 1e3
	}
	full := timeIt("")
	fmt.Printf("PLOO full prefill chain (22 layers, M=%d) = %.2f ms -> %.0f tok/s\n", M, full, float64(M)/full*1e3)
	share := map[string]float64{}
	for _, s := range []string{"matmul", "attention", "rmsnorm", "elementwise"} {
		w := timeIt(s)
		share[s] = 100 * (full - w) / full
		fmt.Printf("PLOO without %-12s %7.2f ms   contribution %6.2f ms (%4.1f%%)\n", s, w, full-w, share[s])
	}
	if share["matmul"] < 60 {
		t.Errorf("prefill profile changed shape: matmul %.1f%% — the conclusions recorded in this "+
			"file were drawn from a matmul-dominated profile", share["matmul"])
	}
}
