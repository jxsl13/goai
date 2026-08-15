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
//  2. THE CHAIN MODELS THE REAL PATH ALMOST EXACTLY. Chain GPU time 240.54 ms against the real
//     decoder's measured GPU-busy of 241.86 ms for the same n=512 prefill — 0.5% apart. There is no
//     unmodelled decoder work: prefill IS these kernels.
//
//     I first read this as a ~12% gap, from two stacked artifacts: comparing the chain's GPU-ONLY
//     time against the real path's WALL time, and pairing numbers taken in different thermal
//     windows (our pp512 reads 1899.9 warm and 2046.8 cool). Both are this session's recurring
//     errors, and the second is why the 0.89x recorded earlier was pessimistic.
//
// Corrected, both sides measured in one window:
//
//	real prefill n=512   wall 250.14 ms = GPU 241.86 ms + 8.28 ms host (96.7% busy)
//	pp512   goai 2046.8  llama.cpp 2216.63 +/- 3.25   0.92x
//	pp1024  goai 1959.4  llama.cpp 2145.26 +/- 0.68   0.91x
//
// So the 8% deficit at pp512 splits into ~4.7% GPU (241.86 vs llama.cpp's implied 231.0 ms) and
// ~3.3% host (the 8.28 ms outside GPU execution).
//
// CORRECTION: "host share" is the wrong name for this quantity, and "~7.4 ms of fixed submission
// latency" below is wrong. LastGPUSeconds reports only the LAST command buffer, so wall minus that
// absorbs any EARLIER command buffers' GPU time as well. Measured across sizes it scales with n
// rather than being fixed — 2.07 / 3.01 / 4.55 / 8.28 / 14.33 ms at n=64/128/256/512/1024, i.e.
// ~1.4 ms fixed + ~12.6us/token — which submission latency would not do, and which the eliminated
// host work (gather at 1.29us/token, upload at 0.12 ms) cannot account for either.
//
// At n=64 the split is 39.34 ms "GPU" against 2.07 ms remainder (95.0% busy), so pp64's deficit is
// GPU-side: llama.cpp's n=64 pass is ~35.9 ms against our ~41.4. Attributing it needs per-command-
// buffer accounting rather than LastGPUSeconds.
//
// The paragraph below is kept because its component eliminations (gather, upload, encode, download)
// remain valid; only the "fixed submission latency" conclusion drawn from them does not.
//
// The host share is NOT ours to reclaim, which took measuring to establish. Its per-token slope
// (~11.8us/token, fitted from 8.28 ms at n=512 and 14.33 ms at n=1024) looked like a host-side
// dequantize loop, and the prefill path does call gatherEmbed once per token. Measured on the real
// model:
//
//	embedding table dtype  f32 (not quantized — no per-token dequantization happens)
//	gather 512 tokens      0.66 ms   (1.29us/token)
//	upload 4.2 MB block    0.12 ms
//
// 0.78 ms of 8.28. Encoding ~154 dispatches at ~0.6us is another ~0.1 ms, and the logits download is
// 128 KB. The remaining ~7.4 ms is the interval between commit and GPUStartTime — submission and
// driver scheduling, which the GPU timestamps do not span and which no change on our side removes.
//
// So the actionable part of the prefill deficit is the ~4.7% GPU share, i.e. out-executing MPS on a
// GEMM already at 67% of peak. There is no cheap host-side win here.
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
