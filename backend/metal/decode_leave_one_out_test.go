//go:build darwin && cgo

package metal

import (
	"fmt"
	"os"
	"testing"
)

// TestDecodeLeaveOneOut decomposes decode by DIFFERENTIAL measurement: build a realistic per-layer
// decode chain with the true TinyLlama shapes, then time it with one op class removed. The
// difference is that class's real marginal cost.
//
// This replaces the per-op timings in TestDecodeNonMatmulBreakdown, which were inflated ~8x by
// timing each op as N identical repeats (see TestIsolatedOpCostIsNotAdditive). An op only costs what
// it adds to the sequence it actually runs in.
//
// Measured on an M2 Pro (22 layers, dim 2048, hidden 5632, 32 heads over 4 KV heads, dk 64, ctx 512):
//
//	full chain                6094.2us
//	without matmul            1888.5us   contribution 4205.6us (69.0%)
//	without attention         4174.7us   contribution 1919.4us (31.5%)
//	without rmsnorm           5964.6us   contribution  129.5us ( 2.1%)
//	without elementwise       6119.6us   contribution  -25.5us (-0.4%)
//
// The chain's 6094us against a real measured token of ~5740us is the check that this synthetic
// mirrors the model rather than some other workload.
//
// What it establishes:
//
//   - The weight matmuls are 69% and run at ~178 GB/s against a ~180 GB/s ceiling
//     (TestDecodeMatmulIsAtMemoryCeiling), so that 69% is not recoverable.
//   - ATTENTION is 31.5% — 1919us per token — and is the only large recoverable item.
//   - rmsnorm is 2.1%, matching the 1.62us-per-call pair measurement (44 x 1.62 = 71us) and
//     confirming that fusing it is worth ~1%, not the ~10% the isolated timing implied.
//   - elementwise is free: removing it does not shorten the chain at all, so it overlaps entirely
//     with neighbouring work. A negative contribution is the expected shape of "no effect" here.
//
// Note the asymmetry: isolated timing OVERSTATED rmsnorm (14.96us vs 1.62us real) but UNDERSTATED
// attention (67.65us x 22 = 1488us vs 1919us real). Ops that overlap with their neighbours are
// flattered by isolation; ops that serialise the pipeline are penalised by it. That is why only the
// differential form can be used to size an optimisation.
//
// Attention moves ~23 MB/token at ctx=512, which is ~128us at the measured streaming rate, against
// 1919us spent — and the kernel runs 32 threadgroups of 32 threads, about 1024 threads for a GPU
// that can host far more. Occupancy, not bandwidth, is the open question there.
//
// Reported, not asserted on absolute timings; the assertion is only that matmul and attention
// dominate.
func TestDecodeLeaveOneOut(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const (
		layers     = 22
		dim        = 2048
		hidden     = 5632
		heads      = 32
		kvh        = 4
		dk         = 64
		ctxDefault = 512
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
	x, xn, q, kk, vv, att, ob := nb(dim), nb(dim), nb(dim), nb(kvDim), nb(kvDim), nb(dim), nb(dim)
	g, u := nb(hidden), nb(hidden)
	gam := nb(dim)
	ctx := ctxDefault
	if v := os.Getenv("GOAI_LOO_CTX"); v != "" {
		fmt.Sscanf(v, "%d", &ctx)
	}
	kc, vc := nb(ctx*kvDim), nb(ctx*kvDim)
	defer func() {
		for _, b := range []*DeviceBuffer{x, xn, q, kk, vv, att, ob, g, u, gam, kc, vc} {
			b.Release()
		}
	}()

	// skip: "" = full chain, else the op class omitted
	chain := func(r *Recorder, skip string) {
		for _, l := range ls {
			if skip != "rmsnorm" {
				_ = r.RMSNorm(x, gam, xn, 1, dim, 1e-5)
			}
			if skip != "matmul" {
				_ = r.QMatMulResident(xn, l.wq, q, 1)
				_ = r.QMatMulResident(xn, l.wk, kk, 1)
				_ = r.QMatMulResident(xn, l.wv, vv, 1)
			}
			if skip != "attention" {
				_ = r.MHAAt(q, kc, vc, att, 0, 1, ctx, dim, heads, kvh, dk, 1, 0, 0.125)
			}
			if skip != "matmul" {
				_ = r.QMatMulResident(att, l.wo, ob, 1)
			}
			if skip != "elementwise" {
				_ = r.BinaryN(x, ob, x, 0, dim)
			}
			if skip != "rmsnorm" {
				_ = r.RMSNorm(x, gam, xn, 1, dim, 1e-5)
			}
			if skip != "matmul" {
				_ = r.QMatMulResident(xn, l.wg, g, 1)
				_ = r.QMatMulResident(xn, l.wu, u, 1)
			}
			if skip != "elementwise" {
				_ = r.Unary(g, g, 6)
				_ = r.BinaryN(g, u, g, 2, hidden)
			}
			if skip != "matmul" {
				_ = r.QMatMulResident(g, l.wd, ob, 1)
			}
			if skip != "elementwise" {
				_ = r.BinaryN(x, ob, x, 0, dim)
			}
		}
	}
	timeIt := func(skip string) float64 {
		best := 1e18
		for range 9 {
			r, _ := NewRecorder()
			chain(r, skip)
			r.Commit()
			r.Wait()
			if d := LastGPUSeconds(); d < best {
				best = d
			}
			r.Free()
		}
		return best * 1e6
	}
	full := timeIt("")
	fmt.Printf("LOO full chain (22 layers) = %.1fus  -> %.1f tok/s if this were the whole token\n", full, 1e6/full)
	share := map[string]float64{}
	for _, s := range []string{"matmul", "attention", "rmsnorm", "elementwise"} {
		w := timeIt(s)
		share[s] = 100 * (full - w) / full
		fmt.Printf("LOO without %-12s %8.1fus   contribution %7.1fus (%4.1f%%)\n", s, w, full-w, share[s])
	}
	// The shares recorded above predate the dk-split attention kernels (pair then quad, 2.9x and
	// 1.05-1.18x). Attention has since fallen from 31.5% to ~10% of the chain and matmul risen to
	// ~86%, which is the optimisation working rather than the profile breaking. The guard now only
	// asserts that matmul still dominates — the claim the rest of this file rests on.
	if share["matmul"] < 50 {
		t.Errorf("decode profile changed shape: matmul %.1f%%, attention %.1f%% — the conclusions "+
			"recorded in this file assume a matmul-dominated decode", share["matmul"], share["attention"])
	}
}
