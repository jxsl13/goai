//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// TestIsolatedOpCostIsNotAdditive corrects a measurement method — and with it the per-op decode
// breakdown in TestDecodeNonMatmulBreakdown, whose numbers are inflated for the reason shown here.
//
// Timing an op by recording it N times into one command buffer and dividing looks obviously right
// and is not: N IDENTICAL dispatches reading and writing the SAME buffers are serialised by
// write-after-write hazards, so each one pays a full pipeline drain that it would never pay in a
// real chain, where neighbouring ops differ and their work overlaps.
//
// The rows sweep shows how extreme that is for a small op (M2 Pro, dim=2048):
//
//	rows=  1   13.31us      rows= 16   14.02us
//	rows=  2   13.38us      rows= 32   15.18us
//	rows=  4   14.63us      rows= 64   17.87us
//	fit: fixed 12.73us/dispatch, marginal 0.080us/row
//
// 64x the work for 1.34x the time, i.e. 96% "fixed cost" at decode's rows=1. Read naively that says
// a decode token spends ~12.7us per dependent dispatch and that fusing rmsnorm into the following
// projection would recover ~13us x 44 = ~560us, nearly 10% of a token, which would put decode past
// llama.cpp. That was the conclusion I was about to build on.
//
// Measuring the actual pair instead of inferring it:
//
//	qmatmul alone          49.42us
//	rmsnorm + qmatmul      51.04us
//	rmsnorm's real cost      1.62us  (3%), not 13.31us
//
// About 8x less. Fusing all 44 would save ~71us of a ~5740us token — 1.2%, under the noise floor,
// and nowhere near worth the architectural change.
//
// CONSEQUENCE for TestDecodeNonMatmulBreakdown: every per-op number there was produced this way and
// is inflated by roughly this factor. Its total landing on the ~2.5ms non-streaming budget was
// coincidence, not corroboration — if the parts are each ~8x too large, agreement with the whole
// means the whole was not what I thought either. What survives is the ORDERING (attention costs
// more than rmsnorm, which costs more than elementwise); what does not survive is any absolute
// figure or any share-of-token claim derived from it.
//
// How to measure a per-op contribution properly: time the realistic sequence with and without the
// op, as the pair above does. Repeating one op in isolation measures a shape the GPU never runs.
//
// Reported, not asserted on absolute timings.
func TestIsolatedOpCostIsNotAdditive(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const dim = 2048
	maxRows := 64
	x, _ := NewDeviceBufferF32(make([]float32, maxRows*dim))
	g, _ := NewDeviceBufferF32(make([]float32, dim))
	o, _ := NewDeviceBufferF32(make([]float32, maxRows*dim))
	defer func() {
		x.Release()
		g.Release()
		o.Release()
	}()
	timeIt := func(f func(*Recorder)) float64 {
		best := 1e18
		for range 11 {
			r, _ := NewRecorder()
			for range 20 {
				f(r)
			}
			r.Commit()
			r.Wait()
			if d := LastGPUSeconds() / 20; d < best {
				best = d
			}
			r.Free()
		}
		return best * 1e6
	}
	fmt.Printf("FIX rmsnorm rows-sweep (dim=%d):\n", dim)
	var xs, ys []float64
	for _, rows := range []int{1, 2, 4, 8, 16, 32, 64} {
		d := timeIt(func(r *Recorder) { _ = r.RMSNorm(x, g, o, rows, dim, 1e-5) })
		kb := float64(rows*dim*4*3) / 1024
		fmt.Printf("FIX   rows=%3d  %7.2fus  (%7.1f KB, %5.2f GB/s)\n", rows, d, kb, kb/1024/1024/(d/1e6))
		xs = append(xs, float64(rows))
		ys = append(ys, d)
	}
	// fit over the larger rows where marginal cost dominates
	n := len(xs)
	m := (ys[n-1] - ys[n-3]) / (xs[n-1] - xs[n-3])
	f := ys[n-1] - m*xs[n-1]
	fmt.Printf("FIX   fit: fixed = %.2fus/dispatch, marginal = %.3fus/row\n", f, m)
	fmt.Printf("FIX   at rows=1 the fixed part is %.0f%% of the call\n", 100*f/ys[0])

	// What a fusion would actually recover: time the real decode pair (rmsnorm then the
	// projection that consumes it) against the projection alone.
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
	xin, _ := NewDeviceBufferF32(make([]float32, K))
	xn, _ := NewDeviceBufferF32(make([]float32, K))
	out, _ := NewDeviceBufferF32(make([]float32, N))
	defer func() { xin.Release(); xn.Release(); out.Release() }()

	mmOnly := timeIt(func(r *Recorder) { _ = r.QMatMulResident(xn, rq, out, 1) })
	pair := timeIt(func(r *Recorder) {
		_ = r.RMSNorm(xin, g, xn, 1, K, 1e-5)
		_ = r.QMatMulResident(xn, rq, out, 1)
	})
	fmt.Printf("FIX pair: qmatmul alone %.2fus, rmsnorm+qmatmul %.2fus -> fusing saves %.2fus (%.0f%%)\n",
		mmOnly, pair, pair-mmOnly, 100*(pair-mmOnly)/pair)
	fmt.Printf("FIX projected: 44 rmsnorm/token x %.2fus = %.0fus of a ~5740us token (%.1f%%)\n",
		pair-mmOnly, 44*(pair-mmOnly), 100*44*(pair-mmOnly)/5740)
}
