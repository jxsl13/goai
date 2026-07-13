//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
	"time"
)

// §T374 (§V22, ADR-0019 Phase 2 PAYOFF measurement): before the big nlp DecodeStep integration,
// measure the ACTUAL ceiling of command-buffer batching on a REALISTIC transformer layer — not
// the trivial unary kernel of §T368 (41.8×). Matmul and attention are compute-heavy; the per-op
// commit/wait overhead may be a small fraction of a real layer, so the batching win could be far
// smaller. This A/B runs the SAME 12-op layer chain two ways over the SAME device buffers:
//
//	A) per-op   — each op in its own command buffer (12 commit/waits) = today's path
//	B) batched  — all 12 ops in ONE command buffer (1 commit/wait)   = ADR-0019 Phase 2
//
// The delta is the honest headroom the full integration can recover. Measured at a decode size
// (seq=1, dispatch-bound) and a prefill size (seq=128, compute-bound) to see how it scales.
func TestLayerBatchBench(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	type cfg struct {
		name        string
		seq, dim, h int
	}
	for _, c := range []cfg{
		{"decode-ish D512 S1", 1, 512, 8},
		{"decode-ish D1024 S1", 1, 1024, 16},
		{"prefill D512 S128", 128, 512, 8},
		{"train D512 S256", 256, 512, 8}, // §T411: the BenchmarkGPTTrainingStep shape — the honest tape-batching ceiling
	} {
		S, D, H := c.seq, c.dim, c.h
		dk := D / H
		F := 4 * D
		const eps = 1e-5
		scale := float32(1.0 / math.Sqrt(float64(dk)))

		mk := func(n int) *DeviceBuffer {
			b, err := NewDeviceBufferF32(make([]float32, n))
			if err != nil {
				t.Fatal(err)
			}
			return b
		}
		// activations + weights, all device-resident (the integration keeps these on-GPU)
		x, xn, xn2 := mk(S*D), mk(S*D), mk(S*D)
		q, k, v, attn, ao, mo := mk(S*D), mk(S*D), mk(S*D), mk(S*D), mk(S*D), mk(S*D)
		h := mk(S * F)
		wq, wk, wv, wo := mk(D*D), mk(D*D), mk(D*D), mk(D*D)
		w1, w2 := mk(D*F), mk(F*D)
		g1, g2 := mk(D), mk(D)
		all := []*DeviceBuffer{x, xn, xn2, q, k, v, attn, ao, mo, h, wq, wk, wv, wo, w1, w2, g1, g2}
		defer func() {
			for _, b := range all {
				b.Release()
			}
		}()

		// The 12-op transformer layer as a list of record-mode steps (order matters).
		ops := []func(r *Recorder) error{
			func(r *Recorder) error { return r.RMSNorm(x, g1, xn, S, D, eps) },
			func(r *Recorder) error { return r.MatMul(xn, wq, q, S, D, D) },
			func(r *Recorder) error { return r.MatMul(xn, wk, k, S, D, D) },
			func(r *Recorder) error { return r.MatMul(xn, wv, v, S, D, D) },
			func(r *Recorder) error { return r.flashattn(q, k, v, attn, S, D, H, dk, 1, H, scale) },
			func(r *Recorder) error { return r.MatMul(attn, wo, ao, S, D, D) },
			func(r *Recorder) error { return r.Binary(x, ao, x, 0) }, // residual add (in-place)
			func(r *Recorder) error { return r.RMSNorm(x, g2, xn2, S, D, eps) },
			func(r *Recorder) error { return r.MatMul(xn2, w1, h, S, D, F) },
			func(r *Recorder) error { return r.Unary(h, h, unaryGELU) },
			func(r *Recorder) error { return r.MatMul(h, w2, mo, S, F, D) },
			func(r *Recorder) error { return r.Binary(x, mo, x, 0) },
		}

		batched := func() {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			for _, op := range ops {
				if err := op(r); err != nil {
					t.Fatal(err)
				}
			}
			if err := r.Finish(); err != nil {
				t.Fatal(err)
			}
			r.Free()
		}
		perOp := func() {
			for _, op := range ops {
				r, err := NewRecorder()
				if err != nil {
					t.Fatal(err)
				}
				if err := op(r); err != nil {
					t.Fatal(err)
				}
				if err := r.Finish(); err != nil {
					t.Fatal(err)
				}
				r.Free()
			}
		}
		timeit := func(fn func()) float64 {
			fn() // warmup
			const N = 50
			s := time.Now()
			for range N {
				fn()
			}
			return float64(time.Since(s).Microseconds()) / N / 1000
		}
		po := timeit(perOp)
		ba := timeit(batched)
		t.Logf("%-20s (%d ops): per-op %.3f ms | batched %.3f ms | speedup %.2fx",
			c.name, len(ops), po, ba, po/ba)
	}
}
