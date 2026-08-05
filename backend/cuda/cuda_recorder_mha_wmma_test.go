//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// refGQACausal computes host-side grouped-query causal attention: Q [seq, qHeads·hd],
// K/V [seq, kvHeads·hd] → out [seq, qHeads·hd]. Query head h reads kv head h/(qHeads/kvHeads).
// Query i (== key i, square) attends to keys j ≤ i. Oracle for the fused Recorder.MHA path.
func refGQACausal(q, k, v []float32, seq, qHeads, kvHeads, hd int, scale float64) []float32 {
	group := qHeads / kvHeads
	out := make([]float32, seq*qHeads*hd)
	for h := 0; h < qHeads; h++ {
		kvh := h / group
		for i := 0; i < seq; i++ {
			sc := make([]float64, i+1)
			mx := math.Inf(-1)
			for j := 0; j <= i; j++ {
				var d float64
				for c := 0; c < hd; c++ {
					d += float64(q[i*qHeads*hd+h*hd+c]) * float64(k[j*kvHeads*hd+kvh*hd+c])
				}
				d *= scale
				sc[j] = d
				if d > mx {
					mx = d
				}
			}
			var sum float64
			for j := 0; j <= i; j++ {
				sc[j] = math.Exp(sc[j] - mx)
				sum += sc[j]
			}
			for c := 0; c < hd; c++ {
				var acc float64
				for j := 0; j <= i; j++ {
					acc += sc[j] / sum * float64(v[j*kvHeads*hd+kvh*hd+c])
				}
				out[i*qHeads*hd+h*hd+c] = float32(acc)
			}
		}
	}
	return out
}

// TestCUDARecorderMHAWMMAPrefill validates the fused tensor-core flash-prefill fast path wired into
// Recorder.MHA (recorder.go): for causal-SQUARE prefill at hd==64 with seq a multiple of 16, MHA now
// runs the one-kernel cu_wmma_attn_gqa (no materialized HBM scores) instead of the scores→softmax→PV
// triple. This proves the wiring (arg order, caller scale, GQA head mapping, in-place o output) matches
// a host GQA-causal oracle within the f16-accumulation tolerance the same kernel already carries in
// gqaCore — so the ~3× prefill-attention speedup is bit-faithful to the incumbent-tolerance contract.
func TestCUDARecorderMHAWMMAPrefill(t *testing.T) {
	skipNoGPU(t)
	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()
	const hd = 64
	gen := func(n, seed int) []float32 {
		s := make([]float32, n)
		x := uint32(seed*2654435761 + 1)
		for i := range s {
			x = x*1664525 + 1013904223
			s[i] = (float32(x>>8)/float32(1<<24))*2 - 1 // ~U(-1,1)
		}
		return s
	}
	cases := []struct {
		seq, qHeads, kvHeads int
	}{
		{32, 2, 2}, // plain MHA (kvHeads==qHeads)
		{32, 4, 2}, // GQA 2:1
		{48, 8, 2}, // GQA 4:1 (TinyLlama-class ratio), seq=48
		{16, 1, 1}, // minimal square
	}
	scale := 1.0 / math.Sqrt(hd)
	for ci, c := range cases {
		q := gen(c.seq*c.qHeads*hd, ci+1)
		k := gen(c.seq*c.kvHeads*hd, ci+11)
		v := gen(c.seq*c.kvHeads*hd, ci+21)
		dq, _ := cuda.NewDeviceBufferF32(q)
		dk, _ := cuda.NewDeviceBufferF32(k)
		dv, _ := cuda.NewDeviceBufferF32(v)
		do, _ := cuda.NewDeviceF32(c.seq, c.qHeads*hd)
		// dm (arg 4) is unused by the attention kernels; pass the q width. causal=1, window=0.
		if err := rec.MHA(dq, dk, dv, do, c.seq, c.seq, c.qHeads*hd, c.qHeads, c.kvHeads, hd, 1, 0, float32(scale)); err != nil {
			t.Fatalf("case %d MHA: %v", ci, err)
		}
		if err := rec.Wait(); err != nil {
			t.Fatalf("case %d wait: %v", ci, err)
		}
		got := make([]float32, c.seq*c.qHeads*hd)
		if err := do.DownloadF32(got); err != nil {
			t.Fatalf("case %d download: %v", ci, err)
		}
		dq.Free()
		dk.Free()
		dv.Free()
		do.Free()

		want := refGQACausal(q, k, v, c.seq, c.qHeads, c.kvHeads, hd, scale)
		var num, den float64
		for idx := range want {
			if math.IsNaN(float64(got[idx])) {
				t.Fatalf("case %d [%d]: NaN", ci, idx)
			}
			d := float64(got[idx] - want[idx])
			num += d * d
			den += float64(want[idx]) * float64(want[idx])
		}
		relRMS := math.Sqrt(num / den)
		if relRMS > 5e-3 { // f16 WMMA accumulation vs f64 host softmax; same tol class as gqaCore's wmma parity
			t.Fatalf("case %d (seq=%d qH=%d kvH=%d): rel-RMS %.3e > 5e-3 (fused MHA diverges from host GQA-causal)",
				ci, c.seq, c.qHeads, c.kvHeads, relRMS)
		}
		t.Logf("case %d seq=%d qH=%d kvH=%d: fused MHA rel-RMS %.3e", ci, c.seq, c.qHeads, c.kvHeads, relRMS)
	}
	t.Logf("fused Recorder.MHA flash-prefill matches host GQA-causal attention across %d shapes", len(cases))
}
