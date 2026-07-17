//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestCUDAMHABias checks cu_attn_softmax_bias (via Recorder.MHABias) — bidirectional multi-head
// attention with a PRECOMPUTED per-head additive bias on the scores, the T5 relative-position-bias
// shape. It reproduces a direct host computation: for head h, query i, key j, score = (q_i·k_j)·scale
// + bias[h,i,j]; softmax over all j (non-causal); out_i = Σ_j softmax·v_j. Bias is [heads,seqQ,seqKV]
// laid out identically to the internal scores buffer. The attention primitive a GPU T5 encoder needs.
func TestCUDAMHABias(t *testing.T) {
	skipNoGPU(t)
	const seq, heads, dk = 5, 2, 3
	const dm = heads * dk

	gen := func(n, seed int, scale float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32((((i+seed)*17)%31 - 15)) * float32(scale)
		}
		return x
	}
	qD := gen(seq*dm, 1, 0.1)
	kD := gen(seq*dm, 2, 0.12)
	vD := gen(seq*dm, 3, 0.09)
	biasD := gen(heads*seq*seq, 4, 0.3) // [heads, seqQ, seqKV]
	scale := float32(1.0 / math.Sqrt(float64(dk)))

	// Reference biased attention.
	ref := make([]float64, seq*dm)
	for h := 0; h < heads; h++ {
		for i := 0; i < seq; i++ {
			sc := make([]float64, seq)
			mx := math.Inf(-1)
			for j := 0; j < seq; j++ {
				var dot float64
				for d := 0; d < dk; d++ {
					dot += float64(qD[i*dm+h*dk+d]) * float64(kD[j*dm+h*dk+d])
				}
				sc[j] = dot*float64(scale) + float64(biasD[h*seq*seq+i*seq+j])
				if sc[j] > mx {
					mx = sc[j]
				}
			}
			var sum float64
			for j := 0; j < seq; j++ {
				sc[j] = math.Exp(sc[j] - mx)
				sum += sc[j]
			}
			for j := 0; j < seq; j++ {
				sc[j] /= sum
			}
			for d := 0; d < dk; d++ {
				var acc float64
				for j := 0; j < seq; j++ {
					acc += sc[j] * float64(vD[j*dm+h*dk+d])
				}
				ref[i*dm+h*dk+d] = acc
			}
		}
	}

	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	dq, _ := cuda.NewDeviceBufferF32(qD)
	dk_, _ := cuda.NewDeviceBufferF32(kD)
	dv, _ := cuda.NewDeviceBufferF32(vD)
	dbias, _ := cuda.NewDeviceBufferF32(biasD)
	do, _ := cuda.NewDeviceF32(seq, dm)
	defer func() {
		dq.Free()
		dk_.Free()
		dv.Free()
		dbias.Free()
		do.Free()
	}()

	if err := rec.MHABias(dq, dk_, dv, do, dbias, seq, seq, dm, heads, heads, dk, 0, 0, scale); err != nil {
		t.Fatalf("MHABias: %v", err)
	}
	if err := rec.Wait(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, seq*dm)
	if err := do.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i := range got {
		d := math.Abs(float64(got[i]) - ref[i])
		if math.IsNaN(float64(got[i])) || d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > 1e-4 {
		t.Fatalf("MHABias diverges from host biased-attention: max abs %.3e", maxAbs)
	}
	t.Logf("cu_attn_softmax_bias / MHABias matches host biased-attention (max abs %.2e), %d heads / seq %d (T5 relpos-bias primitive)", maxAbs, heads, seq)
}
