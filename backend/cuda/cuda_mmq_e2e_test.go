//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAResidentMMQvsF16RealWeights proves the int8 MMQ prefill weight is a valid drop-in for the
// f16 prefill weight on REAL model weights: it builds ResidentBF16 and ResidentMMQ from the SAME
// dequantized Qwen projection matrices, runs both on identical activations, and checks that MMQ
// tracks f16 within the int8 quantization budget while using ~half the weight VRAM.
func TestCUDAResidentMMQvsF16RealWeights(t *testing.T) {
	skipNoGPU(t)
	for _, tc := range []struct{ path, label string }{
		{"../../models/qwen2.5-0.5b-instruct-q8_0.gguf", "Qwen2.5-0.5B"},
		{"../../models/qwen2.5-1.5b-instruct-q8_0.gguf", "Qwen2.5-1.5B"}, // larger dims + GQA
	} {
		t.Run(tc.label, func(t *testing.T) { mmqVsF16RealWeights(t, tc.path) })
	}
}

func mmqVsF16RealWeights(t *testing.T, path string) {
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not present (%s)", path)
	}
	f, err := os.Open(path)
	mustTB(t, err)
	rf, err := gguf.ReadRaw(f)
	f.Close()
	mustTB(t, err)

	deq := func(n string) *tensor.Tensor {
		qt, ok := rf.Tensors[n]
		if !ok {
			t.Fatalf("missing tensor %s", n)
		}
		dt, err := qt.Dequantize()
		mustTB(t, err)
		return dt
	}

	// A spread of real projections (attention + FFN, different K/N shapes).
	names := []string{
		"blk.0.attn_q.weight", "blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight",
	}
	const M = 64
	rng := rand.New(rand.NewSource(2026))
	var totF16Bytes, totMMQBytes int64

	for _, name := range names {
		w := transpose2D(deq(name)) // [K][N] f32, same layout NewResidentBF16 expects
		k, n := w.Shape()[0], w.Shape()[1]
		if k%32 != 0 || n%64 != 0 {
			t.Logf("%s [K=%d,N=%d] skipped (K%%32/N%%64 constraint)", name, k, n)
			continue
		}
		wf := w.Cast(tensor.F32).Contiguous().Storage().F32()

		a := make([]float32, M*k)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		da, err := cuda.NewDeviceF32(M, k)
		mustTB(t, err)
		mustTB(t, da.UploadF32(a))

		rf16, err := cuda.NewResidentBF16(w)
		mustTB(t, err)
		rmmq, err := cuda.NewResidentMMQ(w)
		mustTB(t, err)

		d16, err := rf16.MatMulDevice(da)
		mustTB(t, err)
		dmq, err := rmmq.MatMulDevice(da)
		mustTB(t, err)
		h16, err := d16.ToHost()
		mustTB(t, err)
		hmq, err := dmq.ToHost()
		mustTB(t, err)

		// vs true f32 GEMM, and MMQ vs f16.
		var seF16, seMMQ, seDiff, sr float64
		for m := 0; m < M; m++ {
			for j := 0; j < n; j++ {
				var ref float64
				for kk := 0; kk < k; kk++ {
					ref += float64(a[m*k+kk]) * float64(wf[kk*n+j])
				}
				g16 := h16.AtF64(m, j)
				gmq := hmq.AtF64(m, j)
				seF16 += (g16 - ref) * (g16 - ref)
				seMMQ += (gmq - ref) * (gmq - ref)
				seDiff += (gmq - g16) * (gmq - g16)
				sr += ref * ref
			}
		}
		rmsF16 := math.Sqrt(seF16 / sr)
		rmsMMQ := math.Sqrt(seMMQ / sr)
		rmsDiff := math.Sqrt(seDiff / sr)
		f16Bytes := int64(k) * int64(n) * 2                    // f16 weight
		mmqBytes := int64(k)*int64(n) + int64(k)/32*int64(n)*4 // int8 + per-block f32 scale
		totF16Bytes += f16Bytes
		totMMQBytes += mmqBytes
		t.Logf("%-24s [K=%d N=%d]  f16-vs-f32 %.4f  MMQ-vs-f32 %.4f  MMQ-vs-f16 %.4f  VRAM %.0f%% of f16",
			name, k, n, rmsF16, rmsMMQ, rmsDiff, 100*float64(mmqBytes)/float64(f16Bytes))
		if rmsMMQ > 2e-2 {
			t.Errorf("%s: MMQ vs f32 RMS %.4f > 2e-2 (real-weight int8 accuracy)", name, rmsMMQ)
		}
		if rmsDiff > 2e-2 {
			t.Errorf("%s: MMQ diverges from f16 by RMS %.4f > 2e-2", name, rmsDiff)
		}

		d16.Free()
		dmq.Free()
		rf16.Free()
		rmmq.Free()
		da.Free()
	}

	if totF16Bytes == 0 {
		t.Skip("no eligible weights")
	}
	t.Logf("TOTAL prefill weight VRAM: f16 %.1f MB  MMQ %.1f MB  (%.0f%%, %.1f MB saved)",
		float64(totF16Bytes)/1e6, float64(totMMQBytes)/1e6,
		100*float64(totMMQBytes)/float64(totF16Bytes), float64(totF16Bytes-totMMQBytes)/1e6)
	t.Log("RESIDENT MMQ IS A VALID DROP-IN FOR f16 PREFILL ON REAL WEIGHTS (int8 accuracy + ~half VRAM)")
}
