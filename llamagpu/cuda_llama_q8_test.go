//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDALlamaQ8CloseToF32 validates the direct f32→Q8 transformer decode path (NewLlamaQ8CUDA) against
// the f32 CUDA decoder. Every projection (fused QKV, o_proj, gate/up/down, lm_head) is quantized to
// resident Q8_0 while biases/QK-norm/RoPE stay f32. Q8 is not bit-exact, so parity is directional
// (cosine) of the logit vectors at every prompt position — held through the KV-cache-carrying decode. A
// wrong-orientation/dequant bug (or a broken fused-QKV Q8) collapses cosine far below the 0.999 gate.
func TestCUDALlamaQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 40, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	f32Dec, err := llamagpu.NewCUDA(m)
	if err != nil {
		t.Fatalf("NewCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewLlamaQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewLlamaQ8CUDA: %v", err)
	}
	defer q8Dec.Release()

	prompt := []int{5, 12, 3, 9, 1, 7}
	minCos := 1.0
	for pos, tok := range prompt {
		fRow, err := f32Dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("f32 Step pos %d: %v", pos, err)
		}
		qRow, err := q8Dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("q8 Step pos %d: %v", pos, err)
		}
		var dot, nf, nq float64
		for j := range fRow {
			f, q := float64(fRow[j]), float64(qRow[j])
			if math.IsNaN(q) || math.IsInf(q, 0) {
				t.Fatalf("q8 pos %d logit[%d] non-finite: %v", pos, j, q)
			}
			dot += f * q
			nf += f * f
			nq += q * q
		}
		cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30)
		if cos < minCos {
			minCos = cos
		}
		if cos < 0.999 {
			t.Fatalf("q8 pos %d: cosine similarity %.5f < 0.999 vs f32 — Q8 transformer decode diverged", pos, cos)
		}
	}
	t.Logf("NewLlamaQ8CUDA tracks f32 CUDA decode: min cosine %.6f over %d positions (fused-QKV + all projections Q8_0)", minCos, len(prompt))
}

// realLlama builds a randomly-initialized Llama of realistic decode geometry (~a 7B-class per-layer
// shape at reduced layer count) so the *Large benchmarks below are genuinely weight-bandwidth-bound —
// the regime where the Q8 projection stream (~4× smaller) wins.
func realLlama(tb testing.TB) *nlp.Llama {
	tb.Helper()
	cfg := nlp.LlamaConfig{
		Vocab: 32000, Ctx: 64, Dim: 2048, Heads: 16, KVHeads: 4, Layers: 8,
		Hidden: 5632, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		tb.Fatal(err)
	}
	return m
}

func benchLlamaStep(b *testing.B, dec *llamagpu.Decoder) {
	b.Helper()
	if _, err := dec.Step(1, 0); err != nil {
		b.Fatalf("prime Step: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dec.Step(2, 1); err != nil {
			b.Fatalf("Step: %v", err)
		}
	}
}

// BenchmarkLlamaDecodeStepF32 / _Q8 quantify the Q8 transformer decode win at a realistic (Dim=2048,
// 8-layer) geometry where decode streams the full f32 projection weights per token. The Q8 variant should
// approach the ~4× the 8-bit projections imply, diluted by the f32 attention/KV work.
// Run: go test -tags 'cuda cgo' -run x -bench LlamaDecodeStep ./llamagpu/
func BenchmarkLlamaDecodeStepF32(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewCUDA(realLlama(b))
	if err != nil {
		b.Fatalf("NewCUDA: %v", err)
	}
	defer dec.Release()
	benchLlamaStep(b, dec)
}

func BenchmarkLlamaDecodeStepQ8(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewLlamaQ8CUDA(realLlama(b))
	if err != nil {
		b.Fatalf("NewLlamaQ8CUDA: %v", err)
	}
	defer dec.Release()
	benchLlamaStep(b, dec)
}

// benchLlamaPrefill times a batched-prefill StepN over a 32-token prompt — the M>1 regime.
func benchLlamaPrefill(b *testing.B, dec *llamagpu.Decoder) {
	b.Helper()
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*7 + 1) % 32000
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := dec.StepN(prompt, 0); err != nil {
			b.Fatalf("StepN: %v", err)
		}
	}
}

// BenchmarkLlamaPrefillF32 / _Q8 measure batched prefill (StepN, M=32). NOTE: unlike decode, Q8 prefill
// is currently SLOWER than f32 — QMatMulResident is a GEMV that re-reads the weight per output row (M×
// the weight bandwidth), while f32 uses cuBLAS (weight reused once). The fix is to route M>1 to the
// existing int8 tensor-core MMQ path (cu_matmul_i8_mmq, which ResidentBQ8's q+scales already fit); these
// benchmarks are the before/after gauge for that work. Decode (M=1) is unaffected and remains ~2.58×.
func BenchmarkLlamaPrefillF32(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewCUDA(realLlama(b))
	if err != nil {
		b.Fatalf("NewCUDA: %v", err)
	}
	defer dec.Release()
	benchLlamaPrefill(b, dec)
}

func BenchmarkLlamaPrefillQ8(b *testing.B) {
	if !cuda.Available() {
		b.Skip("cuda: no CUDA-capable device")
	}
	dec, err := llamagpu.NewLlamaQ8CUDA(realLlama(b))
	if err != nil {
		b.Fatalf("NewLlamaQ8CUDA: %v", err)
	}
	defer dec.Release()
	benchLlamaPrefill(b, dec)
}
