//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// cosStep drives both decoders over the prompt and returns the min per-position logit cosine.
func cosStep(t *testing.T, f32Dec, q8Dec *llamagpu.Decoder, prompt []int) float64 {
	t.Helper()
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
				t.Fatalf("q8 pos %d logit[%d] non-finite", pos, j)
			}
			dot += f * q
			nf += f * f
			nq += q * q
		}
		if c := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30); c < minCos {
			minCos = c
		}
	}
	return minCos
}

// TestCUDAGemmaQ8CloseToF32 validates Q8 for Gemma v1 (NewGemmaQ8CUDA) — fills the Q8 gap. All projections
// AND the tied lm_head go Q8_0 (the lm_head matters: Gemma's vocab is huge, so leaving it f32 would dilute
// the win). √dim embed scaling + GeGLU stay f32-side.
func TestCUDAGemmaQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/gemma_hf.safetensors")
	if err != nil {
		t.Skipf("gemma testdata unavailable: %v", err)
	}
	m, err := nlp.GemmaFromHF(ts, nlp.GemmaConfig{Heads: 2, KVHeads: 1, Eps: 1e-6, RopeBase: 10000, Ctx: 32})
	if err != nil {
		t.Fatalf("GemmaFromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewGemmaCUDA(m)
	if err != nil {
		t.Fatalf("NewGemmaCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewGemmaQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewGemmaQ8CUDA: %v", err)
	}
	defer q8Dec.Release()
	minCos := cosStep(t, f32Dec, q8Dec, []int{3, 7, 1, 9, 4})
	if minCos < 0.999 {
		t.Fatalf("Gemma Q8 min cosine %.5f < 0.999 vs f32", minCos)
	}
	t.Logf("NewGemmaQ8CUDA tracks f32 CUDA decode: min cosine %.6f (all projections + tied lm_head Q8)", minCos)
}

// TestCUDAGraniteQ8CloseToF32 validates Q8 for IBM Granite (NewGraniteQ8CUDA) — fills the Q8 gap. Granite
// maps to nlp.Llama with config scalars (embMult/attn-scale/residual-mult/logit-scale, folded into the
// upload), so the scaled Wo/Wdown are quantized post-scale.
func TestCUDAGraniteQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/granite_hf.safetensors")
	if err != nil {
		t.Skipf("granite testdata unavailable: %v", err)
	}
	m, err := nlp.GraniteFromHF(ts, nlp.LlamaConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		EmbeddingMult: 12, AttentionMult: 0.5, ResidualMult: 0.22, LogitsScale: 8,
	})
	if err != nil {
		t.Fatalf("GraniteFromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewGraniteCUDA(m)
	if err != nil {
		t.Fatalf("NewGraniteCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewGraniteQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewGraniteQ8CUDA: %v", err)
	}
	defer q8Dec.Release()
	minCos := cosStep(t, f32Dec, q8Dec, []int{3, 7, 1, 9, 4})
	if minCos < 0.999 {
		t.Fatalf("Granite Q8 min cosine %.5f < 0.999 vs f32", minCos)
	}
	t.Logf("NewGraniteQ8CUDA tracks f32 CUDA decode: min cosine %.6f (Llama-core + scaled Wo/Wdown Q8)", minCos)
}

// TestCUDACohereQ8CloseToF32 validates that the mkLinS fix fully quantizes Cohere Q8: Cohere folds a
// logit_scale (0.0625) into the tied lm_head via linS — which was f32-only before, leaving the lm_head
// f32 under the Q8 hook. Now linS = d.mkLinS (Q8 under hook), so the scaled lm_head goes Q8 too.
func TestCUDACohereQ8CloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/cohere_hf.safetensors")
	if err != nil {
		t.Skipf("cohere testdata unavailable: %v", err)
	}
	m, err := nlp.CohereFromHF(ts, nlp.CohereConfig{Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, LogitScale: 0.0625, Ctx: 32})
	if err != nil {
		t.Fatalf("CohereFromHF: %v", err)
	}
	f32Dec, err := llamagpu.NewCohereCUDA(m)
	if err != nil {
		t.Fatalf("NewCohereCUDA: %v", err)
	}
	defer f32Dec.Release()
	q8Dec, err := llamagpu.NewCohereQ8CUDA(m)
	if err != nil {
		t.Fatalf("NewCohereQ8CUDA: %v", err)
	}
	defer q8Dec.Release()
	minCos := cosStep(t, f32Dec, q8Dec, []int{3, 7, 1, 9, 4})
	if minCos < 0.999 {
		t.Fatalf("Cohere Q8 min cosine %.5f < 0.999 vs f32", minCos)
	}
	t.Logf("NewCohereQ8CUDA tracks f32 CUDA decode: min cosine %.6f (logit-scaled lm_head now Q8 via mkLinS)", minCos)
}
