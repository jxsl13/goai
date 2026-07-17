//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestCUDALlamaQ4KCloseToF32 validates the direct f32→Q4_K decode path (NewLlamaQ4KCUDA). Every projection
// is quantized to resident ggml Q4_K (4-bit affine super-block, K%256==0) via gguf.Quantize, dispatched by
// the recorder's QMatMulResidentQ4K. Q4_K is more aggressive than Q8 (2× smaller) so a looser directional
// (cosine) tolerance applies; the point is that Q4_K tracks the model — a bad transpose/dequant/dispatch
// collapses the cosine far below the gate.
func TestCUDALlamaQ4KCloseToF32(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	// K%256==0 for every projection: Dim=256 (q/k/v/o/gate/up/head K), Hidden=768 (down K).
	cfg := nlp.LlamaConfig{Vocab: 512, Ctx: 40, Dim: 256, Heads: 8, KVHeads: 2, Layers: 2, Hidden: 768, Eps: 1e-5, RopeBase: 10000}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	f32Dec, err := llamagpu.NewCUDA(m)
	if err != nil {
		t.Fatalf("NewCUDA: %v", err)
	}
	defer f32Dec.Release()
	q4Dec, err := llamagpu.NewLlamaQ4KCUDA(m)
	if err != nil {
		t.Fatalf("NewLlamaQ4KCUDA: %v", err)
	}
	defer q4Dec.Release()

	prompt := []int{5, 12, 3, 9, 1, 7}
	minCos := 1.0
	for pos, tok := range prompt {
		fRow, err := f32Dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("f32 Step pos %d: %v", pos, err)
		}
		qRow, err := q4Dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("q4 Step pos %d: %v", pos, err)
		}
		var dot, nf, nq float64
		for j := range fRow {
			f, q := float64(fRow[j]), float64(qRow[j])
			if math.IsNaN(q) || math.IsInf(q, 0) {
				t.Fatalf("q4 pos %d logit[%d] non-finite: %v", pos, j, q)
			}
			dot += f * q
			nf += f * f
			nq += q * q
		}
		if cos := dot / (math.Sqrt(nf)*math.Sqrt(nq) + 1e-30); cos < minCos {
			minCos = cos
		}
	}
	// 0.98 gate: this fixture is UNIFORM-RANDOM Xavier weights — the worst case for 4-bit quant (no
	// near-zero-heavy structure for Q4_K's affine super-blocks to exploit), so ~0.987 is expected here.
	// Real trained weights track far tighter (gguf's TestQuantizeQ4_KAccurate). A dequant/transpose/dispatch
	// bug collapses the cosine well below 0.98.
	if minCos < 0.98 {
		t.Fatalf("Q4_K min cosine %.5f < 0.98 vs f32 — Q4_K decode diverged", minCos)
	}
	t.Logf("NewLlamaQ4KCUDA tracks f32 CUDA decode: min cosine %.6f over %d positions (4-bit k-quant, 2× smaller than Q8; random-weight worst case)", minCos, len(prompt))
}

// TestCUDALlamaQ4KGenerateReuseSafe checks end-to-end Q4_K generation is deterministic and reuse-safe:
// a reused decoder's greedy Generate must equal a fresh one's (Q4_K weights are resident + immutable, KV
// cache reset by StepN(_,0)). Also exercises the NewQwen3Q4KCUDA typed entry point (== NewLlamaQ4KCUDA).
func TestCUDALlamaQ4KGenerateReuseSafe(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{Vocab: 512, Ctx: 40, Dim: 256, Heads: 8, KVHeads: 2, Layers: 2, Hidden: 768, Eps: 1e-5, RopeBase: 10000}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewQwen3Q4KCUDA(m) // typed alias for NewLlamaQ4KCUDA
	if err != nil {
		t.Fatalf("NewQwen3Q4KCUDA: %v", err)
	}
	defer dec.Release()
	fresh, err := llamagpu.NewLlamaQ4KCUDA(m)
	if err != nil {
		t.Fatalf("NewLlamaQ4KCUDA: %v", err)
	}
	defer fresh.Release()

	prompt := []int{5, 2, 9, 1}
	const maxNew = 6
	first, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate #1: %v", err)
	}
	reused, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate #2 (reuse): %v", err)
	}
	freshOut, err := fresh.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("Generate (fresh): %v", err)
	}
	if len(first) != len(prompt)+maxNew {
		t.Fatalf("Generate returned %d tokens, want %d", len(first), len(prompt)+maxNew)
	}
	for i := range first {
		if reused[i] != first[i] {
			t.Fatalf("Q4_K reuse diverged at token %d: %d vs %d", i, reused[i], first[i])
		}
		if freshOut[i] != first[i] {
			t.Fatalf("Q4_K fresh decoder disagrees at token %d: %d vs %d", i, freshOut[i], first[i])
		}
	}
	t.Logf("NewLlamaQ4KCUDA Generate is reuse-safe + deterministic: %v", first)
}
