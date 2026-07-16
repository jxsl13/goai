//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// The CUDA StableLM decoder (LayerNorm-with-bias + partial rotary via the cu_rope_partial*
// kernels, reusing the Llama SwiGLU/GQA/KV core) must greedy-generate token-for-token with the
// nlp.StableLM reference — the e2e anchor that the two shared-core generalizations (rotaryDim,
// lnBias) and the partial-rope kernel plumbing are all correct together.
func TestCUDAStableLMMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/stablelm_hf.safetensors")
	if err != nil {
		t.Skipf("stablelm testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.StableLMFromHF(ts, nlp.StableLMConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("StableLMFromHF: %v", err)
	}

	dec, err := llamagpu.NewStableLMCUDA(m)
	if err != nil {
		t.Fatalf("NewStableLMCUDA: %v", err)
	}
	defer dec.Release()

	prompt := []int{3, 7, 1, 9}
	const maxNew = 12
	gpuOut, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("cuda generate: %v", err)
	}
	refOut, err := m.Generate(prompt, maxNew, nlp.Greedy(), nlp.WithBackend(backend.Reference()))
	if err != nil {
		t.Fatalf("ref generate: %v", err)
	}
	if len(gpuOut) != len(refOut) {
		t.Fatalf("length: cuda %d vs ref %d", len(gpuOut), len(refOut))
	}
	for i := range gpuOut {
		if gpuOut[i] != refOut[i] {
			t.Fatalf("token[%d]: cuda %d vs ref %d\ncuda=%v\nref=%v", i, gpuOut[i], refOut[i], gpuOut, refOut)
		}
	}
	t.Logf("llamagpu NewStableLMCUDA.Generate == nlp.StableLM.Generate greedy: %d tokens (LayerNorm-bias + partial rotary)", len(gpuOut))
}

// The CUDA StarCoder2 decoder (LayerNorm-bias + biased q/k/v/o + biased GELU-MLP + full rope,
// GQA) must greedy-generate token-for-token with the nlp.StarCoder2 reference — the second
// new-architecture GPU decoder, exercising every core generalization except partial rotary.
func TestCUDAStarCoder2MatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/starcoder2_hf.safetensors")
	if err != nil {
		t.Skipf("starcoder2 testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.StarCoder2FromHF(ts, nlp.StarCoder2Config{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("StarCoder2FromHF: %v", err)
	}

	dec, err := llamagpu.NewStarCoder2CUDA(m)
	if err != nil {
		t.Fatalf("NewStarCoder2CUDA: %v", err)
	}
	defer dec.Release()

	prompt := []int{3, 7, 1, 9}
	const maxNew = 12
	gpuOut, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("cuda generate: %v", err)
	}
	refOut, err := m.Generate(prompt, maxNew, nlp.Greedy(), nlp.WithBackend(backend.Reference()))
	if err != nil {
		t.Fatalf("ref generate: %v", err)
	}
	if len(gpuOut) != len(refOut) {
		t.Fatalf("length: cuda %d vs ref %d", len(gpuOut), len(refOut))
	}
	for i := range gpuOut {
		if gpuOut[i] != refOut[i] {
			t.Fatalf("token[%d]: cuda %d vs ref %d\ncuda=%v\nref=%v", i, gpuOut[i], refOut[i], gpuOut, refOut)
		}
	}
	t.Logf("llamagpu NewStarCoder2CUDA.Generate == nlp.StarCoder2.Generate greedy: %d tokens (biased proj + GELU-MLP)", len(gpuOut))
}

// The CUDA Phi decoder exercises EVERY core generalization at once: one-norm parallel residual,
// biased q/k/v/dense, biased GELU-MLP, partial rotary, biased lm_head, final LayerNorm. It must
// greedy-generate token-for-token with the nlp.Phi reference — the third new-arch GPU decoder.
func TestCUDAPhiMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	ts, _, err := safetensors.LoadFile("../nlp/testdata/phi_hf.safetensors")
	if err != nil {
		t.Skipf("phi testdata unavailable (run make golden): %v", err)
	}
	m, err := nlp.PhiFromHF(ts, nlp.PhiConfig{
		Heads: 4, Eps: 1e-5, RopeBase: 10000, RotaryPct: 0.5, Ctx: 32,
	})
	if err != nil {
		t.Fatalf("PhiFromHF: %v", err)
	}

	dec, err := llamagpu.NewPhiCUDA(m)
	if err != nil {
		t.Fatalf("NewPhiCUDA: %v", err)
	}
	defer dec.Release()

	prompt := []int{3, 7, 1, 9}
	const maxNew = 12
	gpuOut, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatalf("cuda generate: %v", err)
	}
	refOut, err := m.Generate(prompt, maxNew, nlp.Greedy(), nlp.WithBackend(backend.Reference()))
	if err != nil {
		t.Fatalf("ref generate: %v", err)
	}
	if len(gpuOut) != len(refOut) {
		t.Fatalf("length: cuda %d vs ref %d", len(gpuOut), len(refOut))
	}
	for i := range gpuOut {
		if gpuOut[i] != refOut[i] {
			t.Fatalf("token[%d]: cuda %d vs ref %d\ncuda=%v\nref=%v", i, gpuOut[i], refOut[i], gpuOut, refOut)
		}
	}
	t.Logf("llamagpu NewPhiCUDA.Generate == nlp.Phi.Generate greedy: %d tokens (all six generalizations)", len(gpuOut))
}
