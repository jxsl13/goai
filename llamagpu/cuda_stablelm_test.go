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
