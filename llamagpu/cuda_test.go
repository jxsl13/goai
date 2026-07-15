//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// The CUDA Decoder variant (the backend-agnostic batched core + the cuda.Recorder adapter) generates
// the SAME greedy token sequence as the library's reference per-op decode — the NVIDIA adapter is
// correct end-to-end (the CUDA analog of the metal §T408 / vulkan §T409 cross-reference).
func TestCUDAGenerateMatchesReference(t *testing.T) {
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
	dec, err := llamagpu.NewCUDA(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := []int{5, 12, 3}
	const maxNew = 20
	gpuOut, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	refOut, err := m.Generate(prompt, maxNew, nlp.Greedy(), nlp.WithBackend(backend.Reference()))
	if err != nil {
		t.Fatal(err)
	}
	if len(gpuOut) != len(refOut) {
		t.Fatalf("length: cuda %d vs ref %d", len(gpuOut), len(refOut))
	}
	for i := range gpuOut {
		if gpuOut[i] != refOut[i] {
			t.Fatalf("token[%d]: cuda %d vs ref %d\ncuda=%v\nref=%v", i, gpuOut[i], refOut[i], gpuOut, refOut)
		}
	}
	t.Logf("llamagpu CUDA Decoder.Generate == nlp.Llama.Generate greedy: %d tokens", len(gpuOut))
}

// StepN prefill then per-token Step over a longer GQA model — exercises the fused-QKV RoPEPair band
// path and the KV-cache attention through the cuda.Recorder at model scale, still matching the
// reference greedy sequence.
func TestCUDAPrefillStepMatchesReference(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 256, Ctx: 128, Dim: 128, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 352, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := llamagpu.NewCUDA(m)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := []int{1, 9, 42, 17, 3, 200}
	const maxNew = 24
	gpuOut, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	refOut, err := m.Generate(prompt, maxNew, nlp.Greedy(), nlp.WithBackend(backend.Reference()))
	if err != nil {
		t.Fatal(err)
	}
	if len(gpuOut) != len(refOut) {
		t.Fatalf("length: cuda %d vs ref %d", len(gpuOut), len(refOut))
	}
	for i := range gpuOut {
		if gpuOut[i] != refOut[i] {
			t.Fatalf("token[%d]: cuda %d vs ref %d\ncuda=%v\nref=%v", i, gpuOut[i], refOut[i], gpuOut, refOut)
		}
	}
	t.Logf("llamagpu CUDA prefill+decode == reference greedy: %d-token prompt → %d tokens", len(prompt), len(gpuOut))
}
