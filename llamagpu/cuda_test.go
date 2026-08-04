//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/gguf"
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

// NewQuantCUDA decodes a Q8-quantized Llama through the unified Decoder: every projection is a
// resident Q8 weight driven by the recorder's QMatMulResident. Quantization changes the outputs vs
// f32, so — as with the metal/vulkan quant tests — the contract is a valid, prefix-preserving,
// correct-length greedy generation (the Q8 numeric accuracy is pinned separately by the backend's
// TestCUDARecorderQ8).
func TestCUDAQuantGenerateValid(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 256, Ctx: 128, Dim: 128, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 352, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuantCUDA(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := []int{1, 9, 42, 17}
	const maxNew = 24
	out, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("generated %d tokens, want %d", len(out)-len(prompt), maxNew)
	}
	for i, tok := range out {
		if i < len(prompt) && tok != prompt[i] {
			t.Fatalf("prompt prefix violated at %d: %d != %d", i, tok, prompt[i])
		}
		if tok < 0 || tok >= cfg.Vocab {
			t.Fatalf("token[%d]=%d out of vocab %d", i, tok, cfg.Vocab)
		}
	}
	t.Logf("llamagpu NewQuantCUDA (Q8) greedy: %d-token prompt → %d valid tokens", len(prompt), len(out))
}

// TestCUDAQuantGenerateValidQ4K drives a Q4_K-quantized Llama through NewQuantCUDA, exercising the
// NATIVE Q4_K serving path added to cudaUploadQWeight (the raw GGUF Q4_K blocks upload as a resident
// Q4_K weight — 0.5625 B/w, ~2× less VRAM than the Q8 upcast — dispatched via QMatMulResidentQ4K's
// GEMV decode + tensor-core WMMA prefill). Dims are multiples of 256 so every projection's K qualifies.
// Contract (as for the Q8 variant): valid, prefix-preserving, correct-length greedy generation.
func TestCUDAQuantGenerateValidQ4K(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 256, Ctx: 128, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(m, gguf.Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuantCUDA(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := []int{1, 9, 42, 17}
	const maxNew = 24
	out, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("generated %d tokens, want %d", len(out)-len(prompt), maxNew)
	}
	for i, tok := range out {
		if i < len(prompt) && tok != prompt[i] {
			t.Fatalf("prompt prefix violated at %d: %d != %d", i, tok, prompt[i])
		}
		if tok < 0 || tok >= cfg.Vocab {
			t.Fatalf("token[%d]=%d out of vocab %d", i, tok, cfg.Vocab)
		}
	}
	t.Logf("llamagpu NewQuantCUDA (native Q4_K) greedy: %d-token prompt → %d valid tokens", len(prompt), len(out))
}

// TestCUDAQuantGenerateValidQ6K drives a Q6_K-quantized Llama through NewQuantCUDA, exercising the
// native Q6_K serving path (raw GGUF Q6_K blocks → resident Q6_K, 0.8203 B/w vs Q8 1.0625, dispatched
// via QMatMulResidentQ6K's GEMV decode + tensor-core WMMA prefill). Dims %256 so every projection's
// K qualifies. Contract: valid, prefix-preserving, correct-length greedy generation.
func TestCUDAQuantGenerateValidQ6K(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 256, Ctx: 128, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(m, gguf.Q6_K)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuantCUDA(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := []int{1, 9, 42, 17}
	const maxNew = 24
	out, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("generated %d tokens, want %d", len(out)-len(prompt), maxNew)
	}
	for i, tok := range out {
		if i < len(prompt) && tok != prompt[i] {
			t.Fatalf("prompt prefix violated at %d: %d != %d", i, tok, prompt[i])
		}
		if tok < 0 || tok >= cfg.Vocab {
			t.Fatalf("token[%d]=%d out of vocab %d", i, tok, cfg.Vocab)
		}
	}
	t.Logf("llamagpu NewQuantCUDA (native Q6_K) greedy: %d-token prompt → %d valid tokens", len(prompt), len(out))
}

// TestCUDAQuantGenerateValidQ5K drives a Q5_K-quantized Llama through NewQuantCUDA, exercising the
// native Q5_K serving path (raw GGUF Q5_K blocks → resident Q5_K, 0.6875 B/w vs Q8 1.0625, dispatched
// via QMatMulResidentQ5K's GEMV decode + tensor-core WMMA prefill). Dims %256. Contract: valid,
// prefix-preserving, correct-length greedy generation.
func TestCUDAQuantGenerateValidQ5K(t *testing.T) {
	if !cuda.Available() {
		t.Skip("cuda: no CUDA-capable device")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 256, Ctx: 128, Dim: 256, Heads: 8, KVHeads: 2, Layers: 4,
		Hidden: 512, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 5)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := nlp.QuantizeLlama(m, gguf.Q5_K)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuantCUDA(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()

	prompt := []int{1, 9, 42, 17}
	const maxNew = 24
	out, err := dec.Generate(prompt, maxNew, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("generated %d tokens, want %d", len(out)-len(prompt), maxNew)
	}
	for i, tok := range out {
		if i < len(prompt) && tok != prompt[i] {
			t.Fatalf("prompt prefix violated at %d: %d != %d", i, tok, prompt[i])
		}
		if tok < 0 || tok >= cfg.Vocab {
			t.Fatalf("token[%d]=%d out of vocab %d", i, tok, cfg.Vocab)
		}
	}
	t.Logf("llamagpu NewQuantCUDA (native Q5_K) greedy: %d-token prompt → %d valid tokens", len(prompt), len(out))
}
