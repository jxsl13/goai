package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// The examples in this file quantize a float model built by the corresponding
// newQuantTestXxx helper (defined alongside the parity tests in
// quant_*_gguf_test.go): those helpers hand-build a small checkpoint whose
// projection widths are already Q8_0-block-aligned (a multiple of 32), unlike the
// tiny testdata/*_hf.safetensors fixtures used elsewhere in this package, which are
// smaller than one quantization block. They demonstrate the QuantizeXxx(model,
// quantType) -> run-on-GPU-in-quantized-form workflow; see [ExampleQuantLlama] for
// the from-scratch-config version of the same pattern.

// A Cohere quantized to Q8_0 runs its linear layers on the GPU with the weights
// kept in quantized byte form.
func ExampleQuantCohere() {
	m := newQuantTestCohere()
	q, err := nlp.QuantizeCohere(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A DeepSeekV2 (Multi-head Latent Attention + DeepSeekMoE) quantized to Q8_0.
func ExampleQuantDeepSeekV2() {
	m := newQuantTestDeepSeekV2()
	q, err := nlp.QuantizeDeepSeekV2(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A Falcon (fused multi-query attention) quantized to Q8_0.
func ExampleQuantFalcon() {
	m := newQuantTestFalcon()
	q, err := nlp.QuantizeFalcon(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A GPTNeoX (parallel residual, partial rotary) quantized to Q8_0.
func ExampleQuantGPTNeoX() {
	m := newQuantTestGPTNeoX()
	q, err := nlp.QuantizeGPTNeoX(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A Gemma (tied head, decoupled head_dim) quantized to Q8_0.
func ExampleQuantGemma() {
	m := newQuantTestGemma()
	q, err := nlp.QuantizeGemma(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A Gemma2 (sandwich norms, soft-capped logits) quantized to Q8_0.
func ExampleQuantGemma2() {
	m := newQuantTestGemma2()
	q, err := nlp.QuantizeGemma2(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A Jamba (interleaved Mamba/attention layers, dense/sparse-MoE MLPs) quantized to
// Q8_0.
func ExampleQuantJamba() {
	m := newQuantTestJamba()
	q, err := nlp.QuantizeJamba(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// An MPT (ALiBi, no RoPE) quantized to Q8_0.
func ExampleQuantMPT() {
	var m *nlp.MPT = newQuantTestMPT()
	q, err := nlp.QuantizeMPT(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A Mamba (selective-scan SSM) quantized to Q8_0.
func ExampleQuantMamba() {
	m := newQuantTestMamba()
	q, err := nlp.QuantizeMamba(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A Mamba2 (state-space duality) quantized to Q8_0.
func ExampleQuantMamba2() {
	m := newQuantTestMamba2()
	q, err := nlp.QuantizeMamba2(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A Mixtral (sparse top-k Mixture-of-Experts) quantized to Q8_0.
func ExampleQuantMixtral() {
	m := newQuantTestMixtral()
	q, err := nlp.QuantizeMixtral(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A Nemotron (LayerNorm1P, ReLU² MLP) quantized to Q8_0.
func ExampleQuantNemotron() {
	m := newQuantTestNemotron()
	q, err := nlp.QuantizeNemotron(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// An OLMo2 (post-norm residual, full-width QK-norm) quantized to Q8_0.
func ExampleQuantOLMo2() {
	m := newQuantTestOLMo2()
	q, err := nlp.QuantizeOLMo2(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A StableLM (biased LayerNorm, partial rotary) quantized to Q8_0.
func ExampleQuantStableLM() {
	m := newQuantTestStableLM()
	q, err := nlp.QuantizeStableLM(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}

// A StarCoder2 (biased attention, biased 2-layer GELU MLP) quantized to Q8_0.
func ExampleQuantStarCoder2() {
	m := newQuantTestStarCoder2()
	q, err := nlp.QuantizeStarCoder2(m, gguf.Q8_0)
	if err != nil {
		panic(err)
	}
	logits, err := q.Forward(backend.NewContext(), []int{1, 2, 3})
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (3, 12)
}
