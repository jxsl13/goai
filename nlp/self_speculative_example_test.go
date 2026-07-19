package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

func exampleLayerSkipGPT() *nlp.GPT {
	ts, _, err := safetensors.LoadFile("testdata/gpt.safetensors")
	if err != nil {
		panic(err)
	}
	m, err := nlp.FromSafetensors(nlp.GPTConfig{
		Vocab: 17, Ctx: 8, Dim: 8, Heads: 2, Layers: 2, Eps: 1e-5,
	}, ts)
	if err != nil {
		panic(err)
	}
	return m
}

// ForwardExit turns an intermediate residual stream into logits through the same
// final norm and LM head as the full model. Layer 0 means "stop after the first
// block"; the sequence and vocabulary dimensions remain unchanged.
func ExampleGPT_ForwardExit() {
	m := exampleLayerSkipGPT()
	logits, err := m.ForwardExit(backend.NewContext(), []int{3, 1}, 0)
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (2, 17)
}

// Self-speculative decoding needs no second model: an early exit drafts several
// tokens and the mature pass corrects them without changing the target distribution.
// This example exits at the final block, where draft and target are identical, so
// every proposal is accepted.
func ExampleSelfSpeculativeDecode() {
	m := exampleLayerSkipGPT()
	out, stats, err := nlp.SelfSpeculativeDecode(m, []int{3, 1}, 3, 1, 2, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Printf("tokens=%d accepted=%.0f%%\n", len(out), stats.AcceptanceRate()*100)
	// Output: tokens=5 accepted=100%
}

// Llama.ForwardExit is the readout used by Meta's released LayerSkip Llama
// checkpoints. This tiny model has one block, so exit 0 is also the mature exit.
func ExampleLlama_ForwardExit() {
	m := exampleTinyLlama()
	logits, err := m.ForwardExit(backend.NewContext(), []int{3, 1}, 0)
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape())
	// Output: (2, 12)
}

// The Llama entry point uses the same lossless accept/reject driver as GPT. With
// the final (here only) block as draft exit, all proposals match the target.
func ExampleLlamaSelfSpeculativeDecode() {
	m := exampleTinyLlama()
	out, stats, err := nlp.LlamaSelfSpeculativeDecode(m, []int{3, 1}, 3, 0, 2, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Printf("tokens=%d accepted=%.0f%%\n", len(out), stats.AcceptanceRate()*100)
	// Output: tokens=5 accepted=100%
}
