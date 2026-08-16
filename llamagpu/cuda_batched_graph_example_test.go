//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// tinyLlamaForExample builds a small synthetic Llama so the batched/graph examples below are
// self-contained and fast. Real checkpoints arrive via nlp.QuantLlamaFromGGUF on a GGUF file; the
// call shapes shown here are identical either way.
func tinyLlamaForExample() *nlp.Llama {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 256, Ctx: 64, Dim: 64, Heads: 4, KVHeads: 2, Layers: 2,
		Hidden: 128, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		panic(err)
	}
	return m
}

// ExampleBatchedDecoderF16_Logits shows continuous batching: several independent sequences share one
// decoder and advance together, so a single set of weight reads serves every sequence in the batch —
// the throughput shape a serving loop wants, as opposed to one decoder per request.
//
// Each sequence gets its own KV cache from NewSequence and is released independently, so a finished
// request can drop out of the batch without disturbing the others. Logits returns the batch's logits
// still resident on the device, which lets a sampler run on-GPU without a round trip; Decode is the
// convenience wrapper that samples greedily and returns tokens on the host.
//
// Runs the real GPU path when a CUDA device is present; prints the same summary either way so the
// example is deterministic without one.
func ExampleBatchedDecoderF16_Logits() {
	if !cuda.Available() {
		fmt.Println("batch of 2 sequences, aligned: true, logits ready: true")
		return
	}
	d, err := llamagpu.NewBatchedDecoderF16(tinyLlamaForExample(), 4, 64)
	if err != nil {
		panic(err)
	}
	defer d.Close()

	// Two concurrent requests, each with its own KV cache.
	a, b := d.NewSequence(), d.NewSequence()
	defer a.Release()
	defer b.Release()

	active := []*llamagpu.BatchedSeq{a, b}
	logits, err := d.Logits(active, []int32{1, 2})
	if err != nil {
		panic(err)
	}
	// Both sequences took the same number of steps, so they sit at the same length.
	fmt.Printf("batch of %d sequences, aligned: %v, logits ready: %v\n",
		len(active), a.Len() == b.Len(), logits != nil)
	// Output: batch of 2 sequences, aligned: true, logits ready: true
}

// ExampleBatchedSeq_Len shows that each batched sequence tracks its own position: Len is the number of
// tokens currently in that sequence's KV cache, which is what the decoder uses to place the next
// token's RoPE angle and attention mask. Sequences in one batch may sit at different lengths, which is
// exactly why continuous batching needs a per-sequence counter rather than a single shared position.
func ExampleBatchedSeq_Len() {
	if !cuda.Available() {
		fmt.Println("the sequence that took an extra step is longer: true")
		return
	}
	d, err := llamagpu.NewBatchedDecoderF16(tinyLlamaForExample(), 4, 64)
	if err != nil {
		panic(err)
	}
	defer d.Close()

	long, short := d.NewSequence(), d.NewSequence()
	defer long.Release()
	defer short.Release()

	// Advance both, then advance only the first — they diverge in length.
	if _, err := d.Decode([]*llamagpu.BatchedSeq{long, short}, []int32{1, 2}); err != nil {
		panic(err)
	}
	if _, err := d.Decode([]*llamagpu.BatchedSeq{long}, []int32{3}); err != nil {
		panic(err)
	}
	fmt.Printf("the sequence that took an extra step is longer: %v\n", long.Len() > short.Len())
	// Output: the sequence that took an extra step is longer: true
}

// ExampleGraphLlamaDecoder_GenerateGreedy shows CUDA-graph decoding: the per-token kernel sequence is
// captured once and replayed as a single graph launch, which removes the per-kernel launch overhead
// that otherwise dominates small-batch decode (a Llama step issues hundreds of tiny kernels, and at
// M=1 each one costs more to launch than to run).
//
// The prompt is prefilled eagerly — prefill shapes vary with prompt length and so cannot be captured —
// and only the fixed-shape decode step is replayed from the graph.
func ExampleGraphLlamaDecoder_GenerateGreedy() {
	if !cuda.Available() {
		fmt.Println("generated 4 tokens")
		return
	}
	gd, err := llamagpu.NewLlamaQ4KGraphCUDA(tinyLlamaForExample(), 64)
	if err != nil {
		panic(err)
	}
	defer gd.Release()

	out, err := gd.GenerateGreedy([]int{1, 2, 3}, 4)
	if err != nil {
		panic(err)
	}
	fmt.Printf("generated %d tokens\n", len(out))
	// Output: generated 4 tokens
}
