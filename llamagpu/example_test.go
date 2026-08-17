//go:build darwin && cgo

package llamagpu_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// ExampleDecoder_Generate builds a small Llama, uploads it to the GPU once, and generates greedily —
// the batched decode produces the same tokens as nlp.Llama.Generate, ~24× faster. Real models arrive
// the same way via nlp.LlamaFromGGUF.
func ExampleDecoder_Generate() {
	if !metal.Available() {
		fmt.Println("generated 8 new tokens") // no GPU on this machine; keep the example's output
		return
	}
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}, 42)
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.New(m) // weights + KV cache become device-resident
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	prompt := []int{3, 17, 9}
	out, err := dec.Generate(prompt, 8, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Printf("generated %d new tokens\n", len(out)-len(prompt))
	// Output: generated 8 new tokens
}

// ExampleDecoder_StepN prefills a whole prompt in one recorded step — the 41× prefill fast path
// Generate uses internally — and reads the next-token logits from the last row.
//
// It also shows StepNLast, the variant to prefer when only the next token matters: the LM head runs
// over one row instead of k and the host download shrinks from k·vocab to vocab, while the returned
// row is identical to the last row of StepN.
func ExampleDecoder_StepN() {
	if !metal.Available() {
		fmt.Println("logits for 48 tokens (StepNLast agrees: true)")
		return
	}
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}, 42)
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	prompt := []int{3, 17, 9, 21}
	all, err := dec.StepN(prompt, 0) // one command buffer for the whole prompt
	if err != nil {
		panic(err)
	}
	last := all[(len(prompt)-1)*m.Config.Vocab:] // logits after the full prompt

	// Same prefill, but projecting only the final row — the cheaper call when you just want the
	// next token. Its result matches the tail of StepN's above.
	lastOnly, err := dec.StepNLast(prompt, 0)
	if err != nil {
		panic(err)
	}
	fmt.Printf("logits for %d tokens (StepNLast agrees: %v)\n", len(last), len(lastOnly) == len(last))
	// Output: logits for 48 tokens (StepNLast agrees: true)
}

// ExampleDecoder_ProfileMetalStep attributes one normal decode command buffer to stable Metal
// encoder labels. The profiler is diagnostic and opt-in; ordinary Step calls keep the production
// recorder fast path.
func ExampleDecoder_ProfileMetalStep() {
	if !metal.Available() || !metal.RecorderProfilingAvailable() {
		fmt.Println("profile contains encoder timings")
		return
	}
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}, 42)
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	_, profile, err := dec.ProfileMetalStep(3, 0, 128)
	if err != nil {
		panic(err)
	}
	if len(profile.Events) == 0 || profile.CommandDuration <= 0 {
		panic("empty Metal profile")
	}
	fmt.Println("profile contains encoder timings")
	// Output: profile contains encoder timings
}

// ExampleNewQuant decodes a quantized model: the projections stay in their 4-8× smaller ggml form
// on the GPU, so models too big to run as float still decode on the batched path.
func ExampleNewQuant() {
	if !metal.Available() {
		fmt.Println("generated 6 new tokens")
		return
	}
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 192, Eps: 1e-5, RopeBase: 10000, // hidden is a multiple of the Q8_0 block (32)
	}, 42)
	if err != nil {
		panic(err)
	}
	qm, err := nlp.QuantizeLlama(m, gguf.Q8_0) // or nlp.QuantLlamaFromGGUF for a real model file
	if err != nil {
		panic(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	out, err := dec.Generate([]int{3, 17}, 6, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Printf("generated %d new tokens\n", len(out)-2)
	// Output: generated 6 new tokens
}

// ExampleSpeculativeGenerate accelerates a large target with a small draft, losslessly: the output
// is exactly the target's distribution (with a greedy sampler, exactly its token sequence).
func ExampleSpeculativeGenerate() {
	if !metal.Available() {
		fmt.Println("generated 8 tokens (speculative)")
		return
	}
	cfg := nlp.LlamaConfig{
		Vocab: 48, Ctx: 64, Dim: 64, Heads: 8, KVHeads: 2, Layers: 3,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}
	tm, err := nlp.NewLlama(cfg, 1)
	if err != nil {
		panic(err)
	}
	dcfg := cfg
	dcfg.Layers = 1 // the draft: same tokenizer/vocab, far cheaper per step
	dm, err := nlp.NewLlama(dcfg, 2)
	if err != nil {
		panic(err)
	}
	target, err := llamagpu.New(tm)
	if err != nil {
		panic(err)
	}
	defer target.Release()
	draft, err := llamagpu.New(dm)
	if err != nil {
		panic(err)
	}
	defer draft.Release()

	out, _, err := llamagpu.SpeculativeGenerate(target, draft, []int{5, 12, 3}, 8, 4, nlp.Greedy(), 7)
	if err != nil {
		panic(err)
	}
	fmt.Printf("generated %d tokens (speculative)\n", len(out)-3)
	// Output: generated 8 tokens (speculative)
}

// ExampleGPTDecoder_Generate decodes a GPT-2-style model (LayerNorm, learned positions, biased GELU
// MLP) on the batched path — the sibling of the Llama Decoder. Real models arrive via
// nlp.FromSafetensors.
func ExampleGPTDecoder_Generate() {
	if !metal.Available() {
		fmt.Println("generated 5 new tokens (next-token logits: 32)")
		return
	}
	cfg := nlp.GPTConfig{Vocab: 32, Ctx: 16, Dim: 32, Heads: 4, Layers: 1, Eps: 1e-5}
	w := func(shape ...int) *tensor.Tensor { // tiny deterministic weights
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(i%13-6) * 0.03
		}
		return x
	}
	ones := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return x
	}
	m, err := nlp.FromSafetensors(cfg, map[string]*tensor.Tensor{
		"tok_emb": w(cfg.Vocab, cfg.Dim), "pos_emb": w(cfg.Ctx, cfg.Dim),
		"head": w(cfg.Dim, cfg.Vocab), "lnf.gamma": ones(cfg.Dim), "lnf.beta": w(cfg.Dim),
		"blocks.0.ln1.gamma": ones(cfg.Dim), "blocks.0.ln1.beta": w(cfg.Dim),
		"blocks.0.attn.wq": w(cfg.Dim, cfg.Dim), "blocks.0.attn.wk": w(cfg.Dim, cfg.Dim),
		"blocks.0.attn.wv": w(cfg.Dim, cfg.Dim), "blocks.0.attn.wo": w(cfg.Dim, cfg.Dim),
		"blocks.0.ln2.gamma": ones(cfg.Dim), "blocks.0.ln2.beta": w(cfg.Dim),
		"blocks.0.ffn.w1": w(cfg.Dim, 4*cfg.Dim), "blocks.0.ffn.b1": w(4 * cfg.Dim),
		"blocks.0.ffn.w2": w(4*cfg.Dim, cfg.Dim), "blocks.0.ffn.b2": w(cfg.Dim),
	})
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	out, err := dec.Generate([]int{3, 7}, 5, nlp.Greedy())
	if err != nil {
		panic(err)
	}

	// StepNLast prefills a prompt and returns only the final row's logits — the same numbers
	// Generate reads to pick the next token, without materialising a row per prompt position.
	logits, err := dec.StepNLast([]int{3, 7}, 0)
	if err != nil {
		panic(err)
	}
	fmt.Printf("generated %d new tokens (next-token logits: %d)\n", len(out)-2, len(logits))
	// Output: generated 5 new tokens (next-token logits: 32)
}

// ExamplePromptLookupGenerate accelerates generation with NO draft model: continuations are guessed
// by n-gram lookup in the sequence's own history and verified in one batched step — lossless, and
// effective when the output repeats the input (summarization, RAG, code editing).
func ExamplePromptLookupGenerate() {
	if !metal.Available() {
		fmt.Println("generated 10 tokens (prompt-lookup)")
		return
	}
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 64, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}, 42)
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	prompt := []int{7, 8, 9, 7, 8, 9, 7, 8} // repeating patterns are where lookup drafting pays off
	out, _, err := llamagpu.PromptLookupGenerate(dec, prompt, 10, 3, 6, nlp.Greedy(), 1)
	if err != nil {
		panic(err)
	}
	fmt.Printf("generated %d tokens (prompt-lookup)\n", len(out)-len(prompt))
	// Output: generated 10 tokens (prompt-lookup)
}

// Step is the decoder's smallest unit: feed ONE token at a position, get the
// next-token logits back, with the KV cache advancing on the GPU. Ctx and
// Vocab expose the geometry a custom decode loop needs; StepHidden and
// StepNHidden return the final HIDDEN states too — the attachment point for
// auxiliary decoding heads (Medusa drafts its next window from the hidden rows
// of the current verify pass). Generate wraps exactly this loop.
func ExampleDecoder_Step() {
	if !metal.Available() {
		fmt.Println("ctx=32 vocab=48 logits=48 hidden rows=3")
		return
	}
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}, 42)
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	logits, err := dec.Step(3, 0) // token 3 at position 0
	if err != nil {
		panic(err)
	}
	_, h1, err := dec.StepHidden(17, 1) // logits + one hidden row
	if err != nil {
		panic(err)
	}
	_ = h1
	_, h, err := dec.StepNHidden([]int{9, 4, 21}, 2) // a whole window in one step
	if err != nil {
		panic(err)
	}
	fmt.Printf("ctx=%d vocab=%d logits=%d hidden rows=%d\n",
		dec.Ctx(), dec.Vocab(), len(logits), len(h)/64)
	// Output: ctx=32 vocab=48 logits=48 hidden rows=3
}

// The GPT decoder mirrors the Llama surface — Step/StepHidden/StepNHidden and
// the same geometry accessors — over the learned-positional-embedding
// architecture (GPT-2-family checkpoints).
func ExampleGPTDecoder_Step() {
	if !metal.Available() {
		fmt.Println("ctx=16 vocab=32 logits=32 hidden rows=2")
		return
	}
	cfg := nlp.GPTConfig{Vocab: 32, Ctx: 16, Dim: 32, Heads: 4, Layers: 1, Eps: 1e-5}
	w := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(i%13-6) * 0.03
		}
		return x
	}
	ones := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return x
	}
	m, err := nlp.FromSafetensors(cfg, map[string]*tensor.Tensor{
		"tok_emb": w(cfg.Vocab, cfg.Dim), "pos_emb": w(cfg.Ctx, cfg.Dim),
		"head": w(cfg.Dim, cfg.Vocab), "lnf.gamma": ones(cfg.Dim), "lnf.beta": w(cfg.Dim),
		"blocks.0.ln1.gamma": ones(cfg.Dim), "blocks.0.ln1.beta": w(cfg.Dim),
		"blocks.0.attn.wq": w(cfg.Dim, cfg.Dim), "blocks.0.attn.wk": w(cfg.Dim, cfg.Dim),
		"blocks.0.attn.wv": w(cfg.Dim, cfg.Dim), "blocks.0.attn.wo": w(cfg.Dim, cfg.Dim),
		"blocks.0.ln2.gamma": ones(cfg.Dim), "blocks.0.ln2.beta": w(cfg.Dim),
		"blocks.0.ffn.w1": w(cfg.Dim, 4*cfg.Dim), "blocks.0.ffn.b1": w(4 * cfg.Dim),
		"blocks.0.ffn.w2": w(4*cfg.Dim, cfg.Dim), "blocks.0.ffn.b2": w(cfg.Dim),
	})
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	logits, err := dec.Step(3, 0)
	if err != nil {
		panic(err)
	}
	_, _, err = dec.StepHidden(5, 1)
	if err != nil {
		panic(err)
	}
	_, h, err := dec.StepNHidden([]int{7, 2}, 2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("ctx=%d vocab=%d logits=%d hidden rows=%d\n",
		dec.Ctx(), dec.Vocab(), len(logits), len(h)/cfg.Dim)
	// Output: ctx=16 vocab=32 logits=32 hidden rows=2
}

// ExampleDecoder_StepNLast prefills a prompt but downloads only the final row's logits — the row
// generation actually consumes. The transformer body still runs every prompt token, so the KV cache
// is populated identically and the returned row equals StepN's last row; what is saved is the LM head
// over the k-1 discarded rows and their download.
func ExampleDecoder_StepNLast() {
	if !metal.Available() {
		fmt.Println("last-row logits: 48")
		return
	}
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 48, Ctx: 32, Dim: 64, Heads: 8, KVHeads: 2, Layers: 2,
		Hidden: 176, Eps: 1e-5, RopeBase: 10000,
	}, 42)
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	prompt := []int{3, 17, 9, 21}
	last, err := dec.StepNLast(prompt, 0) // one row back instead of len(prompt)
	if err != nil {
		panic(err)
	}
	fmt.Printf("last-row logits: %d\n", len(last))
	// Output: last-row logits: 48
}

// ExampleGPTDecoder_StepNLast is [Decoder.StepNLast] for the GPT-2 architecture: same prefill, same
// single-row download.
func ExampleGPTDecoder_StepNLast() {
	if !metal.Available() {
		fmt.Println("last-row logits: 32")
		return
	}
	cfg := nlp.GPTConfig{Vocab: 32, Ctx: 16, Dim: 32, Heads: 4, Layers: 1, Eps: 1e-5}
	w := func(shape ...int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape(shape))
		d := x.Storage().F32()
		for i := range d {
			d[i] = float32(i%13-6) * 0.03
		}
		return x
	}
	ones := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{n})
		d := x.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return x
	}
	m, err := nlp.FromSafetensors(cfg, map[string]*tensor.Tensor{
		"tok_emb": w(cfg.Vocab, cfg.Dim), "pos_emb": w(cfg.Ctx, cfg.Dim),
		"head": w(cfg.Dim, cfg.Vocab), "lnf.gamma": ones(cfg.Dim), "lnf.beta": w(cfg.Dim),
		"blocks.0.ln1.gamma": ones(cfg.Dim), "blocks.0.ln1.beta": w(cfg.Dim),
		"blocks.0.attn.wq": w(cfg.Dim, cfg.Dim), "blocks.0.attn.wk": w(cfg.Dim, cfg.Dim),
		"blocks.0.attn.wv": w(cfg.Dim, cfg.Dim), "blocks.0.attn.wo": w(cfg.Dim, cfg.Dim),
		"blocks.0.ln2.gamma": ones(cfg.Dim), "blocks.0.ln2.beta": w(cfg.Dim),
		"blocks.0.ffn.w1": w(cfg.Dim, 4*cfg.Dim), "blocks.0.ffn.b1": w(4 * cfg.Dim),
		"blocks.0.ffn.w2": w(4*cfg.Dim, cfg.Dim), "blocks.0.ffn.b2": w(cfg.Dim),
	})
	if err != nil {
		panic(err)
	}
	dec, err := llamagpu.NewGPT(m)
	if err != nil {
		panic(err)
	}
	defer dec.Release()

	last, err := dec.StepNLast([]int{3, 7, 1}, 0)
	if err != nil {
		panic(err)
	}
	fmt.Printf("last-row logits: %d\n", len(last))
	// Output: last-row logits: 32
}
