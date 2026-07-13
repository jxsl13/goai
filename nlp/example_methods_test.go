package nlp_test

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// exampleTinyLlama builds the small randomly-initialized Llama shared by the method
// examples below: 1 block, width 16, a 12-token vocabulary — big enough to be real,
// small enough to run in milliseconds.
func exampleTinyLlama() *nlp.Llama {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 16, Heads: 2, Layers: 1, Hidden: 24, Eps: 1e-5,
	}, 1)
	if err != nil {
		panic(err)
	}
	return m
}

// Generate is the everyday "write more text" entry point: give the model a prompt and
// a sampler, get the prompt back with new tokens appended. Each new token costs one
// forward pass because the KV-cache remembers the past. With the deterministic Greedy
// sampler the same prompt always yields the same continuation.
func ExampleLlama_Generate() {
	m := exampleTinyLlama()
	out, err := m.Generate([]int{1, 2, 3}, 4 /*maxNew*/, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Println(len(out), out[:3]) // prompt (3) + 4 generated; prompt is echoed first
	// Output: 7 [1 2 3]
}

// DecodeStep is the engine under Generate: it feeds ONE token through the model,
// appends that token's key/value to the cache from NewCache, and returns next-token
// logits [1, vocab]. Decoding a prompt token-by-token this way gives exactly the same
// final logits as one full Forward over the whole prompt — the KV-cache is a speed
// trick, not an approximation.
func ExampleLlama_DecodeStep() {
	m := exampleTinyLlama()
	prompt := []int{1, 2, 3, 4, 5}

	full, err := m.Forward(backend.NewContext(), prompt) // reference: all at once
	if err != nil {
		panic(err)
	}

	ctx := backend.NewContext()
	cache := m.NewCache() // empty KV-cache, one K/V slot per block
	var last *tensor.Tensor
	for pos, tok := range prompt {
		if last, err = m.DecodeStep(ctx, cache, tok, pos); err != nil {
			panic(err)
		}
	}

	agree := true
	seq := full.Shape()[0]
	for j := range full.Shape()[1] {
		if math.Abs(last.AtF64(0, j)-full.AtF64(seq-1, j)) > 1e-9 {
			agree = false
		}
	}
	fmt.Println(cache.Len(), agree)
	// Output: 5 true
}

// Embed turns token ids into their embedding vectors and ForwardFromEmbed runs the
// rest of the transformer on them — together they are Forward split in half. The split
// exists so a training loop can modify the embeddings in between (e.g. add NEFTune
// noise). Embed gives [seq, dim]; ForwardFromEmbed returns logits [seq, vocab].
func ExampleLlama_ForwardFromEmbed() {
	m := exampleTinyLlama()
	ctx := backend.NewContext()

	x, err := m.Embed(ctx, []int{3, 1, 4})
	if err != nil {
		panic(err)
	}
	logits, err := m.ForwardFromEmbed(ctx, x)
	if err != nil {
		panic(err)
	}
	fmt.Println(x.Shape(), logits.Shape())
	// Output: (3, 16) (3, 12)
}

// ForwardHidden stops one step short of Forward: it returns the final hidden states
// [seq, dim] — the model's internal representation BEFORE the output projection turns
// it into vocabulary logits. Auxiliary decoding heads (like MedusaHeads) attach here:
// Forward(tokens) equals ForwardHidden(tokens) times the output matrix.
func ExampleLlama_ForwardHidden() {
	m := exampleTinyLlama()
	hidden, err := m.ForwardHidden(backend.NewContext(), []int{3, 1, 4})
	if err != nil {
		panic(err)
	}
	fmt.Println(hidden.Shape()) // [seq, dim], not [seq, vocab]
	// Output: (3, 16)
}

// Params lists every trainable tensor so an optimizer can update them all in one loop.
// For this 1-block model that is: the token embedding, six attention/norm tensors and
// three SwiGLU matrices in the block, the final norm gain and the output projection —
// 12 tensors in total.
func ExampleLlama_Params() {
	m := exampleTinyLlama()
	ps := m.Params()
	fmt.Println(len(ps), ps[0].Shape()) // ps[0] is the [vocab, dim] token embedding
	// Output: 12 (12, 16)
}

// StreamingLLM decoding runs forever at constant memory: the cache from NewStreamCache
// keeps only the first `sinks` tokens (attention sinks) plus a rolling window of the
// most recent ones, evicting the middle. After feeding 5 tokens through StreamStep with
// sinks=1 and window=2 the cache holds just 3 rows, yet still yields next-token logits
// [1, vocab]. StreamGenerate wraps the same loop into prompt-in, tokens-out form.
func ExampleLlama_StreamGenerate() {
	m := exampleTinyLlama()
	ctx := backend.NewContext()

	cache := m.NewStreamCache()
	var logits *tensor.Tensor
	var err error
	for _, tok := range []int{1, 2, 3, 4, 5} {
		if logits, err = m.StreamStep(ctx, cache, tok, 1 /*sinks*/, 2 /*window*/); err != nil {
			panic(err)
		}
	}
	fmt.Println(cache.Len(), logits.Shape()) // bounded at sinks+window = 3 rows

	out, err := m.StreamGenerate([]int{1, 2, 3}, 4 /*maxNew*/, 1, 2, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Println(len(out))
	// Output:
	// 3 (1, 12)
	// 7
}

// Self-Extend lets a model read far beyond its training window by grouping distant
// positions: token pairs closer than `window` use their true positions, farther pairs
// use compressed ones (position divided by `group`). With group=1 nothing is compressed,
// so SelfExtendForward must reproduce plain Forward exactly — the sanity check shown
// here.
func ExampleLlama_SelfExtendForward() {
	m := exampleTinyLlama()
	ctx := backend.NewContext()
	tokens := []int{1, 2, 3, 4}

	plain, err := m.Forward(ctx, tokens)
	if err != nil {
		panic(err)
	}
	se, err := m.SelfExtendForward(ctx, tokens, 2 /*window*/, 1 /*group=1 → no grouping*/)
	if err != nil {
		panic(err)
	}

	agree := true
	for i := range se.Shape()[0] {
		for j := range se.Shape()[1] {
			if math.Abs(se.AtF64(i, j)-plain.AtF64(i, j)) > 1e-9 {
				agree = false
			}
		}
	}
	fmt.Println(se.Shape(), agree)
	// Output: (4, 12) true
}

// SelfExtendGenerate greedily extends a prompt using Self-Extend grouped attention:
// each step re-runs the grouped forward over the whole sequence and appends the argmax
// token. It is the generation companion of SelfExtendForward — full attention beyond
// the trained context, at O(seq²) per step (analysis-scale, no KV cache).
func ExampleLlama_SelfExtendGenerate() {
	m := exampleTinyLlama()
	out, err := m.SelfExtendGenerate(backend.NewContext(), []int{1, 2}, 3 /*maxNew*/, 4 /*window*/, 2 /*group*/)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(out), out[:2]) // prompt (2) + 3 generated
	// Output: 5 [1 2]
}

// A quantized Llama decodes exactly like the float one, at a fraction of the weight
// memory: NewCache makes an empty KV-cache, DecodeStep advances one token and returns
// logits [1, vocab], Generate wraps the loop, and Close releases the device-resident
// quantized weight buffers when you are done (idempotent).
func ExampleQuantLlama_Generate() {
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 12, Ctx: 16, Dim: 32, Heads: 4, KVHeads: 2, Layers: 1, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 3)
	if err != nil {
		panic(err)
	}
	q, err := nlp.QuantizeLlama(m, gguf.Q8_0) // 8-bit blocks, near-lossless
	if err != nil {
		panic(err)
	}

	cache := q.NewCache()
	logits, err := q.DecodeStep(backend.NewContext(), cache, 1 /*token*/, 0 /*pos*/)
	if err != nil {
		panic(err)
	}
	fmt.Println(logits.Shape(), cache.Len())

	out, err := q.Generate([]int{1, 2}, 3 /*maxNew*/, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	fmt.Println(len(out))

	fmt.Println(q.Close()) // free the quantized weight buffers
	// Output:
	// (1, 12) 1
	// 5
	// <nil>
}

// Keys and Values dequantize the 8-bit cache back to plain f32 [tokens, dim] tensors
// for attention. The round trip is near-lossless: a row of ones quantizes to Q8_0 and
// comes back within a fraction of a percent, while the store costs ~3.76x less memory
// than f32.
func ExampleQuantKVCache_Keys() {
	c, err := nlp.NewQuantKVCache(32)
	if err != nil {
		panic(err)
	}
	k := tensor.New(tensor.F32, tensor.Shape{32})
	v := tensor.New(tensor.F32, tensor.Shape{32})
	for j := range 32 {
		k.SetF64(1, j)
		v.SetF64(-1, j)
	}
	if err := c.Append(k, v); err != nil {
		panic(err)
	}

	keys, err := c.Keys()
	if err != nil {
		panic(err)
	}
	vals, err := c.Values()
	if err != nil {
		panic(err)
	}
	fmt.Println(keys.Shape(), vals.Shape())
	fmt.Println(math.Abs(keys.AtF64(0, 0)-1) < 0.01, math.Abs(vals.AtF64(0, 0)+1) < 0.01)
	// Output:
	// (1, 32) (1, 32)
	// true true
}

// MaskLogits is the per-step primitive of regex-constrained decoding: it sets the logit
// of every token that would break the pattern to -Inf and reports whether generation
// may stop (EOS) — only allowed once the text so far matches the whole regex. From the
// start state of "ab+" only "a" survives, and stopping is not yet allowed.
func ExampleRegexGuide_MaskLogits() {
	vocab := []string{"a", "b", "c"}
	g, err := nlp.NewRegexGuide("ab+", vocab)
	if err != nil {
		panic(err)
	}

	logits := []float64{0, 0, 0}
	eosAllowed := g.MaskLogits(g.Start(), logits, -1 /*no EOS token*/)
	fmt.Println(eosAllowed, logits[0], math.IsInf(logits[1], -1), math.IsInf(logits[2], -1))
	// Output: false 0 true true
}

// Sampler wraps any TokenSampler so the output is GUARANTEED to match a regex: each
// draw masks the tokens the FSM forbids, lets the inner sampler pick, and advances the
// FSM by the pick. Greedy alone would choose "b" first (its logit is higher), but the
// pattern "ab+" forces "a" before any "b".
func ExampleRegexGuide_Sampler() {
	vocab := []string{"a", "b"}
	g, err := nlp.NewRegexGuide("ab+", vocab)
	if err != nil {
		panic(err)
	}

	s := g.Sampler(nlp.Greedy(), -1 /*no EOS token*/) // stateful: fresh Sampler per sequence
	first := s.Sample([]float64{0, 1})                // greedy alone would pick "b" (id 1)
	second := s.Sample([]float64{0, 1})               // now "b" is legal
	fmt.Println(first, second)
	// Output: 0 1
}

// Mirostat carries feedback state across tokens: its threshold Mu starts at 2*Tau
// (here 10 for the default Tau=5 bits) and is nudged after every sample so the text's
// average surprise stays on target. Reset restores Mu to 2*Tau — call it (or make a
// fresh Mirostat) before reusing the sampler on a new sequence, or the old sequence's
// adaptation leaks into the new one.
func ExampleMirostat_Reset() {
	m := nlp.NewMirostat(1)
	fmt.Println(m.Mu) // fresh: 2*Tau

	m.Sample(make([]float64, 8)) // sampling moves the threshold...
	fmt.Println(m.Mu == 10)

	m.Reset() // ...Reset restores it for the next sequence
	fmt.Println(m.Mu)
	// Output:
	// 10
	// false
	// 10
}

// A watermark hides a statistical signal in generated text: at each step GreenMask
// derives a secret "green" quarter of the vocabulary from the previous token, Sampler
// biases generation toward it, and Detect recomputes the same lists to count green
// tokens — a z-score above 4 flags machine text at a ~3e-5 false-positive rate. Here
// a strong bias makes every sampled token green, so detection is unambiguous.
func ExampleWatermark_Detect() {
	w, err := nlp.NewWatermark(32, nlp.WithWatermarkDelta(8))
	if err != nil {
		panic(err)
	}

	mask := w.GreenMask(0) // the green list seeded by previous token 0
	green := 0
	for _, g := range mask {
		if g {
			green++
		}
	}
	fmt.Println(green) // gamma=0.25 of the 32-token vocabulary

	s := w.Sampler(nlp.Greedy()) // wrap any sampler; plugs into Generate loops
	tokens := []int{0}
	for len(tokens) < 40 {
		tokens = append(tokens, s.SampleWithHistory(make([]float64, 32), tokens))
	}

	z, hits, scored := w.Detect(tokens)
	fmt.Println(z > nlp.WatermarkZThreshold, hits, scored)
	// Output:
	// 8
	// true 39 39
}

// Params exposes the Medusa head matrices (one [dim, vocab] projection per head) so an
// optimizer can train them while the base model stays frozen — the whole point of
// Medusa: only these small heads learn.
func ExampleMedusaHeads_Params() {
	heads, err := nlp.NewMedusaHeads(3 /*heads*/, 8 /*dim*/, 16 /*vocab*/, 1 /*seed*/)
	if err != nil {
		panic(err)
	}
	ps := heads.Params()
	fmt.Println(len(ps), ps[0].Shape())
	// Output: 3 (8, 16)
}

// ToJSON writes the tokenizer as a minimal HuggingFace tokenizer.json (model.type
// "BPE" with vocab and merges), the interchange format other tools read. The round
// trip through BPEFromJSON preserves behavior: "abc" still merges a+b then ab+c into
// the single token id 4.
func ExampleBPETokenizer_ToJSON() {
	tok, err := nlp.NewBPE([]string{"a", "b", "c", "ab", "abc"}, []string{"a b", "ab c"})
	if err != nil {
		panic(err)
	}
	data, err := tok.ToJSON()
	if err != nil {
		panic(err)
	}
	back, err := nlp.BPEFromJSON(data)
	if err != nil {
		panic(err)
	}
	fmt.Println(tok.Encode("abc"), back.Encode("abc"))
	// Output: [4] [4]
}

// ToJSON serializes a Unigram tokenizer to the HuggingFace tokenizer.json form
// (model.type "Unigram" with [piece, score] pairs), so it can be stored or shared and
// loaded back with UnigramFromJSON. The round trip preserves segmentation: "hello"
// still encodes to the single piece "▁hello" (id 1).
func ExampleUnigram_ToJSON() {
	u, err := nlp.NewUnigram([]nlp.UnigramPiece{
		{Piece: "<unk>", Score: 0},
		{Piece: "▁hello", Score: -1},
		{Piece: "▁", Score: -2},
	})
	if err != nil {
		panic(err)
	}
	data, err := u.ToJSON()
	if err != nil {
		panic(err)
	}
	back, err := nlp.UnigramFromJSON(data)
	if err != nil {
		panic(err)
	}
	fmt.Println(u.Encode("hello"), back.Encode("hello"))
	// Output: [1] [1]
}

// ToJSON serializes a WordPiece tokenizer to the HuggingFace tokenizer.json form
// (model.type "WordPiece" with vocab, "##" continuation prefix and unk token), loadable
// again with WordPieceFromJSON. The round trip preserves the greedy longest-match
// split: "unable" is still "un" + "##able" (ids 1, 2).
func ExampleWordPiece_ToJSON() {
	w, err := nlp.NewWordPiece([]string{"[UNK]", "un", "##able"})
	if err != nil {
		panic(err)
	}
	data, err := w.ToJSON()
	if err != nil {
		panic(err)
	}
	back, err := nlp.WordPieceFromJSON(data)
	if err != nil {
		panic(err)
	}
	fmt.Println(w.Encode("unable"), back.Encode("unable"))
	// Output: [1 2] [1 2]
}
