package nlp_test

import (
	"bytes"
	"fmt"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// ExampleLlamaFromGGUF_pipeline is the README quickstart as a runnable, self-contained
// test: a single GGUF file carries BOTH the model weights and the tokenizer, and one
// load call each yields a model and a tokenizer that generate text end to end. Here the
// file is built in memory (LlamaToGGUF + the tokenizer.ggml.* metadata) instead of read
// from disk, so the example needs no external model — but the load → tokenize → generate
// → decode path is exactly the one a real .gguf download follows, which keeps the
// front-page pipeline honest.
func ExampleLlamaFromGGUF_pipeline() {
	// A tiny model whose vocab matches the embedded tokenizer (ids 0..8).
	model, _ := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 9, Ctx: 32, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 64,
		Eps: 1e-5, RopeBase: 10000,
	}, 7)

	// Serialize the model, then fold in the GPT-2/BPE tokenizer metadata — a real
	// llama.cpp GGUF carries both in the same file.
	meta, tensors := nlp.LlamaToGGUF(model)
	toks := []any{"h", "e", "l", "o", "he", "ll", "hello", "hell", "Ġ"}
	merges := []any{"h e", "l l", "he ll", "hell o"}
	meta["tokenizer.ggml.model"] = "gpt2"
	meta["tokenizer.ggml.tokens"] = toks
	meta["tokenizer.ggml.merges"] = merges

	var buf bytes.Buffer
	if err := gguf.Write(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: tensors}); err != nil {
		panic(err)
	}

	// The consumer side: open the file, load the model and the tokenizer from it.
	f, err := gguf.Read(&buf)
	if err != nil {
		panic(err)
	}
	lm, err := nlp.LlamaFromGGUF(f.Metadata, f.Tensors)
	if err != nil {
		panic(err)
	}
	tok, err := nlp.BPEFromGGUF(f.Metadata)
	if err != nil {
		panic(err)
	}

	// Tokenize a prompt, generate deterministically (greedy), decode back to text.
	prompt := tok.Encode("hello")
	out, err := lm.Generate(prompt, 3, nlp.Greedy())
	if err != nil {
		panic(err)
	}
	text := tok.Decode(out)

	fmt.Printf("prompt ids: %v\n", prompt)
	fmt.Printf("generated %d tokens total\n", len(out))
	fmt.Printf("decode∘encode round-trips: %v\n", tok.Decode(tok.Encode("hello")) == "hello")
	fmt.Printf("output is text: %v\n", len(text) > 0)
	// Output:
	// prompt ids: [6]
	// generated 4 tokens total
	// decode∘encode round-trips: true
	// output is text: true
}
