// Command exportgguf writes the benchmark llama (the exact model
// internal/benchcompare's decode benchmarks measure) as a llama.cpp-loadable
// GGUF file, so the SAME weights can be timed by llama.cpp for the
// head-to-head comparison in docs/benchmarking.md (SPEC T607).
//
// In plain terms: it saves our benchmark model in the file format llama.cpp
// understands, tokenizer section included, so both engines can race on
// identical weights.
//
// Usage: go run ./internal/benchcompare/exportgguf <out.gguf>
package main

import (
	"fmt"
	"os"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: exportgguf <out.gguf>")
		os.Exit(2)
	}
	const vocab = 1024
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: vocab, Ctx: 128, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}, 1) // seed 1 = the decode benchmark's model
	if err != nil {
		panic(err)
	}
	meta, ts := nlp.LlamaToGGUF(m)

	// llama.cpp refuses a model without a tokenizer section. The benchmark
	// vocabulary is synthetic, so emit a minimal SPM-style (SentencePiece
	// model) vocab: <unk>/<s>/</s> plus filler pieces — timing needs ids, not
	// linguistics.
	tokens := make([]any, vocab)
	scores := make([]any, vocab)
	types := make([]any, vocab)
	for i := range vocab {
		tokens[i] = fmt.Sprintf("<t%04d>", i)
		scores[i] = float32(0)
		types[i] = int32(1) // NORMAL
	}
	tokens[0], types[0] = "<unk>", int32(2) // UNKNOWN
	tokens[1], types[1] = "<s>", int32(3)   // CONTROL
	tokens[2], types[2] = "</s>", int32(3)  // CONTROL
	meta["tokenizer.ggml.model"] = "llama"
	meta["tokenizer.ggml.tokens"] = tokens
	meta["tokenizer.ggml.scores"] = scores
	meta["tokenizer.ggml.token_type"] = types
	meta["tokenizer.ggml.bos_token_id"] = uint32(1)
	meta["tokenizer.ggml.eos_token_id"] = uint32(2)
	meta["tokenizer.ggml.unknown_token_id"] = uint32(0)
	meta["general.name"] = "goai-bench-llama"

	if err := gguf.WriteFile(os.Args[1], &gguf.File{Version: 3, Metadata: meta, Tensors: ts}); err != nil {
		panic(err)
	}
	fi, _ := os.Stat(os.Args[1])
	fmt.Printf("wrote %s (%.1f MB, vocab %d, dim 512, layers 6)\n", os.Args[1], float64(fi.Size())/1e6, vocab)
}
