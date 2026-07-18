package nlp_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// TestLlamaCppInteropRequiredMetadata pins the metadata contract a GoAI-written
// llama-arch GGUF must satisfy for REAL llama.cpp to load it — the key set was
// established empirically by scripts/interop_llamacpp.sh (a GoAI-written file
// loads and generates in llama.cpp, and its F32 greedy decode is byte-identical
// to GoAI's over 9 tokens; llama.cpp commit 86a9c79, 2026-07-17) and read from
// llama.cpp source:
//
//   - src/llama-model.cpp llama_model_base::load_hparams — required (get_key
//     with required=true throws): general.architecture, <arch>.context_length,
//     <arch>.embedding_length, <arch>.block_count. Optional-but-shape-checked:
//     <arch>.feed_forward_length, <arch>.attention.head_count (default 0 breaks
//     tensor creation), <arch>.attention.head_count_kv (defaults to head_count),
//     <arch>.rope.freq_base (defaults 10000).
//   - src/models/llama.cpp llama_model_llama::load_arch_hparams — required:
//     llama.attention.layer_norm_rms_epsilon.
//   - src/llama-vocab.cpp llama_vocab::impl::load — required for ANY model file:
//     tokenizer.ggml.model ("gpt2" → BPE) and tokenizer.ggml.tokens ("cannot
//     find tokenizer vocab"); required for BPE: tokenizer.ggml.merges ("cannot
//     find tokenizer merges"). n_vocab is len(tokens), so it must equal the
//     model's Vocab. Optional: tokenizer.ggml.pre (warns + "default"),
//     bos/eos_token_id, add_bos_token, token_type, scores.
//
// The tokenizer keys are the CALLER's data (LlamaToGGUF cannot synthesize a
// vocab), so this test exercises the documented recipe — LlamaToGGUF output
// plus the tokenizer.ggml.* keys — through a real gguf.Write→gguf.Read cycle
// and asserts the required keys survive with llama.cpp-compatible GGUF types.
// No llama.cpp needed at test time; the live cross-check is the script.
func TestLlamaCppInteropRequiredMetadata(t *testing.T) {
	t.Parallel()
	model, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 9, Ctx: 64, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 64,
		Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	meta, tensors := nlp.LlamaToGGUF(model)

	// The docs/gguf.md "Exporting to llama.cpp" caller recipe (vocab len == Vocab).
	meta["tokenizer.ggml.model"] = "gpt2"
	meta["tokenizer.ggml.tokens"] = []any{"h", "e", "l", "o", "he", "ll", "hello", "hell", "Ġ"}
	meta["tokenizer.ggml.merges"] = []any{"h e", "l l", "he ll", "hell o"}
	meta["tokenizer.ggml.pre"] = "gpt-2"
	meta["tokenizer.ggml.bos_token_id"] = uint32(0)
	meta["tokenizer.ggml.eos_token_id"] = uint32(0)
	meta["tokenizer.ggml.add_bos_token"] = false

	var buf bytes.Buffer
	if err := gguf.Write(&buf, &gguf.File{Version: 3, Metadata: meta, Tensors: tensors}); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Read(&buf)
	if err != nil {
		t.Fatal(err)
	}

	// Required keys with the dynamic Go types gguf.Read yields for the GGUF
	// types llama.cpp accepts there (get_key coerces within integer kinds; the
	// uint32/float32/string/bool/[]any-of-string forms below are what a real
	// converter emits and what GoAI writes).
	wantType := map[string]any{
		"general.architecture":                   "",         // string, required
		"llama.context_length":                   uint32(0),  // required
		"llama.embedding_length":                 uint32(0),  // required
		"llama.block_count":                      uint32(0),  // required
		"llama.attention.layer_norm_rms_epsilon": float32(0), // required (models/llama.cpp)
		"llama.feed_forward_length":              uint32(0),  // shape-critical
		"llama.attention.head_count":             uint32(0),  // shape-critical
		"llama.attention.head_count_kv":          uint32(0),  // GQA-critical
		"llama.rope.freq_base":                   float32(0), // semantics-critical
		"tokenizer.ggml.model":                   "",         // required (vocab)
		"tokenizer.ggml.tokens":                  []any(nil), // required (vocab)
		"tokenizer.ggml.merges":                  []any(nil), // required for BPE
		"tokenizer.ggml.bos_token_id":            uint32(0),  // recommended
		"tokenizer.ggml.eos_token_id":            uint32(0),  // recommended
		"tokenizer.ggml.add_bos_token":           false,      // recommended
	}
	for key, want := range wantType {
		got, ok := f.Metadata[key]
		if !ok {
			t.Errorf("exported GGUF missing llama.cpp key %s", key)
			continue
		}
		if fmt.Sprintf("%T", got) != fmt.Sprintf("%T", want) {
			t.Errorf("%s: got type %T, want %T", key, got, want)
		}
	}
	if a := f.Metadata["general.architecture"]; a != "llama" {
		t.Errorf("general.architecture = %v, want llama", a)
	}
	// llama.cpp derives n_vocab from len(tokenizer.ggml.tokens) and creates
	// token_embd as {n_embd, n_vocab} — a length mismatch is a load failure.
	toks := f.Metadata["tokenizer.ggml.tokens"].([]any)
	if len(toks) != model.Config.Vocab {
		t.Errorf("len(tokens) = %d, want model Vocab %d", len(toks), model.Config.Vocab)
	}
	for i, tok := range toks {
		if _, ok := tok.(string); !ok {
			t.Fatalf("tokens[%d] is %T, want string", i, tok)
		}
	}
	for i, m := range f.Metadata["tokenizer.ggml.merges"].([]any) {
		if _, ok := m.(string); !ok {
			t.Fatalf("merges[%d] is %T, want string", i, m)
		}
	}

	// The llama-arch tensor set src/models/llama.cpp load_arch_tensors demands
	// (output.weight is optional — absent means tied head — but GoAI writes it).
	wantTensors := []string{"token_embd.weight", "output_norm.weight", "output.weight"}
	for l := 0; l < model.Config.Layers; l++ {
		for _, s := range []string{"attn_norm", "attn_q", "attn_k", "attn_v",
			"attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"} {
			wantTensors = append(wantTensors, fmt.Sprintf("blk.%d.%s.weight", l, s))
		}
	}
	for _, name := range wantTensors {
		if _, ok := f.Tensors[name]; !ok {
			t.Errorf("exported GGUF missing tensor %s", name)
		}
	}
}
