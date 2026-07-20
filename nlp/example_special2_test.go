package nlp_test

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
)

// AddSpecialTokens registers marker text -> id pairs that EncodeSpecial parses as a
// single token instead of feeding them through the ordinary merge loop; plain
// Encode is unaffected. Every tokenizer in this package (BPETokenizer, SPM, Unigram,
// WordPiece) exposes the same pair of methods — see the package doc's special-token
// section (§B60) for the untrusted-input trust boundary EncodeSpecial's doc warns
// about.
func ExampleBPETokenizer_AddSpecialTokens() {
	tok, err := nlp.NewBPE([]string{"h", "i", "hi"}, []string{"h i"})
	if err != nil {
		panic(err)
	}
	tok.AddSpecialTokens(map[string]int{"<eos>": 3})
	fmt.Println(tok.EncodeSpecial("hi<eos>"))
	// Output: [2 3]
}

// SPM implements SentencePiece's rank-merge BPE — the tokenizer llama.cpp-family
// GGUF checkpoints ship (see [SPMFromGGUF]). Its scores are merge RANKS (more
// negative merges later), not log-probabilities, so encoding always merges the
// highest-scoring adjacent pair first rather than running Unigram's Viterbi search;
// [Unigram] and [SPM] can disagree on the very same vocabulary for exactly this
// reason (§B59).
func ExampleSPM() {
	vocab := []nlp.UnigramPiece{
		{Piece: "<unk>", Score: 0},
		{Piece: "▁", Score: -3},
		{Piece: "a", Score: -1},
		{Piece: "b", Score: -2},
		{Piece: "ab", Score: -4},
		{Piece: "▁ab", Score: -5},
	}
	tok, err := nlp.NewSPM(vocab)
	if err != nil {
		panic(err)
	}
	ids := tok.Encode("ab ab")
	fmt.Println(ids, tok.Decode(ids))
	// Output: [5 5] ab ab
}

// SpecialTokens returns the family's control markers (longest first, so a caller
// matching manually gets the same leftmost-longest resolution EncodeSpecial uses).
// Pair it with [RegisterChatSpecials] to resolve those markers against a specific
// tokenizer's vocabulary.
func ExampleChatTemplate_SpecialTokens() {
	tpl, err := nlp.NewChatTemplate("chatml")
	if err != nil {
		panic(err)
	}
	fmt.Println(tpl.SpecialTokens())
	// Output: [<|im_start|> <|im_end|>]
}

// DeepSeek-V2's absorbed-latent decode path: NewLatentCache allocates the small
// per-layer (KVLoraRank+QKRope)-wide cache, PrefillLatent seeds it from a prompt in
// one call, and DecodeStepLatent advances it one token at a time — all three
// mathematically identical to the reconstructed-cache [DeepSeekV2.DecodeStep] path
// but with a far smaller memory footprint (see [ExampleDeepSeekV2_GenerateLatent]
// for the high-level Generate wrapper over this same cache).
func ExampleDeepSeekV2_PrefillLatent() {
	ts, _, err := safetensors.LoadFile("testdata/deepseekv2moe_hf.safetensors")
	if err != nil {
		panic(err)
	}
	model, err := nlp.DeepSeekV2FromHF(ts, nlp.DeepSeekV2Config{
		Heads: 4, QLoraRank: 24, KVLoraRank: 16,
		QKNope: 16, QKRope: 8, VHead: 16,
		Eps: 1e-6, RopeBase: 10000, Ctx: 32,
		FirstKDense: 1, NRoutedExperts: 4, NSharedExperts: 1,
		TopK: 2, MoEHidden: 32, RoutedScale: 1.0,
	})
	if err != nil {
		panic(err)
	}
	ctx := backend.NewContext()
	cache := model.NewLatentCache()
	prompt := []int{3, 7, 1}
	if _, err = model.PrefillLatent(ctx, cache, prompt); err != nil {
		panic(err)
	}
	last, err := model.DecodeStepLatent(ctx, cache, 9, len(prompt))
	if err != nil {
		panic(err)
	}
	fmt.Println(last.Shape(), "cache len:", cache.Len())
	// Output: (1, 32) cache len: 4
}
