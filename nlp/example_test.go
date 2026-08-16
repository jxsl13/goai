package nlp_test

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// Speculative decoding verifies a whole run of draft tokens in one target pass.
// Here the target fully agrees with the draft (matching one-hot distributions),
// so all K=3 proposals are accepted and a bonus 4th token is appended — up to
// K+1 tokens produced from a single target forward pass, with the output
// distributed exactly as the target model would sample.
func ExampleSpeculativeRun() {
	oh := func(at int) []float64 { d := make([]float64, 5); d[at] = 1; return d }
	target := [][]float64{oh(1), oh(2), oh(3), oh(4)} // K+1 = 4 target positions
	draft := [][]float64{oh(1), oh(2), oh(3)}         // K = 3 draft positions
	rng := rand.New(rand.NewPCG(1, 1))

	out := nlp.SpeculativeRun(target, draft, []int{1, 2, 3}, rng)
	fmt.Println(out)
	// Output: [1 2 3 4]
}

// --- Level 1: trivial ---------------------------------------------------------

// Greedy decoding always picks the highest-scoring token — deterministic, no RNG.
// It is the simplest decoder and the baseline the others are measured against.
func ExampleGreedy() {
	logits := []float64{0.1, 2.5, 0.3} // token 1 has the largest score
	tok := nlp.Greedy().Sample(logits)
	fmt.Println(tok)
	// Output: 1
}

// SpeculativeSample is the single-token core of speculative decoding: accept the
// drafted token with probability min(1, p/q), else resample the residual. When the
// draft distribution q equals the target p (here both concentrate on token 1),
// p/q = 1, so the draft is always accepted and the output is exactly the target's.
func ExampleSpeculativeSample() {
	target := []float64{0.2, 0.8} // large "target" model's distribution
	draft := []float64{0.2, 0.8}  // small "draft" model happens to agree
	rng := rand.New(rand.NewPCG(1, 1))

	tok, accepted := nlp.SpeculativeSample(target, draft, 1 /*drafted token*/, rng)
	fmt.Println(tok, accepted)
	// Output: 1 true
}

// --- Level 2: realistic use case ----------------------------------------------

// A Sampler runs the standard temperature → top-k → top-p → min-p → multinomial
// pipeline. Here a sharply peaked distribution plus a top-p nucleus of 0.9 leaves
// only the dominant token inside the nucleus, so sampling is effectively pinned to
// it regardless of the seed — how nucleus sampling stays on-topic when the model
// is confident while still allowing variety when it is not.
func ExampleSampler() {
	s := nlp.NewSampler(42, nlp.WithTopP(0.9))
	logits := []float64{6.0, 1.0, 0.5} // softmax ≈ 0.989 on token 0 → nucleus = {0}
	fmt.Println(s.Sample(logits))
	// Output: 0
}

// ExampleSampler_SampleTopPFromCandidates draws from a GPU TopK shortlist instead of the full
// vocabulary, and gets the SAME token a full-vocab Sample would. The trick is that the caller also
// passes the full-vocab softmax statistics (maxLogit and Zexp), so each candidate keeps its true
// probability and the nucleus is the real one — sampling the shortlist with its own normalizer
// would inflate the masses and cut the nucleus short. The second return reports overflow: the
// nucleus needed more tokens than the shortlist held, so fall back to Sample.
func ExampleSampler_SampleTopPFromCandidates() {
	logits := []float64{6.0, 1.0, 0.5} // the full vocabulary
	// Full-vocab statistics, as a device TopK kernel would return alongside the shortlist.
	maxLogit := 6.0
	Zexp := 0.0
	for _, l := range logits {
		Zexp += math.Exp(l - maxLogit)
	}

	// The shortlist: the top 2 of 3, with their vocabulary indices.
	candLogits, candIdx := []float64{6.0, 1.0}, []int32{0, 1}

	tok, ok := nlp.NewSampler(42, nlp.WithTopP(0.9)).SampleTopPFromCandidates(candLogits, candIdx, maxLogit, Zexp)
	full := nlp.NewSampler(42, nlp.WithTopP(0.9)).Sample(logits)
	fmt.Println(tok, ok, tok == full)
	// Output: 0 true true
}

// --- Level 3: embedded in a larger decode step --------------------------------

// A realistic decode step chains a logit transform with a decoder: penalize tokens
// already produced, then choose. Token 0 leads the raw logits, but it was already
// generated, so a repetition penalty of 1.5 divides its logit down (3.0 → 2.0) and
// the greedy choice shifts to token 1 — anti-repetition steering in two lines.
func Example_penaltyThenDecode() {
	logits := []float64{3.0, 2.9, 0.5}
	generated := []int{0} // token 0 already emitted

	nlp.ApplyPenalties(logits, generated, 1.5 /*repetition*/, 0, 0)
	next := nlp.Greedy().Sample(logits)
	fmt.Println(next)
	// Output: 1
}

// Beam search keeps the `width` best partial hypotheses and returns the
// highest-total-log-probability sequence — often better than greedy, which can be
// trapped by a locally-best first token. Here a tiny model (next-token logits
// depend on the last token) makes greedy pick [0,0,…] while beam width 2 finds the
// higher-scoring [0,1,0].
func ExampleBeamSearch() {
	M := [][]float64{{2.0, 1.9}, {5.0, 0.0}}
	next := func(prefix []int) []float64 { return M[prefix[len(prefix)-1]] }

	beams := nlp.BeamSearch(next, []int{0}, 2 /*width*/, 2 /*maxNew*/, -1 /*no eos*/, 0 /*no length penalty*/)
	fmt.Println(beams[0].Tokens)
	// Output: [0 1 0]
}

// ApplyPenalties discourages repetition before sampling. Token 0 was already
// generated, so a repetition penalty of 1.5 divides its (positive) logit by 1.5,
// lowering the chance of picking it again; the other tokens are untouched.
func ExampleApplyPenalties() {
	logits := []float64{3.0, 1.0, 0.5}
	nlp.ApplyPenalties(logits, []int{0}, 1.5 /*repetition*/, 0 /*frequency*/, 0 /*presence*/)
	fmt.Printf("%.4f %.1f %.1f\n", logits[0], logits[1], logits[2])
	// Output: 2.0000 1.0 0.5
}

// TokenSampler is the interface every sequential generation loop accepts — both the
// truncation-based Sampler and the adaptive Mirostat satisfy it, so either plugs into
// GPT/Llama Generate (or a llamagpu decoder) interchangeably. Greedy argmax picks
// token 0; a repeat penalty on the same history steers it to the runner-up.
func ExampleTokenSampler() {
	logits := []float64{2.0, 1.5, 0.1}
	var s nlp.TokenSampler = nlp.Greedy()
	fmt.Println(s.SampleWithHistory(logits, nil))

	s = nlp.NewSampler(1, nlp.WithTemperature(0), nlp.WithRepeatPenalty(2))
	fmt.Println(s.SampleWithHistory(logits, []int{0}))

	s = nlp.NewMirostat(7) // adaptive: also a TokenSampler
	tok := s.SampleWithHistory(logits, []int{0})
	fmt.Println(tok >= 0 && tok < len(logits))
	// Output:
	// 0
	// 1
	// true
}

// MedusaHeads are extra decoding heads over a frozen base model's final hidden state
// (ForwardHidden): head k predicts the token at offset t+2+k, enabling multi-token
// drafting. Here two heads project a hidden batch into per-head logits.
func ExampleMedusaHeads() {
	heads, err := nlp.NewMedusaHeads(2 /*K*/, 4 /*dim*/, 6 /*vocab*/, 1 /*seed*/)
	if err != nil {
		panic(err)
	}
	hidden := tensor.New(tensor.F32, tensor.Shape{3, 4}) // e.g. gpt.ForwardHidden(ctx, tokens)
	logits, err := heads.Logits(backend.NewContext(), hidden)
	if err != nil {
		panic(err)
	}
	fmt.Println(len(logits), logits[0].Shape(), logits[1].Shape())
	// Output: 2 (3, 6) (3, 6)
}
