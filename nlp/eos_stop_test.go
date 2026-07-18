package nlp_test

import (
	"slices"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// §B76 — EOS stopping in the Generate loops.
//
// These tests drive the decode loops with a SCRIPTED sampler rather than the model's
// own logits. The stop decision (genConfig.stopEOS) acts on the token the sampler
// returned, so scripting that sequence exercises the stop contract exactly while
// keeping the assertions independent of what a randomly-initialized tiny model
// happens to predict — the loop control flow is what is under test, not the weights.

// scriptSampler returns a fixed token sequence, ignoring the logits entirely. Once
// the script runs out it repeats the final token, which is what makes the
// "post-EOS filler" of an unstopped loop visible in the output.
type scriptSampler struct {
	toks []int
	i    int
}

func (s *scriptSampler) next() int {
	if s.i >= len(s.toks) {
		return s.toks[len(s.toks)-1]
	}
	t := s.toks[s.i]
	s.i++
	return t
}

func (s *scriptSampler) Sample([]float64) int                   { return s.next() }
func (s *scriptSampler) SampleWithHistory([]float64, []int) int { return s.next() }

// eosLlama builds a small but roomy model: Ctx is deliberately far larger than any
// maxNew used here so the context-limit break can never be mistaken for an EOS stop.
// Dim and Hidden are multiples of 32 so the same model also quantizes to Q8_0
// (whose block size is 32) for the Quant* half of these tests.
func eosLlama(t *testing.T) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 8, Ctx: 64, Dim: 32, Heads: 4, KVHeads: 2, Layers: 2, Hidden: 32, Eps: 1e-5, RopeBase: 10000,
	}, 7)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func eosQuantLlama(t *testing.T) *nlp.QuantLlama {
	t.Helper()
	q, err := nlp.QuantizeLlama(eosLlama(t), gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	return q
}

// A model drawing EOS as the k-th generated token returns prompt+k tokens, the last
// of which IS the EOS — the documented include-the-stop-token contract (HF/llama.cpp
// and this package's own BeamSearch). Covers the FLOAT decode path.
func TestGenerateWithEOSStopsFloat(t *testing.T) {
	m := eosLlama(t)
	prompt := []int{1, 2}
	const eos = 4
	// EOS is the 3rd generated token; tokens after it would be filler.
	script := []int{3, 5, eos, 6, 6, 6, 6, 6, 6, 6}

	out, err := m.Generate(prompt, 10, &scriptSampler{toks: script}, nlp.WithEOS(eos))
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 2, 3, 5, eos}
	if !slices.Equal(out, want) {
		t.Fatalf("Generate WithEOS(%d) = %v, want %v (stop at the EOS, EOS included)", eos, out, want)
	}
}

// The Quant* twin is a SEPARATE code path (its own Generate, its own loop), so the
// same contract is asserted against it independently.
func TestGenerateWithEOSStopsQuant(t *testing.T) {
	q := eosQuantLlama(t)
	prompt := []int{1, 2}
	const eos = 4
	script := []int{3, 5, eos, 6, 6, 6, 6, 6, 6, 6}

	out, err := q.Generate(prompt, 10, &scriptSampler{toks: script}, nlp.WithEOS(eos))
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 2, 3, 5, eos}
	if !slices.Equal(out, want) {
		t.Fatalf("QuantLlama.Generate WithEOS(%d) = %v, want %v", eos, out, want)
	}
}

// THE REGRESSION GATE THAT MATTERS MOST: without the option nothing changes. The
// loop draws the full maxNew tokens even though the script contains what a
// configured caller would call an EOS — the pre-existing behaviour, byte for byte.
// Passing WithEOS for an id that is never drawn must likewise not perturb anything,
// proving the option's mere presence does not alter the sampling path.
func TestGenerateEOSDefaultUnchanged(t *testing.T) {
	for _, tc := range []struct {
		name string
		gen  func(s nlp.TokenSampler, opts ...nlp.GenerateOption) ([]int, error)
	}{
		{"float", func(s nlp.TokenSampler, o ...nlp.GenerateOption) ([]int, error) {
			return eosLlama(t).Generate([]int{1, 2}, 6, s, o...)
		}},
		{"quant", func(s nlp.TokenSampler, o ...nlp.GenerateOption) ([]int, error) {
			return eosQuantLlama(t).Generate([]int{1, 2}, 6, s, o...)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := []int{3, 5, 4, 6, 6, 6}
			// No option: the full budget, INCLUDING everything after the 4.
			base, err := tc.gen(&scriptSampler{toks: script})
			if err != nil {
				t.Fatal(err)
			}
			want := []int{1, 2, 3, 5, 4, 6, 6, 6}
			if !slices.Equal(base, want) {
				t.Fatalf("default Generate = %v, want %v (no option ⇒ exactly maxNew tokens)", base, want)
			}
			// An armed-but-never-drawn stop id changes nothing.
			armed, err := tc.gen(&scriptSampler{toks: script}, nlp.WithEOS(7))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(armed, base) {
				t.Fatalf("WithEOS(7) (never drawn) = %v, want identical to default %v", armed, base)
			}
		})
	}
}

// Multiple stop ids: models routinely ship two or more (a base </s> and a chat
// <|im_end|>), and EACH must end the loop on its own.
func TestGenerateWithEOSMultipleIDs(t *testing.T) {
	prompt := []int{1, 2}
	for _, tc := range []struct {
		name string
		stop int
		want []int
	}{
		{"first id", 4, []int{1, 2, 3, 4}},
		{"second id", 6, []int{1, 2, 3, 6}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			script := []int{3, tc.stop, 5, 5, 5, 5}
			out, err := eosLlama(t).Generate(prompt, 6, &scriptSampler{toks: script}, nlp.WithEOS(4, 6))
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(out, tc.want) {
				t.Fatalf("WithEOS(4, 6) stopping on %d = %v, want %v", tc.stop, out, tc.want)
			}
		})
	}
}

// The concrete harm from the report: a guided sampler reaches an accepting state,
// draws EOS, freezes its FSM — and the loop keeps going, returning maxNew tokens of
// filler drawn from the frozen state with err == nil.
//
// Two halves, matching the two documented remedies:
//   - WithEOS() (no explicit ids) picks the guide's own eos up through StopTokener
//     and the filler is GONE.
//   - Without the option the filler is still produced (default behaviour is
//     preserved), but it is now REPORTABLE via EOSReporter instead of silent.
func TestGuidedSamplerEOSNoLongerSilentFiller(t *testing.T) {
	vocab := []string{"a", "b", "c", "d", "e", "f", "g", "h"}
	g, err := nlp.NewRegexGuide("ab+", vocab)
	if err != nil {
		t.Fatal(err)
	}
	const eos = 7
	prompt := []int{1, 2}
	// "a"(0), "b"(1) reaches an accepting state; then EOS; then filler.
	script := []int{0, 1, eos, 1, 1, 1, 1, 1}

	t.Run("WithEOS stops the loop", func(t *testing.T) {
		s := g.Sampler(&scriptSampler{toks: script}, eos)
		out, err := eosLlama(t).Generate(prompt, 8, s, nlp.WithEOS())
		if err != nil {
			t.Fatal(err)
		}
		want := []int{1, 2, 0, 1, eos}
		if !slices.Equal(out, want) {
			t.Fatalf("guided Generate WithEOS() = %v, want %v — post-EOS filler must be gone", out, want)
		}
	})

	t.Run("without the option the overrun is detectable", func(t *testing.T) {
		s := g.Sampler(&scriptSampler{toks: script}, eos)
		out, err := eosLlama(t).Generate(prompt, 8, s)
		if err != nil {
			t.Fatal(err)
		}
		// Default behaviour is deliberately unchanged: still the full budget.
		if len(out) != len(prompt)+8 {
			t.Fatalf("default guided Generate returned %d tokens, want %d (default must not change)",
				len(out), len(prompt)+8)
		}
		r, ok := s.(nlp.EOSReporter)
		if !ok {
			t.Fatal("guided sampler does not implement nlp.EOSReporter — the overrun is undetectable")
		}
		if !r.EOSEmitted() {
			t.Error("EOSEmitted() = false after the guide drew EOS; the filler tail is unreportable")
		}
	})
}

// The guided sampler advertises its eos through StopTokener so callers need not
// repeat an id the sampler already holds; a guide built with eos < 0 advertises none.
func TestGuidedSamplerStopTokens(t *testing.T) {
	vocab := []string{"a", "b", "c", "d"}
	g, err := nlp.NewRegexGuide("ab+", vocab)
	if err != nil {
		t.Fatal(err)
	}
	st, ok := g.Sampler(nlp.Greedy(), 3).(nlp.StopTokener)
	if !ok {
		t.Fatal("guided sampler does not implement nlp.StopTokener")
	}
	if got := st.StopTokens(); !slices.Equal(got, []int{3}) {
		t.Errorf("StopTokens() = %v, want [3]", got)
	}
	none, _ := g.Sampler(nlp.Greedy(), -1).(nlp.StopTokener)
	if got := none.StopTokens(); len(got) != 0 {
		t.Errorf("StopTokens() with eos<0 = %v, want none", got)
	}
}
