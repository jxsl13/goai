package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// mambaHF loads the golden Mamba checkpoint used across the decode tests.
func mambaHF(t *testing.T) *nlp.Mamba {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/mamba_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.MambaFromHF(ts, nlp.MambaConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("MambaFromHF: %v", err)
	}
	return m
}

// §V16 tier-1: the recurrent Mamba decode is lossless — stepping a prompt
// through DecodeStep yields the SAME next-token logits as a full Forward over
// that prompt. The gate is bit-identical (<1e-9) because parallel-scan mode and
// recurrent mode are the same recurrence: the conv window replays the OpConv1D
// kernel's taps (same ascending-k accumulation over the last d_conv inputs) and
// the SSM step replays the OpSSM kernel's per-(d,n) loop against the same
// persistent f64 hidden state, while every other sublayer (RMSNorm, matmul,
// SiLU, softplus) is row-independent. This is the correctness contract of the
// O(1) recurrent state.
func TestMambaDecodeMatchesForward(t *testing.T) {
	m := mambaHF(t)
	prompt := []int{3, 7, 1, 9, 4, 2, 8}

	full, err := m.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	seq, vocab := full.Shape()[0], full.Shape()[1]

	ctx := backend.NewContext()
	st := m.NewDecodeState()
	var last *tensor.Tensor
	for _, tok := range prompt {
		if last, err = m.DecodeStep(ctx, st, tok); err != nil {
			t.Fatal(err)
		}
	}
	var maxAbs float64
	for j := range vocab {
		if d := math.Abs(last.AtF64(0, j) - full.AtF64(seq-1, j)); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("Mamba decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("Mamba decode diverges from Forward: %.3e (the recurrence must be exact)", maxAbs)
	}
}

// A greedy Generate over the recurrent state returns prompt+maxNew tokens
// without error, and its first sampled token agrees with argmax-ing a full
// Forward over the prompt (greedy decode ≡ greedy Forward).
func TestMambaGenerateGreedyRuns(t *testing.T) {
	m := mambaHF(t)
	prompt := []int{3, 7, 1}
	const n = 3

	out, err := m.Generate(prompt, n, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != len(prompt)+n {
		t.Fatalf("Generate produced %d tokens, want %d (%d prompt + %d)", len(out), len(prompt)+n, len(prompt), n)
	}
	for i, tok := range out[:len(prompt)] {
		if tok != prompt[i] {
			t.Fatalf("Generate mutated the prompt at %d: got %d, want %d", i, tok, prompt[i])
		}
	}

	full, err := m.Forward(backend.NewContext(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	last := full.Shape()[0] - 1
	best, bestV := 0, math.Inf(-1)
	for j := range full.Shape()[1] {
		if v := full.AtF64(last, j); v > bestV {
			best, bestV = j, v
		}
	}
	if out[len(prompt)] != best {
		t.Fatalf("greedy Generate first token %d != Forward argmax %d", out[len(prompt)], best)
	}
}

// The recurrent advantage vs a KV-cache, documented as a test: the decode
// state does NOT grow with the number of steps. After 3 tokens and after 7
// tokens every per-layer buffer has the same length — the conv window stays
// (d_conv−1)·d_inner and the SSM hidden state stays d_inner·N, where a
// transformer's KV-cache would have grown from 3 to 7 rows per layer.
func TestMambaDecodeStateIsO1(t *testing.T) {
	m := mambaHF(t)

	sizes := func(steps int) [][2]int {
		st := m.NewDecodeState()
		ctx := backend.NewContext()
		tokens := []int{3, 7, 1, 9, 4, 2, 8}[:steps]
		for _, tok := range tokens {
			if _, err := m.DecodeStep(ctx, st, tok); err != nil {
				t.Fatal(err)
			}
		}
		out := make([][2]int, len(st.Layers))
		for l, s := range st.Layers {
			out[l] = [2]int{len(s.ConvBuf), len(s.H)}
		}
		return out
	}

	after3, after7 := sizes(3), sizes(7)
	if len(after3) != len(m.Layers) || len(after7) != len(m.Layers) {
		t.Fatalf("state layer count changed: %d vs %d (layers %d)", len(after3), len(after7), len(m.Layers))
	}
	for l := range after3 {
		if after3[l] != after7[l] {
			t.Fatalf("layer %d state grew with steps: %v after 3 vs %v after 7", l, after3[l], after7[l])
		}
		mb := m.Layers[l].Mixer
		if want := (mb.DConv - 1) * mb.DInner; after3[l][0] != want {
			t.Fatalf("layer %d conv window has %d elements, want (d_conv−1)·d_inner=%d", l, after3[l][0], want)
		}
		if want := mb.DInner * mb.N; after3[l][1] != want {
			t.Fatalf("layer %d SSM state has %d elements, want d_inner·N=%d", l, after3[l][1], want)
		}
	}
}
