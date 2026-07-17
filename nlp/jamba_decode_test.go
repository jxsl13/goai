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

// jambaGolden loads the golden Jamba checkpoint used across the decode tests
// (4 layers: 0,2 Mamba+MoE, 1,3 NoPE-attention+dense — all four component
// families in one stack).
func jambaGolden(t *testing.T) *nlp.Jamba {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/jamba_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.JambaFromHF(ts, nlp.JambaConfig{Heads: 4, TopK: 2, Eps: 1e-6})
	if err != nil {
		t.Fatalf("JambaFromHF: %v", err)
	}
	return m
}

// §V16 tier-1: the hybrid Jamba decode is lossless — stepping a prompt through
// DecodeStep yields the SAME next-token logits as a full Forward over that
// prompt. The gate is bit-identical (<1e-9) because every layer kind is exact
// in single-token mode: the NoPE attention's single query over the cached
// un-rotated keys (Causal:false) is row t of the causal Forward attention, the
// Mamba step replays the OpConv1D taps and the ref OpSSM kernel's f64 loop
// with the dt/b/c norms in Forward's spots, and the MoE/dense FFNs are
// stateless and row-independent (raw top-k routing included). The golden stack
// interleaves all four mixer/FFN families, so one parity check covers every
// combination.
func TestJambaDecodeMatchesForward(t *testing.T) {
	m := jambaGolden(t)
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
	t.Logf("Jamba decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("Jamba decode diverges from Forward: %.3e (the hybrid step must be exact)", maxAbs)
	}
}

// A greedy Generate over the hybrid state returns prompt+maxNew tokens without
// error, and its first sampled token agrees with argmax-ing a full Forward
// over the prompt (greedy decode ≡ greedy Forward).
func TestJambaGenerateGreedyRuns(t *testing.T) {
	m := jambaGolden(t)
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

// The hybrid memory profile, documented as a test: after 3 steps and after 7
// steps every MAMBA layer's state has the same size — the conv window stays
// (d_conv−1)·d_inner and the SSM hidden state stays d_inner·N — while every
// ATTENTION layer's KV-cache has grown from exactly 3 to exactly 7 rows (one
// per token). Each layer holds exactly one kind of state and never the other.
// This is Jamba's cache-size trade in miniature: total decode memory grows
// only with the attention layers.
func TestJambaDecodeStateHybridProfile(t *testing.T) {
	m := jambaGolden(t)

	run := func(steps int) *nlp.JambaDecodeState {
		st := m.NewDecodeState()
		ctx := backend.NewContext()
		for _, tok := range []int{3, 7, 1, 9, 4, 2, 8}[:steps] {
			if _, err := m.DecodeStep(ctx, st, tok); err != nil {
				t.Fatal(err)
			}
		}
		return st
	}

	after3, after7 := run(3), run(7)
	for l, layer := range m.Layers {
		if layer.Attn != nil {
			// Attention layer: KV grows by one row per step, no Mamba state.
			if after3.Mamba[l] != nil || after7.Mamba[l] != nil {
				t.Fatalf("attention layer %d carries Mamba state", l)
			}
			if got := after3.K[l].Shape()[0]; got != 3 {
				t.Fatalf("attention layer %d cached %d key rows after 3 steps, want 3", l, got)
			}
			if got := after7.K[l].Shape()[0]; got != 7 {
				t.Fatalf("attention layer %d cached %d key rows after 7 steps, want 7", l, got)
			}
			if got := after7.V[l].Shape()[0]; got != 7 {
				t.Fatalf("attention layer %d cached %d value rows after 7 steps, want 7", l, got)
			}
			continue
		}
		// Mamba layer: constant-size state, no KV rows ever.
		if after3.K[l] != nil || after7.K[l] != nil || after3.V[l] != nil || after7.V[l] != nil {
			t.Fatalf("mamba layer %d accumulated KV rows", l)
		}
		s3, s7 := after3.Mamba[l], after7.Mamba[l]
		if s3 == nil || s7 == nil {
			t.Fatalf("mamba layer %d has no recurrent state", l)
		}
		if len(s3.ConvBuf) != len(s7.ConvBuf) || len(s3.H) != len(s7.H) {
			t.Fatalf("mamba layer %d state grew with steps: conv %d→%d, h %d→%d",
				l, len(s3.ConvBuf), len(s7.ConvBuf), len(s3.H), len(s7.H))
		}
		mb := layer.Mamba.Block
		if want := (mb.DConv - 1) * mb.DInner; len(s7.ConvBuf) != want {
			t.Fatalf("mamba layer %d conv window has %d elements, want (d_conv−1)·d_inner=%d", l, len(s7.ConvBuf), want)
		}
		if want := mb.DInner * mb.N; len(s7.H) != want {
			t.Fatalf("mamba layer %d SSM state has %d elements, want d_inner·N=%d", l, len(s7.H), want)
		}
	}
	if after7.Len() != 7 {
		t.Fatalf("state Len() = %d after 7 steps, want 7", after7.Len())
	}
}
