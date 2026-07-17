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

// rwkvHF loads the golden RWKV-4 checkpoint used across the decode tests.
func rwkvHF(t *testing.T) *nlp.RWKV {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/rwkv_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run make golden): %v", err)
	}
	m, err := nlp.RWKVFromHF(ts, nlp.RWKVConfig{Eps: 1e-5})
	if err != nil {
		t.Fatalf("RWKVFromHF: %v", err)
	}
	return m
}

// §V16 tier-1: the recurrent RWKV decode is lossless — stepping a prompt
// through DecodeStep yields the SAME next-token logits as a full Forward over
// that prompt. The gate is bit-identical (<1e-9) because RNN mode and parallel
// mode are the same recurrence: nn.RWKVBlock.Step replays the exact stabilized
// WKV update of the OpWKV kernel (same operation order per channel), the token
// shift is replaced by the carried previous row, and every other sublayer
// (LayerNorm, matmul) is row-independent. This is the correctness contract of
// the O(1) recurrent state.
func TestRWKVDecodeMatchesForward(t *testing.T) {
	m := rwkvHF(t)
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
	t.Logf("RWKV decode-vs-Forward max abs logit diff: %.3e", maxAbs)
	if maxAbs > 1e-9 {
		t.Fatalf("RWKV decode diverges from Forward: %.3e (the recurrence must be exact)", maxAbs)
	}
}

// A greedy Generate over the recurrent state returns prompt+maxNew tokens
// without error, and its first sampled token agrees with argmax-ing a full
// Forward over the prompt (greedy decode ≡ greedy Forward).
func TestRWKVGenerateGreedyRuns(t *testing.T) {
	m := rwkvHF(t)
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
// tokens every per-layer state vector has the same length (Dim) — a
// transformer's KV-cache would have grown from 3 to 7 rows per layer.
func TestRWKVDecodeStateIsO1(t *testing.T) {
	m := rwkvHF(t)

	sizes := func(steps int) [][5]int {
		st := m.NewDecodeState()
		ctx := backend.NewContext()
		tokens := []int{3, 7, 1, 9, 4, 2, 8}[:steps]
		for _, tok := range tokens {
			if _, err := m.DecodeStep(ctx, st, tok); err != nil {
				t.Fatal(err)
			}
		}
		out := make([][5]int, len(st.Layers))
		for l, s := range st.Layers {
			out[l] = [5]int{len(s.PrevTM), len(s.PrevCM), len(s.AA), len(s.BB), len(s.PP)}
		}
		return out
	}

	after3, after7 := sizes(3), sizes(7)
	if len(after3) != len(m.Blocks) || len(after7) != len(m.Blocks) {
		t.Fatalf("state layer count changed: %d vs %d (blocks %d)", len(after3), len(after7), len(m.Blocks))
	}
	for l := range after3 {
		if after3[l] != after7[l] {
			t.Fatalf("layer %d state grew with steps: %v after 3 vs %v after 7", l, after3[l], after7[l])
		}
		for i, n := range after3[l] {
			if n != m.Config.Dim {
				t.Fatalf("layer %d state vector %d has %d elements, want Dim=%d", l, i, n, m.Config.Dim)
			}
		}
	}
}
