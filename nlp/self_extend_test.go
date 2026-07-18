package nlp_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (arXiv:2401.01325, neighbor attention): within the window the true relative
// position q−k is used unchanged.
func TestSelfExtendNeighbor(t *testing.T) {
	const window, group = 6, 4
	for d := 0; d < window; d++ {
		if got := nlp.SelfExtendRelPos(100, 100-d, window, group); got != d {
			t.Errorf("neighbor d=%d: got %d, want %d (true distance)", d, got, d)
		}
	}
}

// §V16 tier-1 (grouped attention + continuity shift): beyond the window the relative
// position is ⌊q/G⌋−⌊k/G⌋ + (w − ⌊w/G⌋).
func TestSelfExtendGrouped(t *testing.T) {
	const window, group = 6, 4
	for _, qk := range [][2]int{{100, 90}, {200, 50}, {1000, 0}} {
		q, k := qk[0], qk[1]
		want := q/group - k/group + window - window/group
		if got := nlp.SelfExtendRelPos(q, k, window, group); got != want {
			t.Errorf("grouped q=%d k=%d: got %d, want %d", q, k, got, want)
		}
	}
}

// §V16 tier-1 (monotonic mapping): for a fixed query the effective relative position is
// non-decreasing as the key moves further back — no backward jump at the neighbor/grouped
// boundary, so RoPE angles stay ordered.
func TestSelfExtendMonotonic(t *testing.T) {
	const q, window, group = 300, 8, 4
	prev := -1
	for k := q; k >= 0; k-- { // distance increases as k decreases
		rp := nlp.SelfExtendRelPos(q, k, window, group)
		if rp < prev {
			t.Errorf("non-monotonic at k=%d: %d < previous %d", k, rp, prev)
		}
		prev = rp
	}
}

// §V16 tier-1 (G=1 identity): group size 1 recovers ordinary attention (relpos = q−k).
func TestSelfExtendGroupOneIdentity(t *testing.T) {
	for _, qk := range [][2]int{{5, 0}, {100, 3}, {1000, 900}} {
		q, k := qk[0], qk[1]
		if got := nlp.SelfExtendRelPos(q, k, 4, 1); got != q-k {
			t.Errorf("G=1 q=%d k=%d: got %d, want %d", q, k, got, q-k)
		}
	}
}

// §V2 (the point): grouping compresses a large distance back below the training range —
// the max effective position over a long sequence is ≈ L/G + w, far below L.
func TestSelfExtendCompression(t *testing.T) {
	const seq, window, group = 4000, 8, 8
	maxRel := 0
	pos := nlp.SelfExtendPositions(seq, window, group)
	for q := range pos {
		for k := 0; k <= q; k++ {
			if pos[q][k] > maxRel {
				maxRel = pos[q][k]
			}
		}
	}
	if bound := seq/group + window; maxRel > bound {
		t.Errorf("max relative position %d exceeds compression bound %d", maxRel, bound)
	}
	if maxRel >= seq {
		t.Errorf("no compression: max relpos %d ≥ seq %d", maxRel, seq)
	}
}

// §V2: the matrix is causal (0 above the diagonal), 0 on the diagonal (q=k), and matches
// the scalar mapping below it.
func TestSelfExtendPositionsMatrix(t *testing.T) {
	const seq, window, group = 12, 3, 2
	m := nlp.SelfExtendPositions(seq, window, group)
	for q := range seq {
		if m[q][q] != 0 {
			t.Errorf("diagonal [%d,%d]=%d, want 0", q, q, m[q][q])
		}
		for k := q + 1; k < seq; k++ {
			if m[q][k] != 0 {
				t.Errorf("above diagonal [%d,%d]=%d, want 0 (causal)", q, k, m[q][k])
			}
		}
		for k := 0; k <= q; k++ {
			if want := nlp.SelfExtendRelPos(q, k, window, group); m[q][k] != want {
				t.Errorf("[%d,%d]=%d, want %d", q, k, m[q][k], want)
			}
		}
	}
}

func ExampleSelfExtendRelPos() {
	// Window 4, group 8. A key 3 back is a neighbor (relpos 3); a key 800 back is grouped and
	// compressed to about 800/8 + shift.
	fmt.Println(nlp.SelfExtendRelPos(1000, 997, 4, 8), nlp.SelfExtendRelPos(1000, 200, 4, 8))
	// Output:
	// 3 104
}

// §T513: with group=1 both score sources are identical (positions p/1 + w − w = p), so
// SelfExtendForward must collapse onto the plain Forward — same causal attention through
// a different kernel path (OpMHASelect + host RoPE vs fused OpMHA + OpRoPE), ≤1e−9.
func TestSelfExtendForwardCollapsesToForward(t *testing.T) {
	cfg := nlp.LlamaConfig{Vocab: 17, Ctx: 32, Dim: 16, Heads: 2, KVHeads: 1, Layers: 2, Hidden: 40, Eps: 1e-5}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{1, 5, 9, 2, 7, 3, 11, 4, 6, 8}
	ctx := backend.NewContext()
	plain, err := m.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	se, err := m.SelfExtendForward(ctx, tokens, 4, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range plain.Numel() {
		idx := tensor.Unravel(i, plain.Shape())
		if d := math.Abs(se.AtF64(idx...) - plain.AtF64(idx...)); d > 1e-9 {
			t.Fatalf("group=1 diverges at %v by %g", idx, d)
		}
	}
}

// TestSelfExtendForwardArchParity is the Self-Extend half of the decode-path
// equivalence gate (§V16 tier-1), the sibling of TestStreamGenerateMatchesGenerate.
// SelfExtendForward is a third implementation of the Llama forward pass, and it had
// drifted the same way StreamStep did: it applied neither the Qwen2 q/k/v biases, nor
// the Qwen3 QK-norm, nor any of the four Granite scalars. With group=1 both of its
// score sources collapse onto ordinary RoPE, so it must reproduce Forward EXACTLY for
// every architecture — the collapse identity is what makes the drift detectable.
//
// TestSelfExtendForwardCollapsesToForward above pins the same identity for a plain
// Llama; it passed throughout the drift precisely because a plain Llama has none of
// these features. That is the failure mode this test exists to close.
func TestSelfExtendForwardArchParity(t *testing.T) {
	tokens := []int{1, 5, 9, 2, 7, 3, 10, 4}
	for _, tc := range archParityModels(t) {
		t.Run(tc.name, func(t *testing.T) {
			ctx := backend.NewContext()
			plain, err := tc.m.Forward(ctx, tokens)
			if err != nil {
				t.Fatal(err)
			}
			se, err := tc.m.SelfExtendForward(ctx, tokens, 4, 1)
			if err != nil {
				t.Fatal(err)
			}
			for i := range plain.Numel() {
				idx := tensor.Unravel(i, plain.Shape())
				got, want := se.AtF64(idx...), plain.AtF64(idx...)
				if d := math.Abs(got - want); d > 1e-9*math.Max(1, math.Abs(want)) {
					t.Fatalf("group=1 SelfExtendForward %.12g vs Forward %.12g at %v (Δ%.3g)\n"+
						"the Self-Extend path is dropping an architecture feature Forward applies",
						got, want, idx, d)
				}
			}
		})
	}
}

// §V16 tier-1 (the point of Self-Extend): SelfExtendGenerate must generate PAST
// Config.Ctx — that is the entire reason grouped attention exists. It used to stop
// silently at Ctx (returning fewer tokens than asked for, with a nil error), which
// contradicted its own godoc; the real limit is the GROUPED one, so a group of G buys
// roughly G× the context. Here Ctx=16 and group=4, so ~50 tokens are well within reach
// while being 3× past the unextended limit.
func TestSelfExtendGenerateBeyondCtx(t *testing.T) {
	cfg := nlp.LlamaConfig{Vocab: 17, Ctx: 16, Dim: 16, Heads: 2, KVHeads: 1, Layers: 2, Hidden: 40, Eps: 1e-5}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	prompt := []int{1, 5, 9, 2}
	const maxNew = 40 // 44 total, 2.75× Config.Ctx
	out, err := m.SelfExtendGenerate(backend.NewContext(), prompt, maxNew, 4, 4)
	if err != nil {
		t.Fatalf("SelfExtendGenerate past Ctx: %v", err)
	}
	if len(out) != len(prompt)+maxNew {
		t.Fatalf("SelfExtendGenerate produced %d tokens, want %d — it stopped short of the "+
			"request instead of generating beyond Config.Ctx=%d", len(out), len(prompt)+maxNew, cfg.Ctx)
	}
}

// §V16 tier-1 (fail loudly, never silently short): grouping extends the reachable
// length but does NOT make it unbounded — the largest relative position the grouped
// source produces still has to land inside the pretrained range. A request that cannot
// fit must ERROR, not quietly return fewer tokens than asked for (the old behaviour).
func TestSelfExtendGenerateRejectsUnreachableLength(t *testing.T) {
	cfg := nlp.LlamaConfig{Vocab: 17, Ctx: 16, Dim: 16, Heads: 2, KVHeads: 1, Layers: 2, Hidden: 40, Eps: 1e-5}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatal(err)
	}
	// group=1 buys nothing, so 4+40 tokens cannot fit a Ctx of 16.
	out, err := m.SelfExtendGenerate(backend.NewContext(), []int{1, 5, 9, 2}, 40, 4, 1)
	if err == nil {
		t.Fatalf("an unreachable length must error, got %d tokens and nil error", len(out))
	}
	if !strings.Contains(err.Error(), "Self-Extend") {
		t.Fatalf("error should name the constraint it enforces, got: %v", err)
	}
}
