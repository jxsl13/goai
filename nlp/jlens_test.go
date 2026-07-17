package nlp_test

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// jlensGPT builds a tiny deterministic random GPT for J-lens tests.
func jlensGPT(t *testing.T, vocab, ctxLen, dim, heads, layers int) *nlp.GPT {
	t.Helper()
	seed := uint64(11)
	xav := func(in, out int) *tensor.Tensor {
		w := tensor.New(tensor.F64, tensor.Shape{in, out})
		nn.XavierUniform(w, in, out, seed)
		seed++
		return w
	}
	vec := func(n int, v float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			x.SetF64(v, i)
		}
		return x
	}
	g := &nlp.GPT{
		Config: nlp.GPTConfig{Vocab: vocab, Ctx: ctxLen, Dim: dim, Heads: heads, Layers: layers, Eps: 1e-5},
		TokEmb: xav(vocab, dim),
		PosEmb: xav(ctxLen, dim),
		LNf:    &nn.LayerNorm{Gamma: vec(dim, 1), Beta: vec(dim, 0), Eps: 1e-5},
		Head:   xav(dim, vocab),
	}
	for range layers {
		attn, err := nlp.NewMHA(heads, xav(dim, dim), xav(dim, dim), xav(dim, dim), xav(dim, dim))
		if err != nil {
			t.Fatal(err)
		}
		attn.Causal = true
		g.Blocks = append(g.Blocks, &nlp.Block{
			LN1:  &nn.LayerNorm{Gamma: vec(dim, 1), Beta: vec(dim, 0), Eps: 1e-5},
			LN2:  &nn.LayerNorm{Gamma: vec(dim, 1), Beta: vec(dim, 0), Eps: 1e-5},
			Attn: attn,
			W1:   xav(dim, 4*dim), B1: vec(4*dim, 0),
			W2: xav(4*dim, dim), B2: vec(dim, 0),
		})
	}
	return g
}

func jlensLlama(t *testing.T, layers int) *nlp.Llama {
	t.Helper()
	m, err := nlp.NewLlama(nlp.LlamaConfig{
		Vocab: 16, Ctx: 8, Dim: 8, Heads: 2, Layers: layers, Hidden: 16, Eps: 1e-5,
	}, 5)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func bitEqualF64(t *testing.T, name string, a, b *tensor.Tensor) {
	t.Helper()
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("%s: shape %v != %v", name, a.Shape(), b.Shape())
	}
	as, bs := a.Storage().F64(), b.Storage().F64()
	for i := range as {
		if as[i] != bs[i] {
			t.Fatalf("%s: element %d: %v != %v (not bit-identical)", name, i, as[i], bs[i])
		}
	}
}

// §T810: a nil residual capture is byte-identical to the plain forward — same
// output bits AND the same number of recorded tape ops — and a non-nil capture
// changes nothing about the computation either (it only observes), receiving
// exactly Layers+1 residual snapshots of shape [seq, dim] in layer order.
func TestResidualCaptureByteIdentity(t *testing.T) {
	tokens := []int{1, 5, 3, 7, 2}
	models := map[string]interface {
		nlp.ResidualModel
		ForwardHidden(ctx *backend.Context, tokens []int) (*tensor.Tensor, error)
	}{
		"gpt":   jlensGPT(t, 16, 8, 8, 2, 2),
		"llama": jlensLlama(t, 2),
	}
	for name, m := range models {
		t.Run(name, func(t *testing.T) {
			tapePlain := autograd.NewTape()
			plain, err := m.ForwardHidden(tapePlain.Context(), tokens)
			if err != nil {
				t.Fatal(err)
			}
			tapeNil := autograd.NewTape()
			nilCap, err := m.ForwardResiduals(tapeNil.Context(), tokens, nil)
			if err != nil {
				t.Fatal(err)
			}
			bitEqualF64(t, "nil-capture output", plain, nilCap)
			if tapeNil.Len() != tapePlain.Len() {
				t.Fatalf("nil capture recorded %d ops, plain forward %d", tapeNil.Len(), tapePlain.Len())
			}

			tapeCap := autograd.NewTape()
			var got []*tensor.Tensor
			var order []int
			hooked, err := m.ForwardResiduals(tapeCap.Context(), tokens, func(layer int, h *tensor.Tensor) {
				order = append(order, layer)
				got = append(got, h)
			})
			if err != nil {
				t.Fatal(err)
			}
			bitEqualF64(t, "hooked output", plain, hooked)
			if tapeCap.Len() != tapePlain.Len() {
				t.Fatalf("hook recorded %d ops, plain forward %d — capture must not add ops", tapeCap.Len(), tapePlain.Len())
			}
			layers := 2
			if len(got) != layers+1 {
				t.Fatalf("captured %d residuals, want layers+1 = %d", len(got), layers+1)
			}
			for i, l := range order {
				if l != i {
					t.Fatalf("capture order %v, want 0..%d", order, layers)
				}
			}
			for l, h := range got {
				if !h.Shape().Equal(tensor.Shape{len(tokens), 8}) {
					t.Fatalf("residual %d shape %v, want [%d 8]", l, h.Shape(), len(tokens))
				}
			}
			// The last captured residual is PRE final norm: it must differ from
			// the returned post-norm hidden.
			same := true
			a, b := got[layers].Storage().F64(), hooked.Storage().F64()
			for i := range a {
				if a[i] != b[i] {
					same = false
					break
				}
			}
			if same {
				t.Fatal("last captured residual equals post-norm output — capture must tap PRE final norm")
			}
		})
	}
}

// §T811 self-consistency (a): at the last layer h_L = h_l, so J[Layers] is the
// identity — exactly, since the VJP contribution per source position is the
// identity row by construction. Verified on a 1-layer Llama and a 2-layer GPT.
func TestJLensLastLayerIdentity(t *testing.T) {
	corpus := [][]int{{1, 2, 3, 4}, {5, 6, 7}, {8, 9, 10, 11, 12}}
	t.Run("llama-1layer", func(t *testing.T) {
		jl, err := nlp.FitJLens(jlensLlama(t, 1), corpus)
		if err != nil {
			t.Fatal(err)
		}
		if jl.Layers != 1 || jl.Dim != 8 || jl.Arch != "llama" || jl.Weight != 3 {
			t.Fatalf("meta = {layers %d, dim %d, arch %q, weight %g}", jl.Layers, jl.Dim, jl.Arch, jl.Weight)
		}
		checkIdentity(t, jl.J[1], 1e-12)
	})
	t.Run("gpt-2layer", func(t *testing.T) {
		jl, err := nlp.FitJLens(jlensGPT(t, 16, 8, 8, 2, 2), corpus)
		if err != nil {
			t.Fatal(err)
		}
		if jl.Arch != "gpt2" {
			t.Fatalf("arch %q", jl.Arch)
		}
		checkIdentity(t, jl.J[2], 1e-12)
	})
}

func checkIdentity(t *testing.T, j *tensor.Tensor, tol float64) {
	t.Helper()
	d := j.Shape()[0]
	for a := range d {
		for b := range d {
			want := 0.0
			if a == b {
				want = 1
			}
			if got := j.AtF64(a, b); math.Abs(got-want) > tol {
				t.Fatalf("J[last][%d,%d] = %g, want %g", a, b, got, want)
			}
		}
	}
}

// §T811 self-consistency (b): finite differences. On a single-token sequence the
// fitted J_l is exactly ∂h_L(0)/∂h_l(0), so injecting ±ε·d into h_l via the
// capture hook and central-differencing h_L must match J_l·d to O(ε²).
func TestJLensFiniteDifference(t *testing.T) {
	model := jlensGPT(t, 16, 8, 8, 2, 2)
	tokens := []int{3}
	jl, err := nlp.FitJLens(model, [][]int{tokens})
	if err != nil {
		t.Fatal(err)
	}
	dim, layers := jl.Dim, jl.Layers

	dir := make([]float64, dim)
	for i := range dir {
		dir[i] = math.Sin(1.3*float64(i) + 0.7)
	}
	const eps = 1e-5

	// hLShifted runs a plain (eager, untaped) forward that injects shift·dir
	// into the residual stream at layer inject, returning h_L (pre final norm).
	hLShifted := func(inject int, shift float64) []float64 {
		var hL *tensor.Tensor
		_, err := model.ForwardResiduals(backend.NewContext(), tokens, func(layer int, h *tensor.Tensor) {
			if layer == inject {
				for i := range dim {
					h.SetF64(h.AtF64(0, i)+shift*dir[i], 0, i)
				}
			}
			if layer == layers {
				hL = h
			}
		})
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float64, dim)
		for i := range out {
			out[i] = hL.AtF64(0, i)
		}
		return out
	}

	for l := 0; l <= layers; l++ {
		plus, minus := hLShifted(l, eps), hLShifted(l, -eps)
		for a := range dim {
			fd := (plus[a] - minus[a]) / (2 * eps)
			want := 0.0
			for b := range dim {
				want += jl.J[l].AtF64(a, b) * dir[b]
			}
			if math.Abs(fd-want) > 1e-6 {
				t.Fatalf("layer %d row %d: finite-diff %g vs J·d %g (|Δ| = %.3g)", l, a, fd, want, math.Abs(fd-want))
			}
		}
	}
}

// §T811 self-consistency (c): the fit is a per-sequence expectation, so merging
// two disjoint half-corpus fits with their sequence-count weights reproduces the
// full-corpus fit (up to fp summation order).
func TestJLensMerge(t *testing.T) {
	model := jlensGPT(t, 16, 8, 8, 2, 2)
	corpus := [][]int{{1, 2, 3, 4, 5}, {6, 7, 8}, {9, 10, 11, 12}, {13, 14}}
	full, err := nlp.FitJLens(model, corpus)
	if err != nil {
		t.Fatal(err)
	}
	a, err := nlp.FitJLens(model, corpus[:2])
	if err != nil {
		t.Fatal(err)
	}
	b, err := nlp.FitJLens(model, corpus[2:])
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Merge(b, b.Weight); err != nil {
		t.Fatal(err)
	}
	if a.Weight != 4 {
		t.Fatalf("merged weight %g, want 4", a.Weight)
	}
	for l := range full.J {
		fs, ms := full.J[l].Storage().F64(), a.J[l].Storage().F64()
		for i := range fs {
			if math.Abs(fs[i]-ms[i]) > 1e-12 {
				t.Fatalf("layer %d elem %d: merged %g vs full-corpus %g", l, i, ms[i], fs[i])
			}
		}
	}
	// Geometry mismatch must be rejected.
	other, err := nlp.FitJLens(jlensLlama(t, 1), [][]int{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Merge(other, 1); err == nil {
		t.Fatal("Merge accepted mismatched geometry")
	}
}

// §T811 self-consistency (d): Save → LoadJLens round-trips every matrix
// bit-exactly plus all metadata.
func TestJLensSaveLoadRoundTrip(t *testing.T) {
	jl, err := nlp.FitJLens(jlensLlama(t, 2), [][]int{{1, 2, 3}, {4, 5}})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "lens.safetensors")
	if err := jl.Save(path); err != nil {
		t.Fatal(err)
	}
	got, err := nlp.LoadJLens(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dim != jl.Dim || got.Layers != jl.Layers || got.Arch != jl.Arch || got.Weight != jl.Weight {
		t.Fatalf("meta round-trip: got {%d %d %q %g} want {%d %d %q %g}",
			got.Dim, got.Layers, got.Arch, got.Weight, jl.Dim, jl.Layers, jl.Arch, jl.Weight)
	}
	for l := range jl.J {
		bitEqualF64(t, "layer matrix", jl.J[l], got.J[l])
	}
}

// FitJLens option knobs: MaxSequences caps the fit mass; MaxTargetsPerSeq bounds
// the backward count. Under target capping the last-layer matrix stays exactly
// diagonal (the h_L self-Jacobian rows are one-hot) with a uniform
// |targets|/seq diagonal — the documented truncation of the target sum.
func TestJLensFitOptions(t *testing.T) {
	model := jlensGPT(t, 16, 8, 8, 2, 2)
	corpus := [][]int{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}}
	jl, err := nlp.FitJLens(model, corpus,
		nlp.WithJLensMaxSequences(1), nlp.WithJLensMaxTargetsPerSeq(2))
	if err != nil {
		t.Fatal(err)
	}
	if jl.Weight != 1 {
		t.Fatalf("weight %g, want 1 (MaxSequences)", jl.Weight)
	}
	last := jl.J[jl.Layers]
	diag := last.AtF64(0, 0)
	if math.Abs(diag-2.0/5.0) > 1e-15 {
		t.Fatalf("capped diagonal %g, want |targets|/seq = 0.4", diag)
	}
	for a := range jl.Dim {
		for b := range jl.Dim {
			got := last.AtF64(a, b)
			if a == b && got != diag {
				t.Fatalf("diagonal not uniform: [%d,%d] = %g vs %g", a, b, got, diag)
			}
			if a != b && got != 0 {
				t.Fatalf("off-diagonal [%d,%d] = %g, want exact 0", a, b, got)
			}
		}
	}
}
