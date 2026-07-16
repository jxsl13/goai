package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// linformerRefAttention is an INDEPENDENT pure-Go reference for standard
// multi-head bidirectional softmax attention softmax(Qₕ·Kₕᵀ/√d_k)·Vₕ over the
// full [L,L] score matrix — no length projection. It exists to pin the §V16
// full-rank collapse: a Linformer with identity projections (k=L, E=F=I) must
// equal this to floating-point tolerance.
func linformerRefAttention(q, k, v *tensor.Tensor, heads int) *tensor.Tensor {
	L, dim := q.Shape()[0], q.Shape()[1]
	hd := dim / heads
	scale := 1 / math.Sqrt(float64(hd))
	out := tensor.New(tensor.F64, tensor.Shape{L, dim})
	for h := range heads {
		lo := h * hd
		for i := range L {
			sc := make([]float64, L)
			mx := math.Inf(-1)
			for j := range L {
				var dot float64
				for c := range hd {
					dot += q.AtF64(i, lo+c) * k.AtF64(j, lo+c)
				}
				sc[j] = dot * scale
				if sc[j] > mx {
					mx = sc[j]
				}
			}
			var sum float64
			for j := range L {
				sc[j] = math.Exp(sc[j] - mx)
				sum += sc[j]
			}
			for c := range hd {
				var acc float64
				for j := range L {
					acc += sc[j] / sum * v.AtF64(j, lo+c)
				}
				out.SetF64(acc, i, lo+c)
			}
		}
	}
	return out
}

// linformerProj runs a plain backend matmul a·b (helper for white-box shape
// checks that recompute the length projection from a layer's exported fields).
func linformerMatMul(t *testing.T, a, b *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return out[0]
}

// §V16 tier-1: LinformerAttention maps [L,Dim]→[L,Dim]; Params exposes 6 tensors
// (Wq,Wk,Wv,Wo,E,F) by default and 5 with WithLinformerShareKV (F == E).
func TestLinformerShapes(t *testing.T) {
	const dim, heads, L, k = 8, 2, 6, 3
	m, err := nn.NewLinformer(tensor.F64, dim, L, k, nn.WithLinformerHeads(heads), nn.WithLinformerSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	if m.HeadDim != dim/heads {
		t.Errorf("HeadDim=%d, want %d", m.HeadDim, dim/heads)
	}
	if !m.E.Shape().Equal(tensor.Shape{k, L}) || !m.F.Shape().Equal(tensor.Shape{k, L}) {
		t.Errorf("E,F shapes %v,%v, want [%d %d]", m.E.Shape(), m.F.Shape(), k, L)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, L, dim)
	out, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{L, dim}) {
		t.Fatalf("output shape %v, want [%d %d]", out.Shape(), L, dim)
	}
	if got, want := len(m.Params()), 6; got != want {
		t.Errorf("Params count %d, want %d (independent E,F)", got, want)
	}
	// WithLinformerShareKV: F is the SAME tensor as E, Params reports it once.
	ms, err := nn.NewLinformer(tensor.F64, dim, L, k, nn.WithLinformerHeads(heads), nn.WithLinformerShareKV())
	if err != nil {
		t.Fatal(err)
	}
	if ms.F != ms.E {
		t.Error("WithLinformerShareKV: F must be the same tensor as E")
	}
	if got, want := len(ms.Params()), 5; got != want {
		t.Errorf("shared-KV Params count %d, want %d", got, want)
	}
}

// §V16 (6) SANITY: the constructor rejects bad dims, and Forward rejects a wrong
// width, rank, AND a sequence length that differs from the fixed L (E,F are
// fixed-L, so length mismatch is a clear error — not a silent reshape).
func TestLinformerErrors(t *testing.T) {
	if _, err := nn.NewLinformer(tensor.F64, 0, 4, 2); err == nil {
		t.Error("dModel=0: want error")
	}
	if _, err := nn.NewLinformer(tensor.F64, 8, 4, 2, nn.WithLinformerHeads(3)); err == nil {
		t.Error("dModel%heads != 0: want error")
	}
	if _, err := nn.NewLinformer(tensor.F64, 8, 4, 2, nn.WithLinformerHeads(0)); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := nn.NewLinformer(tensor.F64, 8, 0, 1); err == nil {
		t.Error("seqLen<1: want error")
	}
	if _, err := nn.NewLinformer(tensor.F64, 8, 4, 0); err == nil {
		t.Error("projDim<1: want error")
	}
	if _, err := nn.NewLinformer(tensor.F64, 8, 4, 5); err == nil {
		t.Error("projDim>seqLen: want error")
	}
	if _, err := nn.NewLinformer(tensor.F64, 8, 4, 2, nn.WithLinformerIdentityProj()); err == nil {
		t.Error("identity with k != L: want error")
	}
	m, _ := nn.NewLinformer(tensor.F64, 8, 4, 2)
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{4, 6})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{4, 4, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{5, 8})); err == nil {
		t.Error("input length != fixed L: want error")
	}
}

// §V16 (1) FULL-RANK COLLAPSE — THE anchor: with k = L and E = F = the identity
// (WithLinformerIdentityProj), K̄ = K and V̄ = V, so Linformer is EXACTLY standard
// bidirectional softmax attention. Assert it equals an independent pure-Go
// reference softmax(Q·Kᵀ/√d_k)·V to 1e-10, for single- and multi-head. This pins
// "Linformer = full attention with a low-rank length projection".
func TestLinformerFullRankCollapse(t *testing.T) {
	for _, heads := range []int{1, 2, 4} {
		const dim, L = 8, 5
		m, err := nn.NewLinformer(tensor.F64, dim, L, L,
			nn.WithLinformerHeads(heads), nn.WithLinformerIdentityProj(), nn.WithLinformerSeed(3))
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewPCG(uint64(heads), 99))
		x := randMat(rng, L, dim)
		got, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		q := linformerMatMul(t, x, m.Wq)
		k := linformerMatMul(t, x, m.Wk)
		v := linformerMatMul(t, x, m.Wv)
		ref := linformerRefAttention(q, k, v, heads)
		// The reference has no output projection; apply Wo to match.
		ref = linformerMatMul(t, ref, m.Wo)
		if d := maxAbsDiff(got, ref); d > 1e-10 {
			t.Fatalf("heads=%d: identity-proj Linformer != full softmax attention, max diff %.3g", heads, d)
		} else {
			t.Logf("heads=%d: identity-proj Linformer == full attention, max diff %.3g", heads, d)
		}
	}
}

// §V16 (2) PROJECTION SHAPES / COMPLEXITY: E,F ∈ R^{k×L} shrink the L keys/values
// to k; the compressed K̄,V̄ are [k,Dim] (per head [k,HeadDim]); and the score
// matrix Q·K̄ᵀ is [L,k] — NOT the [L,L] of full attention — so per head the compute
// is O(L·k·HeadDim), linear in L for fixed k. Recomputed white-box from the
// layer's exported Wq,Wk,E.
func TestLinformerShapeComplexity(t *testing.T) {
	const dim, heads, L, k = 12, 3, 10, 4
	hd := dim / heads
	m, err := nn.NewLinformer(tensor.F64, dim, L, k, nn.WithLinformerHeads(heads), nn.WithLinformerSeed(5))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(7, 8))
	x := randMat(rng, L, dim)

	q := linformerMatMul(t, x, m.Wq)    // [L, Dim]
	kk := linformerMatMul(t, x, m.Wk)   // [L, Dim]
	kbar := linformerMatMul(t, m.E, kk) // E[k,L]·K[L,Dim] = [k, Dim]
	vbar := linformerMatMul(t, m.F, linformerMatMul(t, x, m.Wv))
	if !kbar.Shape().Equal(tensor.Shape{k, dim}) {
		t.Fatalf("K̄ shape %v, want [%d %d] (length compressed L→k)", kbar.Shape(), k, dim)
	}
	if !vbar.Shape().Equal(tensor.Shape{k, dim}) {
		t.Fatalf("V̄ shape %v, want [%d %d]", vbar.Shape(), k, dim)
	}
	// Per-head scores: slice head 0, transpose K̄ₕ, matmul.
	sl := func(m *tensor.Tensor, hd int) *tensor.Tensor {
		out, err := backend.Execute(backend.NewContext(), backend.OpSlice, []*tensor.Tensor{m}, backend.SliceAttrs{Axis: 1, Start: 0, End: hd})
		if err != nil {
			t.Fatal(err)
		}
		return out[0]
	}
	qh := sl(q, hd)     // [L, HeadDim]
	kbh := sl(kbar, hd) // [k, HeadDim]
	kbhT, err := backend.Execute(backend.NewContext(), backend.OpTranspose, []*tensor.Tensor{kbh}, nil)
	if err != nil {
		t.Fatal(err)
	}
	scores := linformerMatMul(t, qh, kbhT[0]) // [L, k]
	if !scores.Shape().Equal(tensor.Shape{L, k}) {
		t.Fatalf("score matrix %v, want [%d %d] (Linformer's [L,k], not [L,L])", scores.Shape(), L, k)
	}
	if k >= L {
		t.Fatalf("test setup: k=%d must be < L=%d to prove genuine shrinkage", k, L)
	}
	t.Logf("scores [L,k]=[%d,%d] vs full [L,L]=[%d,%d]: %dx fewer score entries; K̄,V̄ = [k,d]=[%d,%d]",
		L, k, L, L, L/k, k, dim)
}

// linformerGradcheck compares every analytic gradient of every parameter of m
// against central finite differences of the summed forward output (F64 only).
// Returns the entry count and max relative error for the report.
func linformerGradcheck(t *testing.T, m *nn.LinformerAttention, x *tensor.Tensor) (int, float64) {
	t.Helper()
	const h = 1e-6
	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range o.Numel() {
			acc += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return acc
	}
	var entries int
	var maxRel float64
	for pi, p := range m.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("param %d: nil grad — no gradient reached this parameter", pi)
		}
		var nz bool
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := sumLoss()
			p.SetF64(o-h, coord...)
			lm := sumLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			entries++
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
			if rel > maxRel {
				maxRel = rel
			}
			if rel > 1e-4 {
				t.Errorf("param %d grad%v: numeric %.8g vs analytic %.8g", pi, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("param %d: all analytic grads zero — parameter not exercised", pi)
		}
	}
	return entries, maxRel
}

// §V16 (3) GRADCHECK: central finite differences confirm gradients reach EVERY
// parameter — Wq/Wk/Wv/Wo AND the length projections E,F — in both the
// independent-projection and shared-KV (F == E, gradient accumulates both the E·K
// and E·V paths) configurations.
func TestLinformerGradcheck(t *testing.T) {
	const dim, heads, L, k = 4, 2, 4, 2
	rng := rand.New(rand.NewPCG(29, 7))
	x := randMat(rng, L, dim)

	m, err := nn.NewLinformer(tensor.F64, dim, L, k, nn.WithLinformerHeads(heads), nn.WithLinformerSeed(3))
	if err != nil {
		t.Fatal(err)
	}
	n, rel := linformerGradcheck(t, m, x)
	t.Logf("independent E,F gradcheck: %d entries, max rel err %.3g", n, rel)
	if rel > 1e-4 {
		t.Fatalf("gradcheck max rel err %.3g > 1e-4", rel)
	}

	ms, err := nn.NewLinformer(tensor.F64, dim, L, k, nn.WithLinformerHeads(heads), nn.WithLinformerShareKV(), nn.WithLinformerSeed(9))
	if err != nil {
		t.Fatal(err)
	}
	n, rel = linformerGradcheck(t, ms, x)
	t.Logf("shared-KV (F==E) gradcheck: %d entries, max rel err %.3g", n, rel)
	if rel > 1e-4 {
		t.Fatalf("shared-KV gradcheck max rel err %.3g > 1e-4", rel)
	}
}

// linformerPoolProj builds a [k,L] mean-pooling length projection: row a averages
// the contiguous block of L/k positions in bucket a. At k = L it is the identity
// (each bucket is one position), so a Linformer with this projection recovers full
// attention exactly; for k < L it is a genuine, sensible compression.
func linformerPoolProj(k, L int) *tensor.Tensor {
	e := tensor.New(tensor.F64, tensor.Shape{k, L})
	for a := range k {
		lo, hi := a*L/k, (a+1)*L/k
		if hi <= lo {
			hi = lo + 1
		}
		w := 1.0 / float64(hi-lo)
		for j := lo; j < hi && j < L; j++ {
			e.SetF64(w, a, j)
		}
	}
	return e
}

// §V16 (4a) THE VALUE — APPROXIMATION: with a fixed mean-pooling length projection
// (a legitimate Linformer projection), the output error vs full attention shrinks
// monotonically as k → L and hits EXACTLY zero at k = L (where pooling = identity).
// This is the low-rank-approximation promise made concrete: more projection
// capacity k → closer to full attention.
func TestLinformerApproximation(t *testing.T) {
	const dim, heads, L = 16, 2, 64
	full, err := nn.NewLinformer(tensor.F64, dim, L, L,
		nn.WithLinformerHeads(heads), nn.WithLinformerIdentityProj())
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(11, 22))
	ks := []int{L / 4, L / 2, 3 * L / 4, L}
	errs := make([]float64, len(ks))
	const trials = 8
	for ki, k := range ks {
		var sumRel float64
		for range trials {
			m, err := nn.NewLinformer(tensor.F64, dim, L, k, nn.WithLinformerHeads(heads))
			if err != nil {
				t.Fatal(err)
			}
			m.E = linformerPoolProj(k, L)
			m.F = linformerPoolProj(k, L)
			x := randMat(rng, L, dim)
			of, err := full.Forward(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			ol, err := m.Forward(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			var num, den float64
			for i := range of.Numel() {
				c := tensor.Unravel(i, of.Shape())
				d := of.AtF64(c...) - ol.AtF64(c...)
				num += d * d
				den += of.AtF64(c...) * of.AtF64(c...)
			}
			sumRel += math.Sqrt(num / den)
		}
		errs[ki] = sumRel / trials
		t.Logf("k=%d (k/L=%.2f): mean rel err vs full attention %.4f", k, float64(k)/float64(L), errs[ki])
	}
	for i := 1; i < len(errs); i++ {
		if errs[i] >= errs[i-1] {
			t.Errorf("approximation error not shrinking with k: err[k=%d]=%.4f ≥ err[k=%d]=%.4f", ks[i], errs[i], ks[i-1], errs[i-1])
		}
	}
	if errs[len(errs)-1] > 1e-10 {
		t.Errorf("at k=L pooling should equal full attention exactly, got rel err %.3g", errs[len(errs)-1])
	}
}

// §V16 (4b) THE VALUE — LEARNS: a small BIDIRECTIONAL task. Every position must
// predict the MAJORITY token of the whole sequence — a permutation-invariant
// global aggregate that needs each position to see all others (impossible for a
// causal model, natural for bidirectional Linformer). A single Linformer block +
// head trained a few hundred steps must at least HALVE its cross-entropy.
func TestLinformerLearnsToyTask(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a small Linformer; skipped in -short")
	}
	const (
		vocab = 4
		L     = 24
		k     = 12
		dim   = 32
		steps = 300
	)
	rng := rand.New(rand.NewPCG(2, 0x11f0)) // #nosec G404 — deterministic test corpus
	emb := tensor.New(tensor.F64, tensor.Shape{vocab, dim})
	nn.XavierUniform(emb, vocab, dim, 1)
	attn, err := nn.NewLinformer(tensor.F64, dim, L, k, nn.WithLinformerHeads(4), nn.WithLinformerSeed(5))
	if err != nil {
		t.Fatal(err)
	}
	norm := nn.NewRMSNorm(tensor.F64, dim)
	finalNorm := nn.NewRMSNorm(tensor.F64, dim)
	head := nn.NewLinear(tensor.F64, dim, vocab, 2)
	params := []*tensor.Tensor{emb}
	params = append(params, norm.Params()...)
	params = append(params, attn.Params()...)
	params = append(params, finalNorm.Params()...)
	params = append(params, head.Params()...)

	onehot := func(seq []int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{len(seq), vocab})
		for i, tk := range seq {
			x.SetF64(1, i, tk)
		}
		return x
	}
	forward := func(ctx *backend.Context, seq []int) (*tensor.Tensor, error) {
		x, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{onehot(seq), emb}, nil)
		if err != nil {
			return nil, err
		}
		h := x[0]
		n, err := norm.Forward(ctx, h)
		if err != nil {
			return nil, err
		}
		a, err := attn.Forward(ctx, n)
		if err != nil {
			return nil, err
		}
		sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{h, a}, nil) // residual
		if err != nil {
			return nil, err
		}
		fn, err := finalNorm.Forward(ctx, sum[0])
		if err != nil {
			return nil, err
		}
		return head.Forward(ctx, fn)
	}

	sample := func() ([]int, int) {
		seq := make([]int, L)
		counts := make([]int, vocab)
		for i := range seq {
			seq[i] = rng.IntN(vocab)
			counts[seq[i]]++
		}
		maj, best := 0, -1
		for tk, c := range counts {
			if c > best {
				best, maj = c, tk
			}
		}
		return seq, maj
	}

	opt := nn.NewAdamW(params, 3e-3, 0.01)
	targets := tensor.New(tensor.F64, tensor.Shape{L})
	var first, last float64
	for step := range steps {
		seq, maj := sample()
		for i := range L {
			targets.SetF64(float64(maj), i) // every position predicts the sequence majority
		}
		tape := autograd.NewTape()
		logits, err := forward(tape.Context(), seq)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(tape.Context(), logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(params, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("Linformer majority-token task: CE %.3f → %.3f (uniform = %.3f)", first, last, math.Log(float64(vocab)))
	if last >= first*0.5 {
		t.Fatalf("Linformer did not train (%.3f → %.3f)", first, last)
	}
}

// §V16 (6) SANITY: Forward is deterministic — same weights and input give
// bit-identical output across calls.
func TestLinformerDeterminism(t *testing.T) {
	const dim, heads, L, k = 8, 2, 6, 3
	rng := rand.New(rand.NewPCG(2, 3))
	x := randMat(rng, L, dim)
	m, err := nn.NewLinformer(tensor.F64, dim, L, k, nn.WithLinformerHeads(heads), nn.WithLinformerSeed(7))
	if err != nil {
		t.Fatal(err)
	}
	a, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range L {
		for j := range dim {
			if a.AtF64(i, j) != b.AtF64(i, j) {
				t.Fatalf("non-deterministic at (%d,%d): %v vs %v", i, j, a.AtF64(i, j), b.AtF64(i, j))
			}
		}
	}
}

func ExampleLinformerAttention() {
	// A tiny 2-head Linformer over a fixed sequence of length L=6, compressing the
	// key/value length to k=3 via learned projections E,F ∈ R^{3×6}. The score
	// matrix is [L,k]=[6,3], not [L,L]=[6,6] — linear, not quadratic, in L. It is
	// bidirectional (non-causal by construction, arXiv:2006.04768).
	m, _ := nn.NewLinformer(tensor.F64, 4, 6, 3, nn.WithLinformerHeads(2), nn.WithLinformerSeed(1))
	x := tensor.New(tensor.F64, tensor.Shape{6, 4})
	for i := range 6 {
		x.SetF64(1, i, i%4) // arbitrary non-zero input
	}
	out, _ := m.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(m.Params()))
	// Output: (6, 4) params=6
}
