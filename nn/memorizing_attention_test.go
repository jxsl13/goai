package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// memExec runs a single-output op or fails the test — a terse helper for the
// hand-built reference attention below.
func memExec(t *testing.T, ctx *backend.Context, op backend.Op, at backend.Attrs, in ...*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	out, err := backend.Execute(ctx, op, in, at)
	if err != nil {
		t.Fatalf("op %v: %v", op, err)
	}
	return out[0]
}

// memRandMat builds a random [r,c] F64 matrix in [-1,1).
func memRandMat(rng *rand.Rand, r, c int) *tensor.Tensor {
	m := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			m.SetF64(2*rng.Float64()-1, i, j)
		}
	}
	return m
}

// memSumAll reduces every element of x to a scalar through ctx (differentiable).
func memSumAll(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	axes := make([]int, x.Ndim())
	for i := range axes {
		axes[i] = i
	}
	o, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: axes})
	if err != nil {
		return nil, err
	}
	return o[0], nil
}

// memRandPairs builds n random [1,dim] key and value rows (independent) for
// seeding a memory bank in tests.
func memRandPairs(rng *rand.Rand, n, dim int) (keys, vals *tensor.Tensor) {
	keys = tensor.New(tensor.F64, tensor.Shape{n, dim})
	vals = tensor.New(tensor.F64, tensor.Shape{n, dim})
	for i := range n {
		for j := range dim {
			keys.SetF64(2*rng.Float64()-1, i, j)
			vals.SetF64(2*rng.Float64()-1, i, j)
		}
	}
	return keys, vals
}

// memReferenceLocal recomputes the layer's local-only path independently:
// InProj → slice Q,K,V → OpMHA(causal, window) → OutProj. This is the standard
// causal (optionally windowed) self-attention the empty-memory layer must equal.
func memReferenceLocal(t *testing.T, m *MemorizingAttention, x *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	ctx := backend.NewContext()
	p, err := m.InProj.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	sl := func(i int) *tensor.Tensor {
		return memExec(t, ctx, backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: i * m.Dim, End: (i + 1) * m.Dim}, p)
	}
	q, k, v := sl(0), sl(1), sl(2)
	attn := memExec(t, ctx, backend.OpMHA, backend.AttnAttrs{Heads: m.Heads, Causal: true, Window: m.window}, q, k, v)
	o, err := m.OutProj.Forward(ctx, attn)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

func memMaxAbsDiff(a, b *tensor.Tensor) float64 {
	var mx float64
	for i := range a.Numel() {
		c := tensor.Unravel(i, a.Shape())
		if d := math.Abs(a.AtF64(c...) - b.AtF64(c...)); d > mx {
			mx = d
		}
	}
	return mx
}

// §V16.1 EMPTY-MEMORY COLLAPSE (the anchor): with an empty memory the layer is
// EXACTLY ordinary causal self-attention (InProj → OpMHA → OutProj) — the memory
// branch and gate add nothing. Asserted at 1e-10 for the full-causal default and
// for a sliding-window local branch.
func TestMemorizingEmptyMemoryCollapse(t *testing.T) {
	for _, win := range []int{0, 2} {
		rng := rand.New(rand.NewPCG(1, uint64(win)+7))
		x := memRandMat(rng, 5, 8)
		m, err := NewMemorizingAttention(tensor.F64, 8, 2, 4, WithMemorizingLocalWindow(win))
		if err != nil {
			t.Fatal(err)
		}
		if m.Memory.Size() != 0 {
			t.Fatalf("fresh memory size %d, want 0", m.Memory.Size())
		}
		got, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		ref := memReferenceLocal(t, m, x)
		if d := memMaxAbsDiff(got, ref); d > 1e-10 {
			t.Errorf("window=%d: empty memory != local attention, max diff %.3g", win, d)
		}
	}
}

// §V16.2 RETRIEVAL correctness: the exact top-K k-NN returns the stored keys
// with the highest dot product to the query, in descending score order, matching
// an independent brute-force argsort (indices + scores); a query equal to a
// stored key retrieves that key first.
func TestMemorizingRetrieval(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 3))
	const n, dim = 12, 6
	m := newMemMemory(dim, 100)
	keys, vals := memRandPairs(rng, n, dim)
	if err := m.AddSegment(keys, vals); err != nil {
		t.Fatal(err)
	}
	// Arbitrary query over head 0 (heads=1 layout: full width).
	q := make([]float64, dim)
	for d := range dim {
		q[d] = 2*rng.Float64() - 1
	}
	const topM = 5
	idx, scores := m.retrieveHead(q, 0, dim, topM)

	// Independent brute-force ranking.
	type ps struct {
		i int
		s float64
	}
	brute := make([]ps, n)
	for i := range n {
		var s float64
		for d := range dim {
			s += q[d] * keys.AtF64(i, d)
		}
		brute[i] = ps{i, s}
	}
	for a := 0; a < n; a++ { // stable selection sort by score desc, index asc
		best := a
		for b := a + 1; b < n; b++ {
			if brute[b].s > brute[best].s {
				best = b
			}
		}
		brute[a], brute[best] = brute[best], brute[a]
	}
	for r := range topM {
		if idx[r] != brute[r].i {
			t.Errorf("rank %d: idx %d, want %d", r, idx[r], brute[r].i)
		}
		if math.Abs(scores[r]-brute[r].s) > 1e-12 {
			t.Errorf("rank %d: score %.6g, want %.6g", r, scores[r], brute[r].s)
		}
	}
	// Scores must be non-increasing.
	for r := 1; r < topM; r++ {
		if scores[r] > scores[r-1]+1e-15 {
			t.Errorf("scores not descending at %d: %.6g > %.6g", r, scores[r], scores[r-1])
		}
	}
	// A query identical to stored key #7 retrieves #7 first (self dot is maximal).
	qk := make([]float64, dim)
	for d := range dim {
		qk[d] = keys.AtF64(7, d)
	}
	id2, _ := m.retrieveHead(qk, 0, dim, 1)
	if id2[0] != 7 {
		t.Errorf("query == key#7 retrieved %d first, want 7", id2[0])
	}
}

// §V16.3 MEMORY GROWTH / detach: AddSegment grows the bank and caps it at
// MemorySize with FIFO eviction of the oldest pairs; the stored pairs are
// detached copies, so no gradient flows into a past segment's source tensor —
// gradients reach only the current projections and the gate.
func TestMemorizingMemoryGrowthAndDetach(t *testing.T) {
	// Growth + FIFO eviction.
	m := newMemMemory(4, 5) // cap 5
	rng := rand.New(rand.NewPCG(4, 1))
	k1, v1 := memRandPairs(rng, 3, 4)
	if err := m.AddSegment(k1, v1); err != nil {
		t.Fatal(err)
	}
	if m.Size() != 3 {
		t.Fatalf("size %d after 3, want 3", m.Size())
	}
	k2, v2 := memRandPairs(rng, 4, 4) // total 7 > cap 5 → evict oldest 2
	if err := m.AddSegment(k2, v2); err != nil {
		t.Fatal(err)
	}
	if m.Size() != 5 {
		t.Fatalf("size %d after cap, want 5", m.Size())
	}
	// The 2 oldest (k1 rows 0,1) evicted; newest surviving rows are k2's rows.
	for d := range 4 {
		if math.Abs(m.keys[len(m.keys)-1][d]-k2.AtF64(3, d)) > 0 {
			t.Fatalf("newest stored key != last added row")
		}
		if math.Abs(m.keys[0][d]-k1.AtF64(2, d)) > 0 {
			t.Fatalf("oldest surviving key != k1 row 2 (FIFO eviction wrong)")
		}
	}

	// Detach: a source tensor whose (taped) output is added to memory receives
	// NO gradient from a later Forward that retrieves it; the current projections
	// and gate DO.
	lay, err := NewMemorizingAttention(tensor.F64, 4, 1, 9, WithMemorizingTopK(2), WithMemorizingMemorySize(4))
	if err != nil {
		t.Fatal(err)
	}
	src := memRandMat(rng, 3, 4) // "past segment" leaf tensor
	past := autograd.NewTape()
	pastKV, err := lay.InProj.Forward(past.Context(), src) // a taped op producing K,V territory
	if err != nil {
		t.Fatal(err)
	}
	kSeg := memExec(t, past.Context(), backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 4, End: 8}, pastKV)
	vSeg := memExec(t, past.Context(), backend.OpSlice, backend.SliceAttrs{Axis: 1, Start: 8, End: 12}, pastKV)
	if err := lay.AddSegment(kSeg, vSeg); err != nil {
		t.Fatal(err)
	}

	tape := autograd.NewTape()
	x := memRandMat(rng, 3, 4)
	out, err := lay.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := memSumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if g := tape.Grad(src); g != nil {
		t.Errorf("gradient reached the detached past-segment source (%v) — memory not detached", g.Shape())
	}
	if tape.Grad(lay.InProj.W) == nil {
		t.Error("no gradient reached the current InProj (projections must train)")
	}
	if tape.Grad(lay.GateLogit) == nil {
		t.Error("no gradient reached the gate logit (gate must train)")
	}
}

// memForceGate sets every head's gate logit so σ(logit) ≈ target (0 or 1) to a
// far tail; f64 tail 4e-18 leaves the opposite branch's coefficient below 1e-10.
func memForceGate(m *MemorizingAttention, high bool) {
	v := -40.0
	if high {
		v = 40.0
	}
	for h := range m.Heads {
		m.GateLogit.SetF64(v, 0, h)
	}
}

// §V16.4 GATE limits: gate=0 ⇒ pure local (the collapse even with a populated
// memory); gate=1 ⇒ pure memory attention. Both asserted against the
// independently recomputed branches.
func TestMemorizingGateLimits(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 2))
	x := memRandMat(rng, 4, 8)
	m, err := NewMemorizingAttention(tensor.F64, 8, 2, 6, WithMemorizingTopK(3), WithMemorizingMemorySize(8))
	if err != nil {
		t.Fatal(err)
	}
	keys, vals := memRandPairs(rng, 6, 8)
	if err := m.AddSegment(keys, vals); err != nil {
		t.Fatal(err)
	}

	// gate=0 ⇒ pure local == reference local path.
	memForceGate(m, false)
	got0, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	ref := memReferenceLocal(t, m, x)
	if d := memMaxAbsDiff(got0, ref); d > 1e-9 {
		t.Errorf("gate=0: not pure local, max diff %.3g", d)
	}

	// gate=1 ⇒ pure memory == OutProj(memoryAttention(Q)).
	memForceGate(m, true)
	got1, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	q, _, _, err := m.project(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	mem, err := m.memoryAttention(ctx, q)
	if err != nil {
		t.Fatal(err)
	}
	refMem, err := m.OutProj.Forward(ctx, mem)
	if err != nil {
		t.Fatal(err)
	}
	if d := memMaxAbsDiff(got1, refMem); d > 1e-9 {
		t.Errorf("gate=1: not pure memory, max diff %.3g", d)
	}
	// The two limits must genuinely differ (the gate does something).
	if d := memMaxAbsDiff(got0, got1); d < 1e-6 {
		t.Errorf("gate limits identical (diff %.3g) — gate has no effect", d)
	}
}

// §V16.5 GRADCHECK: central finite differences confirm analytic gradients reach
// every parameter (fused Q/K/V in-proj, out-proj W_O, and the per-head gate
// logit) with a fixed non-empty memory. topK == memory size, so retrieval is the
// full set and the softmax weights vary smoothly under perturbation.
func TestMemorizingGradcheck(t *testing.T) {
	const dim, heads, seq = 4, 2, 3
	rng := rand.New(rand.NewPCG(7, 11))
	x := memRandMat(rng, seq, dim)
	m, err := NewMemorizingAttention(tensor.F64, dim, heads, 3, WithMemorizingTopK(3), WithMemorizingMemorySize(3))
	if err != nil {
		t.Fatal(err)
	}
	keys, vals := memRandPairs(rng, 3, dim)
	if err := m.AddSegment(keys, vals); err != nil {
		t.Fatal(err)
	}
	// Perturb the gate off 0.5 so its gradient is well away from any symmetry.
	m.GateLogit.SetF64(0.3, 0, 0)
	m.GateLogit.SetF64(-0.4, 0, 1)

	const h = 1e-6
	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := memSumAll(tape.Context(), out)
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
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad — no gradient reached this parameter", name)
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
			if math.Abs(ana) > 1e-9 {
				nz = true
			}
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
		if !nz {
			t.Errorf("%s: all analytic grads zero — parameter not exercised", name)
		}
	}
	check("InProj.W", m.InProj.W)
	check("InProj.B", m.InProj.B)
	check("OutProj.W", m.OutProj.W)
	check("OutProj.B", m.OutProj.B)
	check("GateLogit", m.GateLogit)
}

// memSetIdentityProjections turns InProj into the block identity [I|I|I] (so
// Q=K=V=x) and OutProj into the identity, making the layer a transparent probe:
// the memory branch outputs exactly the retrieved value, the local branch a
// causal average of the current segment's tokens.
func memSetIdentityProjections(m *MemorizingAttention) {
	d := m.Dim
	for i := range d {
		for j := range 3 * d {
			v := 0.0
			if j%d == i {
				v = 1
			}
			m.InProj.W.SetF64(v, i, j)
		}
	}
	for i := range 3 * d {
		m.InProj.B.SetF64(0, i)
	}
	for i := range d {
		for j := range d {
			v := 0.0
			if i == j {
				v = 1
			}
			m.OutProj.W.SetF64(v, i, j)
		}
		m.OutProj.B.SetF64(0, i)
	}
}

// §V16.6 THE VALUE — long-context recall: an associative-recall task whose answer
// lives in an EARLIER segment stored only in memory (never in the local window).
// The memorizing layer retrieves the matching key and emits its value; a local-
// only baseline (empty memory) cannot. Reports both accuracies.
func TestMemorizingLongContextRecall(t *testing.T) {
	const L = 6 // dim = L: tokens are the L orthonormal basis vectors
	perm := func(i int) int { return (i + 2) % L }

	mkLayer := func() *MemorizingAttention {
		m, err := NewMemorizingAttention(tensor.F64, L, 1, 1, WithMemorizingTopK(1), WithMemorizingMemorySize(L))
		if err != nil {
			t.Fatal(err)
		}
		memSetIdentityProjections(m)
		return m
	}
	basis := func(i int) []float64 {
		r := make([]float64, L)
		r[i] = 1
		return r
	}
	// Query segment: the L cues e_0..e_{L-1} (a later segment; its local context
	// holds only these cues, never the answers).
	x := tensor.New(tensor.F64, tensor.Shape{L, L})
	for i := range L {
		x.SetF64(1, i, i)
	}
	argmaxRow := func(o *tensor.Tensor, r int) int {
		best, bv := 0, math.Inf(-1)
		for c := range L {
			if v := o.AtF64(r, c); v > bv {
				best, bv = c, v
			}
		}
		return best
	}

	// Memory layer: store (key=e_i, value=e_{perm(i)}) as the "earlier segment".
	mem := mkLayer()
	for i := range L {
		k := tensor.FromFloat64(tensor.Shape{1, L}, basis(i))
		v := tensor.FromFloat64(tensor.Shape{1, L}, basis(perm(i)))
		if err := mem.AddSegment(k, v); err != nil {
			t.Fatal(err)
		}
	}
	memForceGate(mem, true) // gate=1: read from memory
	memOut, err := mem.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}

	// Local-only baseline: identical layer, EMPTY memory (pure local attention).
	loc := mkLayer()
	locOut, err := loc.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}

	var memCorrect, locCorrect int
	for i := range L {
		if argmaxRow(memOut, i) == perm(i) {
			memCorrect++
		}
		if argmaxRow(locOut, i) == perm(i) {
			locCorrect++
		}
	}
	memAcc := float64(memCorrect) / float64(L)
	locAcc := float64(locCorrect) / float64(L)
	t.Logf("long-context recall: memory=%.0f%% (%d/%d)  local-only=%.0f%% (%d/%d)",
		memAcc*100, memCorrect, L, locAcc*100, locCorrect, L)
	if memAcc < 1.0 {
		t.Errorf("memory recall accuracy %.2f, want 1.0 (retrieval must recover the stored value)", memAcc)
	}
	if locAcc > 0.0 {
		t.Errorf("local-only accuracy %.2f, want 0 (the answer is beyond the local window)", locAcc)
	}
}

// §V16.7 sanity: constructor + Forward + AddSegment boundary and shape errors,
// and determinism.
func TestMemorizingSanity(t *testing.T) {
	if _, err := NewMemorizingAttention(tensor.F64, 7, 2, 1); err == nil {
		t.Error("dim%heads != 0: want error")
	}
	if _, err := NewMemorizingAttention(tensor.F64, 0, 1, 1); err == nil {
		t.Error("dim=0: want error")
	}
	if _, err := NewMemorizingAttention(tensor.F64, 8, 0, 1); err == nil {
		t.Error("heads=0: want error")
	}
	if _, err := NewMemorizingAttention(tensor.F64, 8, 2, 1, WithMemorizingTopK(0)); err == nil {
		t.Error("topK=0: want error")
	}
	if _, err := NewMemorizingAttention(tensor.F64, 8, 2, 1, WithMemorizingMemorySize(0)); err == nil {
		t.Error("memSize=0: want error")
	}
	if _, err := NewMemorizingAttention(tensor.F64, 8, 2, 1, WithMemorizingTopK(9), WithMemorizingMemorySize(8)); err == nil {
		t.Error("topK>memSize: want error")
	}
	m, err := NewMemorizingAttention(tensor.F64, 8, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 4})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := m.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
	if err := m.AddSegment(tensor.New(tensor.F64, tensor.Shape{2, 4}), tensor.New(tensor.F64, tensor.Shape{2, 8})); err == nil {
		t.Error("AddSegment wrong width: want error")
	}
	if err := m.AddSegment(tensor.New(tensor.F64, tensor.Shape{2, 8}), tensor.New(tensor.F64, tensor.Shape{3, 8})); err == nil {
		t.Error("AddSegment mismatched rows: want error")
	}
	if err := m.AddSegment(tensor.New(tensor.F64, tensor.Shape{8}), tensor.New(tensor.F64, tensor.Shape{8})); err == nil {
		t.Error("AddSegment rank-1: want error")
	}

	// Determinism: same seed → identical parameters and identical Forward output.
	rng := rand.New(rand.NewPCG(3, 3))
	x := memRandMat(rng, 4, 8)
	a, _ := NewMemorizingAttention(tensor.F64, 8, 2, 77)
	b, _ := NewMemorizingAttention(tensor.F64, 8, 2, 77)
	oa, err := a.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	ob, err := b.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if d := memMaxAbsDiff(oa, ob); d != 0 {
		t.Errorf("same seed non-deterministic, max diff %.3g", d)
	}
}

// ExampleMemorizingAttention builds a memorizing-attention layer, stores an
// earlier segment's keys/values in the external memory, and runs a later
// segment through it — the layer attends both locally and over the retrieved
// memory (arXiv:2203.08913).
func ExampleMemorizingAttention() {
	m, err := NewMemorizingAttention(tensor.F64, 8, 2, 42, WithMemorizingTopK(4), WithMemorizingMemorySize(64))
	if err != nil {
		panic(err)
	}
	// Store a past segment's (key,value) pairs in the non-differentiable memory.
	past := tensor.New(tensor.F64, tensor.Shape{16, 8})
	for i := range 16 {
		for j := range 8 {
			past.SetF64(0.1*float64((i+j)%5), i, j)
		}
	}
	if err := m.AddSegment(past, past); err != nil {
		panic(err)
	}
	// Run a later segment: local attention + retrieval over the 16 stored pairs.
	x := tensor.New(tensor.F64, tensor.Shape{4, 8})
	for i := range 4 {
		x.SetF64(1, i, i)
	}
	out, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		panic(err)
	}
	fmt.Printf("memory=%d/%d out=%v params=%d\n", m.Memory.Size(), m.Memory.Cap(), out.Shape(), len(m.Params()))
	// Output:
	// memory=16/64 out=(4, 8) params=5
}
