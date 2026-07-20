package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// Bit-identity + fallback gates for the contiguous typed-slice fast paths added
// to the quantization / specialized-layer per-element loops (the fillGen pattern,
// commit b8a7927). Each fast path keeps a verbatim copy of its pre-change loop as
// the oracle here; the gate asserts math.Float64bits equality (deterministic ops)
// across shapes and F32/F64, plus non-contiguous fallback correctness. A/B
// benchmarks (…Slow = oracle, …Fast = source) sit at the bottom.

// bitsEqual asserts a and b are bit-for-bit identical, element by element.
func bitsEqual(t *testing.T, a, b *tensor.Tensor, ctx string) {
	t.Helper()
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("%s: shape %v != %v", ctx, a.Shape(), b.Shape())
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, a.Shape())
		x, y := a.AtF64(idx...), b.AtF64(idx...)
		if math.Float64bits(x) != math.Float64bits(y) {
			t.Fatalf("%s: elem %d: fast %v != slow %v (not bit-identical)", ctx, i, x, y)
		}
	}
}

var fastPathShapes = []tensor.Shape{{1}, {7}, {4, 5}, {2, 3, 4}, {64, 48}}
var fastPathMats = []tensor.Shape{{1, 1}, {3, 7}, {8, 16}, {48, 64}}

// ---------- bitnet: BitQuantizeWeight absmean ternary fill ----------

func slowBitTernaryFill(w, q *tensor.Tensor) float64 {
	n := w.Numel()
	var sum float64
	for i := range n {
		sum += math.Abs(w.AtF64(tensor.Unravel(i, w.Shape())...))
	}
	gamma := sum / float64(n)
	scale := math.Max(gamma, bitnetScaleFloor)
	for i := range n {
		c := tensor.Unravel(i, w.Shape())
		t := math.Round(w.AtF64(c...) / scale)
		q.SetF64(scale*math.Max(-1, math.Min(1, t)), c...)
	}
	return gamma
}

func TestBitTernaryFillBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range fastPathShapes {
			w := tensor.Randn(dt, 11, shape)
			qf := tensor.New(dt, shape)
			qs := tensor.New(dt, shape)
			gf := bitTernaryFill(w, qf)
			gs := slowBitTernaryFill(w, qs)
			if math.Float64bits(gf) != math.Float64bits(gs) {
				t.Fatalf("dt=%v shape=%v gamma fast %v != slow %v", dt, shape, gf, gs)
			}
			bitsEqual(t, qf, qs, "bitTernaryFill dt="+dt.String()+" "+shape.String())
		}
	}
}

// A non-contiguous view (fallback) must quantize its logical values just like the
// contiguous materialization (fast path).
func TestBitTernaryFillNonContiguous(t *testing.T) {
	base := tensor.Randn(tensor.F64, 12, tensor.Shape{5, 4})
	view, err := base.Transpose(0, 1) // [4,5], non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	qView := tensor.New(tensor.F64, view.Shape())
	slowBitTernaryFill(view, qView) // reference via the generic accessor
	qFall := tensor.New(tensor.F64, view.Shape())
	if view.IsContiguous() {
		t.Fatal("expected a non-contiguous view")
	}
	bitTernaryFill(view, qFall) // must hit the fallback and match the reference
	bitsEqual(t, qFall, qView, "bitTernaryFill non-contiguous")
}

// F16 must fall THROUGH the typed switch to the generic path (never grab .F64()).
func TestBitTernaryFillF16FallThrough(t *testing.T) {
	shape := tensor.Shape{4, 5}
	w := tensor.New(tensor.F16, shape)
	for i := range w.Numel() {
		w.SetF64(float64(i%7)-3, tensor.Unravel(i, shape)...)
	}
	qf := tensor.New(tensor.F16, shape)
	qs := tensor.New(tensor.F16, shape)
	bitTernaryFill(w, qf) // must not panic; routes to generic path
	slowBitTernaryFill(w, qs)
	bitsEqual(t, qf, qs, "bitTernaryFill F16 fall-through")
}

// ---------- bitnet: BitQuantizeAct8 per-token 8-bit fill ----------

func slowBitAct8Fill(x, q *tensor.Tensor) {
	rows, cols := x.Shape()[0], x.Shape()[1]
	for r := range rows {
		absmax := 0.0
		for c := range cols {
			if a := math.Abs(x.AtF64(r, c)); a > absmax {
				absmax = a
			}
		}
		s := 127 / math.Max(absmax, bitnetScaleFloor)
		for c := range cols {
			v := math.Max(-128, math.Min(127, math.Round(x.AtF64(r, c)*s)))
			q.SetF64(v/s, r, c)
		}
	}
}

func TestBitAct8FillBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range fastPathMats {
			x := tensor.Randn(dt, 13, shape)
			qf := tensor.New(dt, shape)
			qs := tensor.New(dt, shape)
			bitAct8Fill(x, qf)
			slowBitAct8Fill(x, qs)
			bitsEqual(t, qf, qs, "bitAct8Fill dt="+dt.String()+" "+shape.String())
		}
	}
}

func TestBitAct8FillNonContiguous(t *testing.T) {
	base := tensor.Randn(tensor.F32, 14, tensor.Shape{16, 8})
	view, err := base.Transpose(0, 1) // [8,16], non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	qFall := tensor.New(tensor.F32, view.Shape())
	qRef := tensor.New(tensor.F32, view.Shape())
	bitAct8Fill(view, qFall)
	slowBitAct8Fill(view, qRef)
	bitsEqual(t, qFall, qRef, "bitAct8Fill non-contiguous")
}

// ---------- lsq: LSQ straight-through round ----------

func slowLSQRound(vbar *tensor.Tensor) *tensor.Tensor {
	rounded := tensor.New(vbar.Dtype(), vbar.Shape())
	for i := range vbar.Numel() {
		c := tensor.Unravel(i, vbar.Shape())
		rounded.SetF64(math.Round(vbar.AtF64(c...)), c...)
	}
	return rounded
}

func TestLSQRoundBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range fastPathShapes {
			v := tensor.Randn(dt, 15, shape)
			bitsEqual(t, lsqRound(v), slowLSQRound(v), "lsqRound dt="+dt.String()+" "+shape.String())
		}
	}
}

func TestLSQRoundNonContiguous(t *testing.T) {
	base := tensor.Randn(tensor.F64, 16, tensor.Shape{5, 4})
	view, err := base.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	bitsEqual(t, lsqRound(view), slowLSQRound(view), "lsqRound non-contiguous")
}

// ---------- ngpt: normalizeColumns ----------

func slowNormalizeColumns(w *tensor.Tensor) {
	in, out := w.Shape()[0], w.Shape()[1]
	for j := range out {
		var ss float64
		for i := range in {
			v := w.AtF64(i, j)
			ss += v * v
		}
		n := math.Sqrt(ss)
		if n == 0 {
			continue
		}
		for i := range in {
			w.SetF64(w.AtF64(i, j)/n, i, j)
		}
	}
}

func TestNormalizeColumnsBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range fastPathMats {
			a := tensor.Randn(dt, 17, shape) // same seed → identical data
			b := tensor.Randn(dt, 17, shape)
			normalizeColumns(a)
			slowNormalizeColumns(b)
			bitsEqual(t, a, b, "normalizeColumns dt="+dt.String()+" "+shape.String())
		}
	}
}

// ---------- si: SI.Accumulate inner update ----------

func slowSIAccumulate(p, grad *tensor.Tensor, omega, prev []float64) {
	for i := range p.Numel() {
		c := tensor.Unravel(i, p.Shape())
		cur := p.AtF64(c...)
		dTheta := cur - prev[i]
		omega[i] += -grad.AtF64(c...) * dTheta
		prev[i] = cur
	}
}

func slicesBitEqual(t *testing.T, a, b []float64, ctx string) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: len %d != %d", ctx, len(a), len(b))
	}
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			t.Fatalf("%s: elem %d: fast %v != slow %v", ctx, i, a[i], b[i])
		}
	}
}

func TestSIAccumulateBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range fastPathShapes {
			p := tensor.Randn(dt, 18, shape)
			g := tensor.Randn(dt, 19, shape)
			n := p.Numel()
			// non-trivial prev (so Δθ ≠ 0) and a pre-existing omega to accumulate onto
			prev0 := make([]float64, n)
			om0 := make([]float64, n)
			for i := range n {
				prev0[i] = 0.01 * float64(i%5-2)
				om0[i] = 0.001 * float64(i%3)
			}
			omF, prF := append([]float64(nil), om0...), append([]float64(nil), prev0...)
			omS, prS := append([]float64(nil), om0...), append([]float64(nil), prev0...)
			siAccumulate(p, g, omF, prF)
			slowSIAccumulate(p, g, omS, prS)
			slicesBitEqual(t, omF, omS, "siAccumulate omega dt="+dt.String()+" "+shape.String())
			slicesBitEqual(t, prF, prS, "siAccumulate prev dt="+dt.String()+" "+shape.String())
		}
	}
}

func TestSIAccumulateNonContiguous(t *testing.T) {
	base := tensor.Randn(tensor.F64, 20, tensor.Shape{5, 4})
	p, err := base.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	gbase := tensor.Randn(tensor.F64, 21, tensor.Shape{5, 4})
	g, err := gbase.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	n := p.Numel()
	omF, prF := make([]float64, n), make([]float64, n)
	omS, prS := make([]float64, n), make([]float64, n)
	siAccumulate(p, g, omF, prF)
	slowSIAccumulate(p, g, omS, prS)
	slicesBitEqual(t, omF, omS, "siAccumulate non-contiguous omega")
	slicesBitEqual(t, prF, prS, "siAccumulate non-contiguous prev")
}

// ---------- focal: SigmoidFocalLoss per-element constants ----------

func slowSigmoidFocalConstants(sign, alphaT, targets *tensor.Tensor, alpha float64) {
	for i := range targets.Numel() {
		c := tensor.Unravel(i, targets.Shape())
		y := targets.AtF64(c...)
		sign.SetF64(1-2*y, c...)
		switch {
		case alpha < 0:
			alphaT.SetF64(1, c...)
		case y == 1:
			alphaT.SetF64(alpha, c...)
		default:
			alphaT.SetF64(1-alpha, c...)
		}
	}
}

func binaryTargets(dt tensor.Dtype, shape tensor.Shape) *tensor.Tensor {
	tg := tensor.New(dt, shape)
	for i := range tg.Numel() {
		tg.SetF64(float64(i%2), tensor.Unravel(i, shape)...) // 0,1,0,1,…
	}
	return tg
}

func TestSigmoidFocalConstantsBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, alpha := range []float64{0.25, -1} { // α-weighted and α-disabled branches
			for _, shape := range fastPathShapes {
				tg := binaryTargets(dt, shape)
				sf, af := tensor.New(dt, shape), tensor.New(dt, shape)
				ss, as := tensor.New(dt, shape), tensor.New(dt, shape)
				fillSigmoidFocalConstants(sf, af, tg, alpha)
				slowSigmoidFocalConstants(ss, as, tg, alpha)
				bitsEqual(t, sf, ss, "sigfocal sign dt="+dt.String()+" "+shape.String())
				bitsEqual(t, af, as, "sigfocal alphaT dt="+dt.String()+" "+shape.String())
			}
		}
	}
}

// Mismatched sign/target dtypes must fall through (sign is logits-dtype, targets
// may differ); verify the fallback still matches the oracle.
func TestSigmoidFocalConstantsDtypeMismatchFallThrough(t *testing.T) {
	shape := tensor.Shape{4, 5}
	tg := binaryTargets(tensor.F64, shape)                                 // targets F64
	sf, af := tensor.New(tensor.F32, shape), tensor.New(tensor.F32, shape) // sign/alphaT F32
	ss, as := tensor.New(tensor.F32, shape), tensor.New(tensor.F32, shape)
	fillSigmoidFocalConstants(sf, af, tg, 0.25)
	slowSigmoidFocalConstants(ss, as, tg, 0.25)
	bitsEqual(t, sf, ss, "sigfocal dtype-mismatch sign")
	bitsEqual(t, af, as, "sigfocal dtype-mismatch alphaT")
}

// ---------- quant_linear: asF32 ----------

func slowAsF32(x *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(tensor.F32, x.Shape())
	dst := out.Storage().F32()
	for i := range x.Numel() {
		dst[i] = float32(x.AtF64(tensor.Unravel(i, x.Shape())...))
	}
	return out
}

func TestAsF32BitIdentical(t *testing.T) {
	for _, shape := range fastPathShapes {
		x := tensor.Randn(tensor.F64, 22, shape) // F64 → the fast path
		bitsEqual(t, asF32(x), slowAsF32(x), "asF32 F64 "+shape.String())
	}
}

func TestAsF32NonContiguousF64(t *testing.T) {
	base := tensor.Randn(tensor.F64, 23, tensor.Shape{5, 4})
	view, err := base.Transpose(0, 1)
	if err != nil {
		t.Fatal(err)
	}
	bitsEqual(t, asF32(view), slowAsF32(view), "asF32 non-contiguous F64")
}

// ---------- mixup: PermuteRows ----------

func slowPermuteRows(t *tensor.Tensor, perm []int) *tensor.Tensor {
	out := tensor.New(t.Dtype(), t.Shape())
	if t.Ndim() == 0 {
		out.SetF64(t.AtF64())
		return out
	}
	n := t.Numel()
	for i := range n {
		idx := tensor.Unravel(i, t.Shape())
		src := append([]int(nil), idx...)
		src[0] = perm[idx[0]]
		out.SetF64(t.AtF64(src...), idx...)
	}
	return out
}

func TestPermuteRowsBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, shape := range []tensor.Shape{{7}, {4, 5}, {3, 2, 4}, {8, 32}} {
			src := tensor.Randn(dt, 24, shape)
			b := shape[0]
			perm := make([]int, b)
			for i := range perm {
				perm[i] = (i*3 + 1) % b
			}
			bitsEqual(t, PermuteRows(src, perm), slowPermuteRows(src, perm), "PermuteRows dt="+dt.String()+" "+shape.String())
		}
	}
}

func TestPermuteRowsNonContiguous(t *testing.T) {
	base := tensor.Randn(tensor.F64, 25, tensor.Shape{5, 4})
	view, err := base.Transpose(0, 1) // [4,5], non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	perm := []int{3, 0, 2, 1}
	bitsEqual(t, PermuteRows(view, perm), slowPermuteRows(view, perm), "PermuteRows non-contiguous")
}

// ---------- kan: buildBasis ----------

func slowBuildBasis(l *KANLayer, x *tensor.Tensor) *tensor.Tensor {
	B := x.Shape()[0]
	nbasis := l.gridSize + l.splineOrder
	out := tensor.New(l.dtype, tensor.Shape{B, l.inDim, nbasis})
	scratch := make([]float64, len(l.knots))
	for b := range B {
		for i := range l.inDim {
			vals := evalBSplineBasis(x.AtF64(b, i), l.knots, l.splineOrder, scratch)
			for c := range nbasis {
				out.SetF64(vals[c], b, i, c)
			}
		}
	}
	return out
}

func TestBuildBasisBitIdentical(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		l, err := NewKAN(8, 4, 26, WithKANDtype(dt))
		if err != nil {
			t.Fatal(err)
		}
		x := tensor.Randn(dt, 27, tensor.Shape{5, 8})
		bitsEqual(t, l.buildBasis(x), slowBuildBasis(l, x), "buildBasis dt="+dt.String())
	}
}

func TestBuildBasisNonContiguous(t *testing.T) {
	l, err := NewKAN(4, 4, 28) // F64 default
	if err != nil {
		t.Fatal(err)
	}
	base := tensor.Randn(tensor.F64, 29, tensor.Shape{4, 6})
	view, err := base.Transpose(0, 1) // [6,4] → batch 6, in 4, non-contiguous
	if err != nil {
		t.Fatal(err)
	}
	bitsEqual(t, l.buildBasis(view), slowBuildBasis(l, view), "buildBasis non-contiguous")
}

// ===================== A/B benchmarks (Slow = oracle, Fast = source) =====================

func BenchmarkBitTernaryFillSlow(b *testing.B) {
	w := tensor.Randn(tensor.F32, 1, tensor.Shape{2048, 2048})
	q := tensor.New(tensor.F32, w.Shape())
	b.ResetTimer()
	for range b.N {
		slowBitTernaryFill(w, q)
	}
}
func BenchmarkBitTernaryFillFast(b *testing.B) {
	w := tensor.Randn(tensor.F32, 1, tensor.Shape{2048, 2048})
	q := tensor.New(tensor.F32, w.Shape())
	b.ResetTimer()
	for range b.N {
		bitTernaryFill(w, q)
	}
}

func BenchmarkBitAct8FillSlow(b *testing.B) {
	x := tensor.Randn(tensor.F32, 2, tensor.Shape{512, 4096})
	q := tensor.New(tensor.F32, x.Shape())
	b.ResetTimer()
	for range b.N {
		slowBitAct8Fill(x, q)
	}
}
func BenchmarkBitAct8FillFast(b *testing.B) {
	x := tensor.Randn(tensor.F32, 2, tensor.Shape{512, 4096})
	q := tensor.New(tensor.F32, x.Shape())
	b.ResetTimer()
	for range b.N {
		bitAct8Fill(x, q)
	}
}

func BenchmarkLSQRoundSlow(b *testing.B) {
	v := tensor.Randn(tensor.F32, 3, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for range b.N {
		slowLSQRound(v)
	}
}
func BenchmarkLSQRoundFast(b *testing.B) {
	v := tensor.Randn(tensor.F32, 3, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for range b.N {
		lsqRound(v)
	}
}

func BenchmarkNormalizeColumnsSlow(b *testing.B) {
	w := tensor.Randn(tensor.F32, 4, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for range b.N {
		slowNormalizeColumns(w)
	}
}
func BenchmarkNormalizeColumnsFast(b *testing.B) {
	w := tensor.Randn(tensor.F32, 4, tensor.Shape{2048, 2048})
	b.ResetTimer()
	for range b.N {
		normalizeColumns(w)
	}
}

func BenchmarkSIAccumulateSlow(b *testing.B) {
	p := tensor.Randn(tensor.F64, 6, tensor.Shape{2048, 2048})
	g := tensor.Randn(tensor.F64, 7, tensor.Shape{2048, 2048})
	om, pr := make([]float64, p.Numel()), make([]float64, p.Numel())
	b.ResetTimer()
	for range b.N {
		slowSIAccumulate(p, g, om, pr)
	}
}
func BenchmarkSIAccumulateFast(b *testing.B) {
	p := tensor.Randn(tensor.F64, 6, tensor.Shape{2048, 2048})
	g := tensor.Randn(tensor.F64, 7, tensor.Shape{2048, 2048})
	om, pr := make([]float64, p.Numel()), make([]float64, p.Numel())
	b.ResetTimer()
	for range b.N {
		siAccumulate(p, g, om, pr)
	}
}

func BenchmarkSigmoidFocalConstantsSlow(b *testing.B) {
	tg := binaryTargets(tensor.F32, tensor.Shape{256, 1024})
	sign, alphaT := tensor.New(tensor.F32, tg.Shape()), tensor.New(tensor.F32, tg.Shape())
	b.ResetTimer()
	for range b.N {
		slowSigmoidFocalConstants(sign, alphaT, tg, 0.25)
	}
}
func BenchmarkSigmoidFocalConstantsFast(b *testing.B) {
	tg := binaryTargets(tensor.F32, tensor.Shape{256, 1024})
	sign, alphaT := tensor.New(tensor.F32, tg.Shape()), tensor.New(tensor.F32, tg.Shape())
	b.ResetTimer()
	for range b.N {
		fillSigmoidFocalConstants(sign, alphaT, tg, 0.25)
	}
}

func BenchmarkAsF32Slow(b *testing.B) {
	x := tensor.Randn(tensor.F64, 5, tensor.Shape{512, 4096})
	b.ResetTimer()
	for range b.N {
		slowAsF32(x)
	}
}
func BenchmarkAsF32Fast(b *testing.B) {
	x := tensor.Randn(tensor.F64, 5, tensor.Shape{512, 4096})
	b.ResetTimer()
	for range b.N {
		asF32(x)
	}
}

func benchPerm(n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = (i*7 + 3) % n
	}
	return p
}

func BenchmarkPermuteRowsSlow(b *testing.B) {
	tg := tensor.Randn(tensor.F32, 10, tensor.Shape{256, 1000})
	p := benchPerm(256)
	b.ResetTimer()
	for range b.N {
		slowPermuteRows(tg, p)
	}
}
func BenchmarkPermuteRowsFast(b *testing.B) {
	tg := tensor.Randn(tensor.F32, 10, tensor.Shape{256, 1000})
	p := benchPerm(256)
	b.ResetTimer()
	for range b.N {
		PermuteRows(tg, p)
	}
}

func BenchmarkBuildBasisSlow(b *testing.B) {
	l, _ := NewKAN(256, 256, 1)
	x := tensor.Randn(tensor.F64, 9, tensor.Shape{256, 256})
	b.ResetTimer()
	for range b.N {
		slowBuildBasis(l, x)
	}
}
func BenchmarkBuildBasisFast(b *testing.B) {
	l, _ := NewKAN(256, 256, 1)
	x := tensor.Randn(tensor.F64, 9, tensor.Shape{256, 256})
	b.ResetTimer()
	for range b.N {
		l.buildBasis(x)
	}
}
