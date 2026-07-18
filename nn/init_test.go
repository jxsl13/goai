package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Xavier: all values within ±√(6/(fanIn+fanOut)), mean ≈ 0, deterministic.
func TestXavierUniform(t *testing.T) {
	const fanIn, fanOut = 30, 20
	bound := math.Sqrt(6.0 / float64(fanIn+fanOut))
	w := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.XavierUniform(w, fanIn, fanOut, 1)

	var sum float64
	n := w.Numel()
	for i := range n {
		v := w.AtF64(tensor.Unravel(i, w.Shape())...)
		if v < -bound || v > bound {
			t.Fatalf("value %v outside ±%v", v, bound)
		}
		sum += v
	}
	if mean := sum / float64(n); math.Abs(mean) > bound/5 {
		t.Errorf("mean %v too far from 0 (bound %v)", mean, bound)
	}

	// determinism (§V13): same seed → identical tensor
	w2 := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.XavierUniform(w2, fanIn, fanOut, 1)
	for i := range n {
		idx := tensor.Unravel(i, w.Shape())
		if w.AtF64(idx...) != w2.AtF64(idx...) {
			t.Fatal("same seed must reproduce identical init")
		}
	}
	// different seed → different tensor
	w3 := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.XavierUniform(w3, fanIn, fanOut, 2)
	same := true
	for i := range n {
		idx := tensor.Unravel(i, w.Shape())
		if w.AtF64(idx...) != w3.AtF64(idx...) {
			same = false
			break
		}
	}
	if same {
		t.Error("different seeds must differ")
	}
}

// Kaiming-normal: sample std ≈ √(2/fanIn) within 15% at n=6000.
func TestKaimingNormal(t *testing.T) {
	const fanIn, fanOut = 100, 60
	want := math.Sqrt(2.0 / float64(fanIn))
	w := tensor.New(tensor.F64, tensor.Shape{fanIn, fanOut})
	nn.KaimingNormal(w, fanIn, 7)

	var sum, sq float64
	n := w.Numel()
	for i := range n {
		v := w.AtF64(tensor.Unravel(i, w.Shape())...)
		sum += v
		sq += v * v
	}
	mean := sum / float64(n)
	std := math.Sqrt(sq/float64(n) - mean*mean)
	if math.Abs(std-want)/want > 0.15 {
		t.Errorf("std %v, want ~%v", std, want)
	}
	if math.Abs(mean) > want/5 {
		t.Errorf("mean %v too far from 0", mean)
	}
}

func TestZerosAndNewLinear(t *testing.T) {
	l := nn.NewLinear(tensor.F64, 4, 3, 99)
	for j := range 3 {
		if l.B.AtF64(j) != 0 {
			t.Error("bias must init to zero")
		}
	}
	if !l.W.Shape().Equal(tensor.Shape{4, 3}) {
		t.Errorf("W shape %v, want (4,3)", l.W.Shape())
	}
}

// --- TruncNormal ---

// truncMoments returns the theoretical mean and std of N(mean, std²) truncated
// to [a, b] (Johnson & Kotz, standard truncated-normal moments).
func truncMoments(mean, std, a, b float64) (m, s float64) {
	pdf := func(x float64) float64 { return math.Exp(-x*x/2) / math.Sqrt(2*math.Pi) }
	cdf := func(x float64) float64 { return 0.5 * (1 + math.Erf(x/math.Sqrt2)) }
	al, be := (a-mean)/std, (b-mean)/std
	z := cdf(be) - cdf(al)
	pa, pb := pdf(al), pdf(be)
	m = mean + std*(pa-pb)/z
	v := 1 + (al*pa-be*pb)/z - math.Pow((pa-pb)/z, 2)
	return m, std * math.Sqrt(v)
}

func sampleMoments(w *tensor.Tensor) (mean, std float64) {
	n := w.Numel()
	for i := range n {
		mean += w.AtF64(tensor.Unravel(i, w.Shape())...)
	}
	mean /= float64(n)
	for i := range n {
		d := w.AtF64(tensor.Unravel(i, w.Shape())...) - mean
		std += d * d
	}
	return mean, math.Sqrt(std / float64(n))
}

// TruncNormal: every value inside [a,b], AND the sample moments match the
// theoretical truncated-normal moments. The moment check is the real gate — a
// plain normal passes containment-free but must FAIL the moments (see the
// discriminator below).
func TestTruncNormalMoments(t *testing.T) {
	const n = 20000
	const mean, std, a, b = 0.0, 1.0, -2.0, 2.0
	w := tensor.New(tensor.F64, tensor.Shape{n})
	if err := nn.TruncNormal(w, 7,
		nn.WithTruncMean(mean), nn.WithTruncStd(std), nn.WithTruncBounds(a, b)); err != nil {
		t.Fatal(err)
	}

	for i := range n {
		v := w.AtF64(tensor.Unravel(i, w.Shape())...)
		if v < a || v > b {
			t.Fatalf("value %v outside [%v,%v]", v, a, b)
		}
	}

	wantMean, wantStd := truncMoments(mean, std, a, b)
	// Sanity-check the analytic reference against the known ±2σ values.
	if math.Abs(wantMean) > 1e-12 || math.Abs(wantStd-0.879626) > 1e-5 {
		t.Fatalf("reference moments wrong: mean %v std %v (want 0, 0.879626)", wantMean, wantStd)
	}
	// Tolerance from sample size: SE(mean) = σ/√n ≈ 0.0062; 5·SE ≈ 0.031.
	// SE(std) ≈ σ/√(2n) ≈ 0.0044; 5·SE ≈ 0.022. Both are fixed-seed, so this is
	// a deterministic assertion, not a flaky one.
	const tolMean, tolStd = 0.031, 0.022
	gotMean, gotStd := sampleMoments(w)
	if math.Abs(gotMean-wantMean) > tolMean {
		t.Errorf("mean %v, want %v ± %v", gotMean, wantMean, tolMean)
	}
	if math.Abs(gotStd-wantStd) > tolStd {
		t.Errorf("std %v, want %v ± %v", gotStd, wantStd, tolStd)
	}
}

// Discriminator: an UNTRUNCATED normal of the same parent std must fail the very
// checks TestTruncNormalMoments passes. This proves the moment gate is measuring
// truncation and not just "some numbers near zero" (§V22).
func TestTruncNormalDiscriminatesUntruncated(t *testing.T) {
	const n = 20000
	const mean, std, a, b = 0.0, 1.0, -2.0, 2.0
	w := tensor.New(tensor.F64, tensor.Shape{n})
	nn.KaimingNormal(w, 2, 7) // √(2/2) = 1 → N(0,1), the untruncated parent

	var outside int
	for i := range n {
		v := w.AtF64(tensor.Unravel(i, w.Shape())...)
		if v < a || v > b {
			outside++
		}
	}
	if outside == 0 {
		t.Fatal("untruncated normal produced no out-of-range values; containment gate is vacuous")
	}

	_, wantStd := truncMoments(mean, std, a, b)
	_, gotStd := sampleMoments(w)
	const tolStd = 0.022
	if math.Abs(gotStd-wantStd) <= tolStd {
		t.Fatalf("untruncated std %v passed the truncated-moment gate (want %v ± %v); gate is vacuous",
			gotStd, wantStd, tolStd)
	}
	t.Logf("discriminator ok: %d/%d outside [%v,%v]; untruncated std %.4f vs truncated %.4f",
		outside, n, a, b, gotStd, wantStd)
}

// Non-default mean/std/bounds must also hit their theoretical moments, and the
// window is ABSOLUTE (torch semantics) not in units of std.
func TestTruncNormalNonDefaultWindow(t *testing.T) {
	const n = 20000
	const mean, std, a, b = 1.0, 2.0, -0.5, 3.0
	w := tensor.New(tensor.F64, tensor.Shape{n})
	if err := nn.TruncNormal(w, 11,
		nn.WithTruncMean(mean), nn.WithTruncStd(std), nn.WithTruncBounds(a, b)); err != nil {
		t.Fatal(err)
	}
	for i := range n {
		if v := w.AtF64(tensor.Unravel(i, w.Shape())...); v < a || v > b {
			t.Fatalf("value %v outside [%v,%v]", v, a, b)
		}
	}
	wantMean, wantStd := truncMoments(mean, std, a, b)
	gotMean, gotStd := sampleMoments(w)
	if math.Abs(gotMean-wantMean) > 5*wantStd/math.Sqrt(n) {
		t.Errorf("mean %v, want %v", gotMean, wantMean)
	}
	if math.Abs(gotStd-wantStd) > 5*wantStd/math.Sqrt(2*n) {
		t.Errorf("std %v, want %v", gotStd, wantStd)
	}
}

func TestTruncNormalDeterminismAndValidation(t *testing.T) {
	shape := tensor.Shape{6, 5}
	a, b := tensor.New(tensor.F64, shape), tensor.New(tensor.F64, shape)
	if err := nn.TruncNormal(a, 3); err != nil {
		t.Fatal(err)
	}
	if err := nn.TruncNormal(b, 3); err != nil {
		t.Fatal(err)
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, shape)
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatal("same seed must reproduce identical init")
		}
	}
	c := tensor.New(tensor.F64, shape)
	if err := nn.TruncNormal(c, 4); err != nil {
		t.Fatal(err)
	}
	same := true
	for i := range a.Numel() {
		idx := tensor.Unravel(i, shape)
		if a.AtF64(idx...) != c.AtF64(idx...) {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different seed must change the init")
	}

	if err := nn.TruncNormal(a, 1, nn.WithTruncStd(0)); err == nil {
		t.Error("std=0 accepted")
	}
	if err := nn.TruncNormal(a, 1, nn.WithTruncStd(-1)); err == nil {
		t.Error("negative std accepted")
	}
	if err := nn.TruncNormal(a, 1, nn.WithTruncBounds(2, 2)); err == nil {
		t.Error("a==b accepted")
	}
	if err := nn.TruncNormal(a, 1, nn.WithTruncBounds(3, -3)); err == nil {
		t.Error("a>b accepted")
	}
}

// --- Orthogonal ---

// gramDeviation returns max|GᵀG − g²I| over the min(rows,cols) orthogonal
// directions: columns when tall/square, rows when wide.
func gramDeviation(w *tensor.Tensor, rows, cols int, gain float64) float64 {
	at := func(i, j int) float64 { return w.AtF64(tensor.Unravel(i*cols+j, w.Shape())...) }
	g2 := gain * gain
	var worst float64
	if rows >= cols { // WᵀW = g²I over cols
		for p := range cols {
			for q := range cols {
				var s float64
				for i := range rows {
					s += at(i, p) * at(i, q)
				}
				want := 0.0
				if p == q {
					want = g2
				}
				worst = math.Max(worst, math.Abs(s-want))
			}
		}
		return worst
	}
	for p := range rows { // WWᵀ = g²I over rows
		for q := range rows {
			var s float64
			for j := range cols {
				s += at(p, j) * at(q, j)
			}
			want := 0.0
			if p == q {
				want = g2
			}
			worst = math.Max(worst, math.Abs(s-want))
		}
	}
	return worst
}

// Orthogonal: WᵀW ≈ I (tall/square) or WWᵀ ≈ I (wide), to tight tolerance.
func TestOrthogonalGram(t *testing.T) {
	const tol = 1e-12
	cases := []struct {
		name  string
		shape tensor.Shape
		rows  int
		cols  int
	}{
		{"square", tensor.Shape{8, 8}, 8, 8},
		{"tall", tensor.Shape{12, 5}, 12, 5},
		{"wide", tensor.Shape{4, 11}, 4, 11},
		{"conv4d", tensor.Shape{6, 2, 3, 3}, 6, 18}, // trailing dims flatten → wide
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := tensor.New(tensor.F64, tc.shape)
			if err := nn.Orthogonal(w, 42); err != nil {
				t.Fatal(err)
			}
			if d := gramDeviation(w, tc.rows, tc.cols, 1); d > tol {
				t.Fatalf("max|GᵀG − I| = %g > %g", d, tol)
			} else {
				t.Logf("max|GᵀG − I| = %g", d)
			}
		})
	}
}

// Discriminator: a random-normal matrix must FAIL the Gram gate by a wide
// margin, proving the orthogonality assertion is not vacuous (§V22).
func TestOrthogonalDiscriminatesRandomNormal(t *testing.T) {
	const tol = 1e-12
	for _, tc := range []struct {
		name       string
		shape      tensor.Shape
		rows, cols int
	}{
		{"square", tensor.Shape{8, 8}, 8, 8},
		{"tall", tensor.Shape{12, 5}, 12, 5},
		{"wide", tensor.Shape{4, 11}, 4, 11},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := tensor.New(tensor.F64, tc.shape)
			nn.KaimingNormal(w, 2, 42) // N(0,1), NOT orthogonalized
			d := gramDeviation(w, tc.rows, tc.cols, 1)
			if d <= tol {
				t.Fatalf("random normal passed the orthogonality gate (dev %g); gate is vacuous", d)
			}
			if d < 0.1 {
				t.Fatalf("random-normal deviation %g suspiciously small; gate lacks margin", d)
			}
			t.Logf("discriminator ok: random-normal max|GᵀG − I| = %g", d)
		})
	}
}

// gain scales every singular value: WᵀW = gain²·I.
func TestOrthogonalGain(t *testing.T) {
	const gain = 1.4142135623730951 // √2, the ReLU convention
	w := tensor.New(tensor.F64, tensor.Shape{9, 4})
	if err := nn.Orthogonal(w, 5, nn.WithOrthogonalGain(gain)); err != nil {
		t.Fatal(err)
	}
	if d := gramDeviation(w, 9, 4, gain); d > 1e-12 {
		t.Fatalf("max|WᵀW − gain²I| = %g", d)
	}
	// gain=1 and gain=g must differ only by the scalar g.
	base := tensor.New(tensor.F64, tensor.Shape{9, 4})
	if err := nn.Orthogonal(base, 5); err != nil {
		t.Fatal(err)
	}
	for i := range w.Numel() {
		idx := tensor.Unravel(i, w.Shape())
		if got, want := w.AtF64(idx...), base.AtF64(idx...)*gain; math.Abs(got-want) > 1e-12 {
			t.Fatalf("gain scaling at %v: %v vs %v", idx, got, want)
		}
	}
}

func TestOrthogonalDeterminismAndValidation(t *testing.T) {
	shape := tensor.Shape{7, 4}
	a, b := tensor.New(tensor.F64, shape), tensor.New(tensor.F64, shape)
	if err := nn.Orthogonal(a, 9); err != nil {
		t.Fatal(err)
	}
	if err := nn.Orthogonal(b, 9); err != nil {
		t.Fatal(err)
	}
	for i := range a.Numel() {
		idx := tensor.Unravel(i, shape)
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatal("same seed must reproduce identical init")
		}
	}
	c := tensor.New(tensor.F64, shape)
	if err := nn.Orthogonal(c, 10); err != nil {
		t.Fatal(err)
	}
	same := true
	for i := range a.Numel() {
		idx := tensor.Unravel(i, shape)
		if a.AtF64(idx...) != c.AtF64(idx...) {
			same = false
			break
		}
	}
	if same {
		t.Fatal("different seed must change the init")
	}

	if err := nn.Orthogonal(tensor.New(tensor.F64, tensor.Shape{6}), 1); err == nil {
		t.Error("rank-1 tensor accepted")
	}
}
