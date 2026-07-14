package nlp

import (
	"math"
	"testing"
)

// §T619 TurboQuant, rotation stage. The fixed random Π is orthogonal, so: QᵀQ = I, the
// transform round-trips (applyInverse∘apply = identity), and — the property that makes it safe
// to insert before attention — it PRESERVES inner products ⟨Πx,Πy⟩ = ⟨x,y⟩.

func TestPolarRotationOrthogonal(t *testing.T) {
	const d = 8
	p, err := newPolarRotation(d, 42)
	if err != nil {
		t.Fatal(err)
	}
	// QᵀQ should be the identity (columns orthonormal since rows are).
	for i := range d {
		for j := range d {
			var dot float64
			for k := range d {
				dot += p.q[k][i] * p.q[k][j]
			}
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(dot-want) > 1e-9 {
				t.Fatalf("QᵀQ[%d,%d]=%v, want %v", i, j, dot, want)
			}
		}
	}
}

func TestPolarRotationRoundTripAndInnerProduct(t *testing.T) {
	const d = 16
	p, err := newPolarRotation(d, 7)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, d)
	y := make([]float64, d)
	for i := range d {
		x[i] = math.Sin(float64(i)*0.7) + 0.3
		y[i] = math.Cos(float64(i)*0.4) - 0.2
	}
	rx, _ := p.apply(x)
	back, _ := p.applyInverse(rx)
	for i := range d {
		if math.Abs(back[i]-x[i]) > 1e-9 {
			t.Fatalf("round-trip[%d]=%v, want %v", i, back[i], x[i])
		}
	}
	// inner product preserved under rotation
	ry, _ := p.apply(y)
	var ipOrig, ipRot float64
	for i := range d {
		ipOrig += x[i] * y[i]
		ipRot += rx[i] * ry[i]
	}
	if math.Abs(ipOrig-ipRot) > 1e-9 {
		t.Fatalf("⟨Πx,Πy⟩=%v ≠ ⟨x,y⟩=%v", ipRot, ipOrig)
	}
}

// PolarQuant storage round-trip: quantize→dequantize preserves DIRECTION well (high cosine),
// improving with more bits — the TurboQuant property. A zero vector round-trips to zero.
func TestPolarQuantRoundTrip(t *testing.T) {
	const d = 128
	p, err := newPolarRotation(d, 3)
	if err != nil {
		t.Fatal(err)
	}
	// deterministic pseudo-random vector
	x := make([]float64, d)
	for i := range x {
		x[i] = math.Sin(float64(i)*1.3) + 0.4*math.Cos(float64(i)*0.2)
	}
	cosFor := func(b int) float64 {
		idx, norm, err := p.polarQuantize(x, b)
		if err != nil {
			t.Fatal(err)
		}
		xh, err := p.polarDequantize(idx, norm, b)
		if err != nil {
			t.Fatal(err)
		}
		var dot, nx, nxh float64
		for i := range x {
			dot += x[i] * xh[i]
			nx += x[i] * x[i]
			nxh += xh[i] * xh[i]
		}
		return dot / (math.Sqrt(nx) * math.Sqrt(nxh))
	}
	c1, c2 := cosFor(1), cosFor(2)
	if c2 <= c1 {
		t.Fatalf("more bits should reconstruct better: cos b=2 %.3f !> b=1 %.3f", c2, c1)
	}
	if c2 < 0.9 {
		t.Fatalf("b=2 cosine %.3f too low (expect ≈0.94)", c2)
	}

	// zero vector → zero reconstruction
	zero := make([]float64, d)
	idx, norm, _ := p.polarQuantize(zero, 2)
	if norm != 0 {
		t.Fatalf("zero vector norm should be 0, got %v", norm)
	}
	xh, _ := p.polarDequantize(idx, norm, 2)
	for i := range xh {
		if xh[i] != 0 {
			t.Fatalf("zero round-trip nonzero at %d: %v", i, xh[i])
		}
	}
}

func TestPolarQuantErrors(t *testing.T) {
	p, _ := newPolarRotation(4, 1)
	if _, _, err := p.polarQuantize([]float64{1, 2, 3, 4}, 3); err == nil {
		t.Fatal("b=3 should error (only b=1,2 supported)")
	}
	if _, err := p.polarDequantize([]int{0, 0, 0}, 1, 2); err == nil {
		t.Fatal("wrong index length should error")
	}
}

// §T619 QJL residual: the 1-bit sketch makes the attention inner-product estimate UNBIASED. At
// 1-bit PolarQuant the score is heavily biased (most of the residual is dropped); adding the QJL
// residual and averaging over the sketch randomness S recovers the true ⟨q,k⟩. (Any single S is
// noisy — unbiasedness is the property softmax-over-many-keys relies on, §T619 finding.)
func TestQJLUnbiasedInnerProduct(t *testing.T) {
	const d = 64
	p, err := newPolarRotation(d, 3)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float64, d)
	y := make([]float64, d)
	for i := range d {
		x[i] = math.Sin(float64(i)*1.1) - 0.3
		y[i] = math.Cos(float64(i)*0.6) + 0.2
	}
	var ipTrue float64
	for i := range d {
		ipTrue += x[i] * y[i]
	}
	idx, norm, _ := p.polarQuantize(x, 1)
	xhPolar, _ := p.polarDequantize(idx, norm, 1)
	var ipPolar float64
	for i := range d {
		ipPolar += xhPolar[i] * y[i]
	}
	// residual (rotated-unit space)
	var nx float64
	for _, v := range x {
		nx += v * v
	}
	nx = math.Sqrt(nx)
	u := make([]float64, d)
	for i := range d {
		u[i] = x[i] / nx
	}
	ru, _ := p.apply(u)
	cb, _ := polarCodebook(1, d)
	r := make([]float64, d)
	for i := range d {
		r[i] = ru[i] - cb[idx[i]]
	}
	// QJL-corrected estimate averaged over sketch seeds → unbiased
	const seeds = 2000
	var mean float64
	for sd := range seeds {
		q := newQJLSketch(d, uint64(sd)+1)
		signs, rn := q.encode(r)
		res := q.decodeResidual(signs, rn)
		ruT := make([]float64, d)
		for i := range d {
			ruT[i] = cb[idx[i]] + res[i]
		}
		uT, _ := p.applyInverse(ruT)
		var ip float64
		for i := range d {
			ip += norm * uT[i] * y[i]
		}
		mean += ip
	}
	mean /= seeds
	// the QJL-corrected mean must be much closer to the true score than biased polar-only.
	if math.Abs(mean-ipTrue) > 0.15*math.Abs(ipPolar-ipTrue) {
		t.Fatalf("QJL not debiasing: |mean-true|=%.4f vs |polar-true|=%.4f (true=%.3f)", math.Abs(mean-ipTrue), math.Abs(ipPolar-ipTrue), ipTrue)
	}
}
