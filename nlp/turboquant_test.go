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
