package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: the eigenbasis rotations are exact inverses (Q_L, Q_R orthogonal), so
// rotateBack(Q_L, rotateForward(Q_L, G, Q_R), Q_R) = G.
func TestSOAPRotationInverse(t *testing.T) {
	// symmetric PSD matrices → real orthogonal eigenbases
	l := [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}}
	r := [][]float64{{2, 0.5}, {0.5, 1}}
	ql, qr := eigenBasis(l), eigenBasis(r)
	g := [][]float64{{1, -2}, {0.5, 3}, {-1, 0.25}} // 3×2

	back := rotateBack(ql, rotateForward(ql, g, qr), qr)
	for i := range 3 {
		for j := range 2 {
			if math.Abs(back[i][j]-g[i][j]) > 1e-10 {
				t.Errorf("rotate round-trip [%d,%d] = %.10g, want %.10g", i, j, back[i][j], g[i][j])
			}
		}
	}
	// Q_L columns are orthonormal: QᵀQ = I
	for a := range 3 {
		for b := range 3 {
			var d float64
			for i := range 3 {
				d += ql[i][a] * ql[i][b]
			}
			want := 0.0
			if a == b {
				want = 1
			}
			if math.Abs(d-want) > 1e-10 {
				t.Errorf("(Q_LᵀQ_L)[%d,%d] = %.10g, want %.10g", a, b, d, want)
			}
		}
	}
}

func benchSOAPStep(b *testing.B, nParams, dim int) {
	rng := rand.New(rand.NewPCG(3, 4))
	params := make([]*tensor.Tensor, nParams)
	grads := make(map[*tensor.Tensor]*tensor.Tensor, nParams)
	for i := range params {
		p := tensor.New(tensor.F64, tensor.Shape{dim, dim})
		gv := tensor.New(tensor.F64, tensor.Shape{dim, dim})
		gs := gv.Storage().F64()
		for k := range gs {
			gs[k] = rng.NormFloat64()
		}
		params[i], grads[p] = p, gv
	}
	opt := NewSOAP(params, 1e-3, WithSOAPFreq(1)) // Freq=1: eigenbasis refresh every step
	gf := func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
	opt.Step(gf)
	b.ResetTimer()
	for range b.N {
		opt.Step(gf)
	}
}
func BenchmarkSOAPStep_8x512(b *testing.B) { benchSOAPStep(b, 8, 512) }
