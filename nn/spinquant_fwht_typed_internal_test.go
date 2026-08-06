package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// The typed+parallel F64 fast paths of applyFWHT / quantizeActivationsFWHT / foldWeightFWHT must be
// BIT-IDENTICAL to the original serial AtF64/SetF64 form: the per-element expression and the
// within-row/column ascending butterfly order are unchanged, rows/columns are disjoint, and the
// absmax is an order-independent MAX reduction. These guards recompute a serial reference on the
// same host (not a frozen cross-host digest) and require maxAbs == 0.

func serialApplyFWHTRef(s *SpinRotation, x *tensor.Tensor) []float64 {
	rows, n := x.Shape()[0], s.dim
	out := make([]float64, rows*n)
	buf := make([]float64, n)
	for i := 0; i < rows; i++ {
		for j := 0; j < n; j++ {
			buf[j] = x.AtF64(i, j)
		}
		fwhtInPlace(buf)
		for j := 0; j < n; j++ {
			out[i*n+j] = s.inv * s.signs[j] * buf[j]
		}
	}
	return out
}

func serialQuantizeActivationsFWHTRef(s *SpinRotation, x *tensor.Tensor, levels int) []float64 {
	rows, n := x.Shape()[0], s.dim
	rot := make([]float64, rows*n)
	buf := make([]float64, n)
	var absmax float64
	for i := 0; i < rows; i++ {
		for j := 0; j < n; j++ {
			buf[j] = x.AtF64(i, j)
		}
		fwhtInPlace(buf)
		for j := 0; j < n; j++ {
			v := s.inv * s.signs[j] * buf[j]
			rot[i*n+j] = v
			if a := math.Abs(v); a > absmax {
				absmax = a
			}
		}
	}
	if absmax == 0 {
		absmax = 1
	}
	q := UniformQuantizer(levels, -absmax, absmax)
	out := make([]float64, rows*n)
	for i := 0; i < rows; i++ {
		for j := 0; j < n; j++ {
			buf[j] = q(rot[i*n+j]) * s.signs[j]
		}
		fwhtInPlace(buf)
		for k := 0; k < n; k++ {
			out[i*n+k] = s.inv * buf[k]
		}
	}
	return out
}

func serialFoldWeightFWHTRef(s *SpinRotation, w *tensor.Tensor) []float64 {
	n, out := s.dim, w.Shape()[1]
	res := make([]float64, n*out)
	buf := make([]float64, n)
	for l := 0; l < out; l++ {
		for k := 0; k < n; k++ {
			buf[k] = w.AtF64(k, l)
		}
		fwhtInPlace(buf)
		for j := 0; j < n; j++ {
			res[j*out+l] = s.inv * s.signs[j] * buf[j]
		}
	}
	return res
}

func maxAbsDiff(a []float64, t *tensor.Tensor) float64 {
	st := t.Storage().F64()
	var m float64
	for i := range a {
		if d := math.Abs(a[i] - st[i]); d > m {
			m = d
		}
	}
	return m
}

func TestSpinFWHTTypedPathsBitExact(t *testing.T) {
	const dim, rows = 4096, 33 // 33: an odd row count that straddles the parallel chunk boundaries
	s, err := NewHadamardRotation(tensor.F64, dim, WithSpinSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(3, 4))
	x := tensor.New(tensor.F64, tensor.Shape{rows, dim})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = rng.NormFloat64() * 3
	}

	if got := s.applyFWHT(x); maxAbsDiff(serialApplyFWHTRef(s, x), got) != 0 {
		t.Errorf("applyFWHT F64 fast path not bit-exact vs serial: maxAbs=%g", maxAbsDiff(serialApplyFWHTRef(s, x), got))
	}

	gotQ, err := s.quantizeActivationsFWHT(x, 16)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(serialQuantizeActivationsFWHTRef(s, x, 16), gotQ); d != 0 {
		t.Errorf("quantizeActivationsFWHT F64 fast path not bit-exact vs serial: maxAbs=%g", d)
	}

	const wout = 2048
	w := tensor.New(tensor.F64, tensor.Shape{dim, wout})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = rng.NormFloat64()
	}
	if d := maxAbsDiff(serialFoldWeightFWHTRef(s, w), s.foldWeightFWHT(w)); d != 0 {
		t.Errorf("foldWeightFWHT F64 fast path not bit-exact vs serial: maxAbs=%g", d)
	}
}
