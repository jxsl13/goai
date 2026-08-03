package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// reconErrMat's inner loop accumulates into a shared per-column buffer, which is what makes
// unroll-and-jam over the residual index possible: the accumulator is loaded once per group
// instead of once per element. Jamming claims to preserve the arithmetic exactly — each
// acc[s] still takes the same products in the same ascending i — and AWQ is a calibration
// search whose selected scale would absorb a small numeric change without any test noticing,
// so a digest is the only gate that would see one.
//
// The float64 return is hashed by BITS, not compared to a tolerance. Go may contract x*y+z
// into a fused multiply-add, and the jammed form has to contract the same way term for term
// or the digest moves — which is precisely the check worth having.
func reconErrDigest(t *testing.T, out, in, samples int, dt tensor.Dtype) uint64 {
	t.Helper()
	w := make([][]float64, out)
	for r := range w {
		w[r] = make([]float64, in)
		for i := range w[r] {
			w[r][i] = math.Sin(float64(r*in+i) * 0.013)
		}
	}
	wq := tensor.New(dt, tensor.Shape{out, in})
	x := tensor.New(dt, tensor.Shape{in, samples})
	fill := func(tn *tensor.Tensor, fn func(i int) float64) {
		n := tn.Numel()
		if dt == tensor.F64 {
			s := tn.Storage().F64()
			for i := range n {
				s[i] = fn(i)
			}
			return
		}
		s := tn.Storage().F32()
		for i := range n {
			s[i] = float32(fn(i))
		}
	}
	fill(wq, func(i int) float64 { return math.Round(math.Sin(float64(i)*0.013)*8) / 8 })
	fill(x, func(i int) float64 { return math.Cos(float64(i) * 0.021) })
	v := reconErrMat(w, wq, x, out, in, samples)
	h := uint64(14695981039346656037)
	u := math.Float64bits(v)
	for s := 0; s < 64; s += 8 {
		h = (h ^ (u>>s)&0xff) * 1099511628211
	}
	return h
}

func TestReconErrMatIsBitIdentical(t *testing.T) {
	// The sample counts straddle every remainder a jam of 2, 4 or 8 can leave: 13 is odd and
	// prime, 30 leaves 6 modulo 8, and 64 divides all three so the tail path is skipped.
	cases := []struct {
		out, in, samples int
		dt               tensor.Dtype
		want             uint64
	}{
		{7, 13, 13, tensor.F64, 977744709366639345},
		{7, 13, 13, tensor.F32, 7120259750929446520},
		{5, 30, 30, tensor.F64, 10025742202500828901},
		{5, 30, 30, tensor.F32, 2373790302529226826},
		{9, 64, 64, tensor.F64, 11298896412261288980},
		{3, 1, 5, tensor.F64, 11995421468116675637}, // in=1: the jammed loop never runs, only its tail
	}
	for _, c := range cases {
		got := reconErrDigest(t, c.out, c.in, c.samples, c.dt)
		if got != c.want {
			t.Errorf("out=%d in=%d samples=%d %v: digest %d", c.out, c.in, c.samples, c.dt, got)
		}
	}
}
