package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestLLMInt8FlatParity checks the contiguous-F64 fast path is BIT-IDENTICAL to the
// per-element AtF64 fallback. The fallback is forced by feeding non-contiguous views
// (a transpose of a base holding the same logical values) so flatF64 returns nil.
// Covered: the no-outlier int8 path AND the mixed outlier + int8 path.
func TestLLMInt8FlatParity(t *testing.T) {
	const tokens, cin, cout = 33, 40, 24 // sizes not multiples of 4 → exercise the dot remainder

	// contiguous inputs (fast path)
	xc := tensor.New(tensor.F64, tensor.Shape{tokens, cin})
	wc := tensor.New(tensor.F64, tensor.Shape{cin, cout})
	xs, ws := xc.Storage().F64(), wc.Storage().F64()
	for i := range xs {
		xs[i] = math.Sin(float64(i)*0.11) * 3
	}
	for i := range ws {
		ws[i] = math.Cos(float64(i)*0.07) * 2
	}
	// plant a couple of outlier columns (large |x|) so a low threshold hits path (2a)
	for _, c := range []int{5, 19} {
		for tk := 0; tk < tokens; tk++ {
			xs[tk*cin+c] = 12 + float64(tk)*0.01
		}
	}

	// non-contiguous views with identical values: base[b,a] = M[a,b], then Transpose(0,1).
	viewLike := func(m *tensor.Tensor, rows, cols int) *tensor.Tensor {
		base := tensor.New(tensor.F64, tensor.Shape{cols, rows})
		bs, ms := base.Storage().F64(), m.Storage().F64()
		for a := 0; a < rows; a++ {
			for b := 0; b < cols; b++ {
				bs[b*rows+a] = ms[a*cols+b]
			}
		}
		v, err := base.Transpose(0, 1)
		if err != nil {
			t.Fatalf("transpose: %v", err)
		}
		if v.IsContiguous() {
			t.Fatal("view unexpectedly contiguous — fallback would not trigger")
		}
		return v
	}
	xv := viewLike(xc, tokens, cin)
	wv := viewLike(wc, cin, cout)

	for _, thr := range []float64{6.0, 1e9} { // 6.0 → outliers present; 1e9 → none
		yFast, oFast, err := nn.LLMInt8MatMul(xc, wc, thr)
		if err != nil {
			t.Fatalf("fast thr=%g: %v", thr, err)
		}
		ySlow, oSlow, err := nn.LLMInt8MatMul(xv, wv, thr)
		if err != nil {
			t.Fatalf("slow thr=%g: %v", thr, err)
		}
		if len(oFast) != len(oSlow) {
			t.Fatalf("thr=%g outIdx len %d vs %d", thr, len(oFast), len(oSlow))
		}
		for i := range oFast {
			if oFast[i] != oSlow[i] {
				t.Fatalf("thr=%g outIdx[%d] %d vs %d", thr, i, oFast[i], oSlow[i])
			}
		}
		ff, sf := yFast.Storage().F64(), ySlow.Storage().F64()
		for k := range ff {
			if math.Float64bits(ff[k]) != math.Float64bits(sf[k]) {
				t.Fatalf("thr=%g out[%d]: fast %v vs fallback %v (not bit-identical)", thr, k, ff[k], sf[k])
			}
		}
	}
}
