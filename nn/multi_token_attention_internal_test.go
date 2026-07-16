package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// mtaMakeMap builds a [r,c] F64 tensor from row-major data.
func mtaMakeMap(r, c int, data []float64) *tensor.Tensor {
	return tensor.FromFloat64(tensor.Shape{r, c}, data)
}

// §V16 anchor 3 (CONV correctness): the key-query convolution reproduces a
// hand-computed cross-correlation with the paper's centered-key / past-query
// padding — out[i,j] = Σ_{a,b} θ[a,b]·L[i+a−(CQ−1), j+b−⌊CK/2⌋], zero outside.
func TestMultiTokenAttentionKeyQueryConvHandCheck(t *testing.T) {
	const CQ, CK, T = 3, 3, 4
	kern := []float64{ // θ[a,b], a=query tap, b=key tap
		0.5, -1.0, 2.0,
		0.0, 3.0, -0.5,
		1.5, 0.25, -2.0,
	}
	L := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	get := func(d []float64, r, c, i, j int) float64 {
		if i < 0 || j < 0 || i >= r || j >= c {
			return 0
		}
		return d[i*c+j]
	}
	leftPad, topPad := CK/2, CQ-1 // 1, 2
	want := make([]float64, T*T)
	for i := range T {
		for j := range T {
			var s float64
			for a := range CQ {
				for b := range CK {
					s += kern[a*CK+b] * get(L, T, T, i+a-topPad, j+b-leftPad)
				}
			}
			want[i*T+j] = s
		}
	}
	m := &MultiTokenAttention{Heads: 1, CQ: CQ, CK: CK, CH: 1}
	m.KQKernel = mtaMakeMap(1, CQ*CK, kern) // [1,CQ,CK] row-major
	m.KQKernel, _ = m.KQKernel.Reshape(tensor.Shape{1, CQ, CK})
	got, err := m.keyQueryConv(backend.NewContext(), mtaMakeMap(T, T, L), 0, T)
	if err != nil {
		t.Fatal(err)
	}
	for i := range T {
		for j := range T {
			if g, w := got.AtF64(i, j), want[i*T+j]; math.Abs(g-w) > 1e-10 {
				t.Errorf("conv[%d,%d]=%.10g want %.10g", i, j, g, w)
			}
		}
	}
}

// §V16 anchor 1 (delta ⇒ identity): with θ a delta at the current-position tap
// [CQ−1,⌊CK/2⌋] the key-query convolution returns the logit map unchanged.
func TestMultiTokenAttentionKeyQueryConvDeltaIdentity(t *testing.T) {
	const CQ, CK, T = 4, 5, 5
	m := &MultiTokenAttention{Heads: 1, CQ: CQ, CK: CK, CH: 1}
	m.KQKernel = tensor.New(tensor.F64, tensor.Shape{1, CQ, CK})
	m.KQKernel.SetF64(1, 0, CQ-1, CK/2) // delta at current tap
	L := make([]float64, T*T)
	for i := range L {
		L[i] = float64(i)*0.7 - 3
	}
	got, err := m.keyQueryConv(backend.NewContext(), mtaMakeMap(T, T, L), 0, T)
	if err != nil {
		t.Fatal(err)
	}
	for i := range T {
		for j := range T {
			if g, w := got.AtF64(i, j), L[i*T+j]; math.Abs(g-w) > 1e-12 {
				t.Errorf("delta conv[%d,%d]=%.12g want %.12g", i, j, g, w)
			}
		}
	}
}

// §V16 anchor 3 (head conv): the dense within-group head mixing reproduces the
// hand-computed L”[o] = Σ_p HeadKernel[o,p]·map[group,p] (paper Eq. 7).
func TestMultiTokenAttentionHeadConvHandCheck(t *testing.T) {
	m := &MultiTokenAttention{Heads: 2, CQ: 1, CK: 1, CH: 2}
	w := []float64{0.5, -1.0, 2.0, 0.25} // [[0.5,-1],[2,0.25]]
	m.HeadKernel = mtaMakeMap(2, 2, w)
	L0 := mtaMakeMap(2, 2, []float64{1, 2, 3, 4})
	L1 := mtaMakeMap(2, 2, []float64{10, 20, 30, 40})
	out, err := m.headConv(backend.NewContext(), []*tensor.Tensor{L0, L1})
	if err != nil {
		t.Fatal(err)
	}
	d0 := []float64{1, 2, 3, 4}
	d1 := []float64{10, 20, 30, 40}
	for idx := range 4 {
		want0 := w[0]*d0[idx] + w[1]*d1[idx]
		want1 := w[2]*d0[idx] + w[3]*d1[idx]
		i, j := idx/2, idx%2
		if g := out[0].AtF64(i, j); math.Abs(g-want0) > 1e-12 {
			t.Errorf("head0[%d,%d]=%.12g want %.12g", i, j, g, want0)
		}
		if g := out[1].AtF64(i, j); math.Abs(g-want1) > 1e-12 {
			t.Errorf("head1[%d,%d]=%.12g want %.12g", i, j, g, want1)
		}
	}
}
