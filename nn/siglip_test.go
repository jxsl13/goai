package nn

import (
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// sigLIPFixture is the fixed input pair (seeds 3/4) the torch golden was made from.
func sigLIPFixture() (*tensor.Tensor, *tensor.Tensor, *SigLIPHead) {
	img := tensor.Randn(tensor.F64, 3, tensor.Shape{4, 6})
	txt := tensor.Randn(tensor.F64, 4, tensor.Shape{4, 6})
	return img, txt, NewSigLIPHead(tensor.F64)
}

// TestSigLIPTorchParity (§V16 tier-1, §T591): loss AND gradients match the torch
// reference (manual formula per the paper's eq. 1) on identical inputs at f64.
// Golden generated via SIGLIP_DUMP + scratch torch script; rtol 1e-9.
func TestSigLIPTorchParity(t *testing.T) {
	img, txt, head := sigLIPFixture()
	if dump := os.Getenv("SIGLIP_DUMP"); dump != "" {
		if err := safetensors.SaveFile(dump, map[string]*tensor.Tensor{
			"img": img, "txt": txt, "lt": head.LogTemp, "b": head.Bias,
		}, nil); err != nil {
			t.Fatal(err)
		}
	}
	tape := autograd.NewTape()
	loss, err := SigLIPLoss(tape.Context(), img, txt, head)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	check := func(name string, got, want float64) {
		t.Helper()
		if tol := 1e-9 * math.Max(1, math.Abs(want)); math.Abs(got-want) > tol {
			t.Errorf("%s: got %.15g, want %.15g", name, got, want)
		}
	}
	// torch 2.x f64 goldens (see test comment above).
	check("loss", loss.AtF64(), 8.94023894271059)
	check("dLogTemp", tape.Grad(head.LogTemp).AtF64(), -0.922337813875307)
	check("dBias", tape.Grad(head.Bias).AtF64(), -0.976315333949366)
	check("dImg[0,0]", tape.Grad(img).Storage().F64()[0], -0.055757465031363)
}

// TestSigLIPLearnsPairing (§T591 e2e): training raw embeddings + head with the loss
// drives matched pairs together and unmatched apart — diagonal similarity ends far
// above off-diagonal, and the loss drops by an order of magnitude.
func TestSigLIPLearnsPairing(t *testing.T) {
	img := tensor.Randn(tensor.F64, 11, tensor.Shape{6, 8})
	txt := tensor.Randn(tensor.F64, 12, tensor.Shape{6, 8})
	head := NewSigLIPHead(tensor.F64)
	params := append([]*tensor.Tensor{img, txt}, head.Params()...)
	opt := NewAdamW(params, 0.05, 0)
	var first, last float64
	for step := range 200 {
		tape := autograd.NewTape()
		loss, err := SigLIPLoss(tape.Context(), img, txt, head)
		if err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = loss.AtF64()
		}
		last = loss.AtF64()
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	if last > first/10 {
		t.Fatalf("SigLIP did not learn: loss %.4f → %.4f", first, last)
	}
	// diagonal cosine similarity must dominate every off-diagonal in its row.
	dot := func(a, b []float64) float64 {
		var s, na, nb float64
		for k := range a {
			s += a[k] * b[k]
			na += a[k] * a[k]
			nb += b[k] * b[k]
		}
		return s / math.Sqrt(na*nb)
	}
	iv, tv := img.Storage().F64(), txt.Storage().F64()
	row := func(m []float64, r int) []float64 { return m[r*8 : (r+1)*8] }
	for i := range 6 {
		d := dot(row(iv, i), row(tv, i))
		for j := range 6 {
			if j != i && dot(row(iv, i), row(tv, j)) >= d {
				t.Fatalf("pair %d not separated from %d", i, j)
			}
		}
	}
}

// TestSigLIPValidation: shape and nil-head errors.
func TestSigLIPValidation(t *testing.T) {
	a := tensor.Randn(tensor.F64, 1, tensor.Shape{2, 3})
	b := tensor.Randn(tensor.F64, 2, tensor.Shape{3, 2})
	if _, err := SigLIPLoss(autograd.NewTape().Context(), a, b, NewSigLIPHead(tensor.F64)); err == nil {
		t.Error("shape mismatch must error")
	}
	if _, err := SigLIPLoss(autograd.NewTape().Context(), a, a, nil); err == nil {
		t.Error("nil head must error")
	}
}
