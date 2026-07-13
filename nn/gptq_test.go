package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// reconErr returns ‖(W − Ŵ)·X‖_F, the layer output reconstruction error GPTQ minimizes.
func reconErr(w, wq, x *tensor.Tensor) float64 {
	out, in := w.Shape()[0], w.Shape()[1]
	samples := x.Shape()[1]
	var ss float64
	for r := range out {
		for s := range samples {
			var v float64
			for i := range in {
				v += (w.AtF64(r, i) - wq.AtF64(r, i)) * x.AtF64(i, s)
			}
			ss += v * v
		}
	}
	return math.Sqrt(ss)
}

// rtn quantizes each weight independently (round-to-nearest, no compensation).
func rtn(w *tensor.Tensor, quant func(float64) float64) *tensor.Tensor {
	out, in := w.Shape()[0], w.Shape()[1]
	q := tensor.New(w.Dtype(), tensor.Shape{out, in})
	for r := range out {
		for i := range in {
			q.SetF64(quant(w.AtF64(r, i)), r, i)
		}
	}
	return q
}

// §V16 tier-1 (the defining property): on correlated calibration inputs GPTQ achieves a
// strictly LOWER output reconstruction error than round-to-nearest at the same grid,
// because it compensates rounding error using the inter-column (Hessian) correlations.
func TestGPTQBeatsRTN(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	out, in, samples := 5, 8, 40
	w := tensor.New(tensor.F64, tensor.Shape{out, in})
	for r := range out {
		for i := range in {
			w.SetF64(rng.NormFloat64()*0.5, r, i) // weights in ~[-1,1]
		}
	}
	// correlated inputs: each column shares a common component (non-diagonal Hessian)
	x := tensor.New(tensor.F64, tensor.Shape{in, samples})
	shared := make([]float64, samples)
	for s := range samples {
		shared[s] = rng.NormFloat64()
	}
	for i := range in {
		for s := range samples {
			x.SetF64(shared[s]+0.4*rng.NormFloat64(), i, s)
		}
	}
	quant := nn.UniformQuantizer(5, -1, 1) // coarse grid → rounding error matters

	wq, err := nn.GPTQuantize(w, x, quant)
	if err != nil {
		t.Fatal(err)
	}
	gptqErr := reconErr(w, wq, x)
	rtnErr := reconErr(w, rtn(w, quant), x)
	if gptqErr >= rtnErr {
		t.Errorf("GPTQ error %.6g not < RTN error %.6g", gptqErr, rtnErr)
	}
	// the compensation should give a clear improvement on correlated data
	if gptqErr > 0.9*rtnErr {
		t.Errorf("GPTQ error %.6g barely below RTN %.6g (expected a clear margin)", gptqErr, rtnErr)
	}
}

// §V16 tier-1: with an identity (diagonal) Hessian the off-diagonal error propagation
// vanishes, so GPTQ reduces exactly to round-to-nearest.
func TestGPTQDiagonalEqualsRTN(t *testing.T) {
	n := 6
	w := tensor.New(tensor.F64, tensor.Shape{4, n})
	for r := range 4 {
		for i := range n {
			w.SetF64(float64((r*7+i*3)%11)/5-1, r, i) // spread of values
		}
	}
	x := tensor.New(tensor.F64, tensor.Shape{n, n}) // X = I ⇒ H = 2I (diagonal)
	for i := range n {
		x.SetF64(1, i, i)
	}
	quant := nn.UniformQuantizer(7, -1, 1)

	wq, err := nn.GPTQuantize(w, x, quant)
	if err != nil {
		t.Fatal(err)
	}
	rtnq := rtn(w, quant)
	for r := range 4 {
		for i := range n {
			if math.Abs(wq.AtF64(r, i)-rtnq.AtF64(r, i)) > 1e-9 {
				t.Errorf("diagonal H: Ŵ[%d,%d] = %g, want RTN %g", r, i, wq.AtF64(r, i), rtnq.AtF64(r, i))
			}
		}
	}
}

func TestGPTQErrors(t *testing.T) {
	w := tensor.New(tensor.F64, tensor.Shape{2, 3})
	if _, err := nn.GPTQuantize(w, tensor.New(tensor.F64, tensor.Shape{4, 5}), nn.UniformQuantizer(4, -1, 1)); err == nil {
		t.Error("mismatched input dim: want error")
	}
	if _, err := nn.GPTQuantize(w, tensor.New(tensor.F64, tensor.Shape{3}), nn.UniformQuantizer(4, -1, 1)); err == nil {
		t.Error("rank-1 x: want error")
	}
}

func ExampleGPTQuantize() {
	// Coarse 3-level grid {−1,0,1}. With correlated inputs, GPTQ does NOT round every
	// weight the same way RTN would ([1,1,1]); it redistributes to reduce output error.
	w := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{0.6, 0.6, 0.6})
	x := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 1, 1, 1, 1, 1}) // correlated columns
	wq, _ := nn.GPTQuantize(w, x, nn.UniformQuantizer(3, -1, 1))
	fmt.Printf("%.0f %.0f %.0f\n", wq.AtF64(0, 0), wq.AtF64(0, 1), wq.AtF64(0, 2))
	// Output:
	// 1 0 1
}
