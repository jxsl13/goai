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

// perRowRTN quantizes each output row with a symmetric levels-point grid over its own
// max-abs — exactly AWQ at α=0 (no channel scaling).
func perRowRTN(w *tensor.Tensor, levels int) *tensor.Tensor {
	out, in := w.Shape()[0], w.Shape()[1]
	q := tensor.New(w.Dtype(), tensor.Shape{out, in})
	for r := range out {
		var maxabs float64
		for i := range in {
			maxabs = math.Max(maxabs, math.Abs(w.AtF64(r, i)))
		}
		if maxabs == 0 {
			continue
		}
		grid := nn.UniformQuantizer(levels, -maxabs, maxabs)
		for i := range in {
			q.SetF64(grid(w.AtF64(r, i)), r, i)
		}
	}
	return q
}

func reconErrAWQ(w, wq, x *tensor.Tensor) float64 {
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

// §V16 tier-1 (the defining property): when some input channels carry much larger
// activations, AWQ's activation-aware scaling protects those salient weights and yields
// a strictly LOWER output error than per-row round-to-nearest at the same grid.
func TestAWQBeatsRTN(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 9))
	out, in, samples := 6, 8, 50
	w := tensor.New(tensor.F64, tensor.Shape{out, in})
	for r := range out {
		for i := range in {
			w.SetF64(rng.NormFloat64()*0.5, r, i)
		}
	}
	// skewed per-channel activation magnitudes: channels 0,1 are ~10× larger (salient).
	x := tensor.New(tensor.F64, tensor.Shape{in, samples})
	for i := range in {
		mag := 1.0
		if i < 2 {
			mag = 10
		}
		for s := range samples {
			x.SetF64(mag*rng.NormFloat64(), i, s)
		}
	}
	const levels = 5 // coarse grid

	wq, err := nn.AWQuantize(w, x, levels)
	if err != nil {
		t.Fatal(err)
	}
	awqErr := reconErrAWQ(w, wq, x)
	rtnErr := reconErrAWQ(w, perRowRTN(w, levels), x)
	if awqErr >= rtnErr {
		t.Errorf("AWQ error %.6g not < RTN error %.6g", awqErr, rtnErr)
	}
}

// §V16 tier-1: α=0 (no scaling) reproduces per-row round-to-nearest exactly.
func TestAWQAlphaZeroIsRTN(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 5))
	out, in, samples := 4, 6, 20
	w := tensor.New(tensor.F64, tensor.Shape{out, in})
	x := tensor.New(tensor.F64, tensor.Shape{in, samples})
	for r := range out {
		for i := range in {
			w.SetF64(rng.NormFloat64(), r, i)
		}
	}
	for i := range in {
		for s := range samples {
			x.SetF64(rng.NormFloat64()*float64(i+1), i, s)
		}
	}
	const levels = 7
	wq, err := nn.AWQuantize(w, x, levels, nn.WithAWQAlpha(0))
	if err != nil {
		t.Fatal(err)
	}
	rtn := perRowRTN(w, levels)
	for r := range out {
		for i := range in {
			if math.Abs(wq.AtF64(r, i)-rtn.AtF64(r, i)) > 1e-9 {
				t.Errorf("α=0 Ŵ[%d,%d] = %g, want RTN %g", r, i, wq.AtF64(r, i), rtn.AtF64(r, i))
			}
		}
	}
}

func TestAWQErrors(t *testing.T) {
	w := tensor.New(tensor.F64, tensor.Shape{2, 3})
	if _, err := nn.AWQuantize(w, tensor.New(tensor.F64, tensor.Shape{4, 5}), 4); err == nil {
		t.Error("mismatched input dim: want error")
	}
	if _, err := nn.AWQuantize(w, tensor.New(tensor.F64, tensor.Shape{3, 5}), 1); err == nil {
		t.Error("levels<2: want error")
	}
}

func ExampleAWQuantize() {
	// One salient channel (0) with 20× activation; AWQ protects its weight so the
	// output error beats plain rounding. Here we just confirm a valid quantization runs.
	w := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{0.9, 0.2, -0.5})
	x := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{20, 20, 1, 1, 1, 1})
	wq, _ := nn.AWQuantize(w, x, 3, nn.WithAWQGrid(20))
	// the salient channel 0 is quantized closer to its original value than a naive grid
	fmt.Printf("%.1f\n", wq.AtF64(0, 0))
	// Output:
	// 0.9
}
