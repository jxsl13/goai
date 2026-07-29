package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func wandaInputs(cin, cout, tokens int) (w, x *tensor.Tensor) {
	w = tensor.New(tensor.F64, tensor.Shape{cin, cout})
	ws := w.Storage().F64()
	for i := range ws {
		ws[i] = math.Sin(float64(i)*0.001) * 0.5
	}
	x = tensor.New(tensor.F64, tensor.Shape{tokens, cin})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Cos(float64(i) * 0.0013)
	}
	return w, x
}

// BenchmarkWandaPrune covers unstructured Wanda pruning (WandaScore + actL2Norm +
// per-column select + kept/mask writes) over a transformer-sized weight, F64.
func BenchmarkWandaPrune(b *testing.B) {
	w, x := wandaInputs(2048, 2048, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := nn.WandaPrune(w, x, 0.5); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWandaPruneNM covers 2:4 structured Wanda pruning over the same weight.
func BenchmarkWandaPruneNM(b *testing.B) {
	w, x := wandaInputs(2048, 2048, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := nn.WandaPruneNM(w, x, 2, 4); err != nil {
			b.Fatal(err)
		}
	}
}

// wandaInputsF32 mirrors wandaInputs for the F32 typed branch, which had no benchmark and so
// could not be validated when the F64 path was panelled and switched to a selection.
func wandaInputsF32(cin, cout, tokens int) (w, x *tensor.Tensor) {
	w = tensor.New(tensor.F32, tensor.Shape{cin, cout})
	ws := w.Storage().F32()
	for i := range ws {
		ws[i] = float32(math.Sin(float64(i)*0.001) * 0.5)
	}
	x = tensor.New(tensor.F32, tensor.Shape{tokens, cin})
	xs := x.Storage().F32()
	for i := range xs {
		xs[i] = float32(math.Cos(float64(i) * 0.0013))
	}
	return w, x
}

func BenchmarkWandaPruneF32(b *testing.B) {
	w, x := wandaInputsF32(2048, 2048, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := nn.WandaPrune(w, x, 0.5); err != nil {
			b.Fatal(err)
		}
	}
}
