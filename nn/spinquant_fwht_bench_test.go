package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSpinQuantizeActivations measures QuantizeActivations at a realistic LLM hidden dim
// (power-of-two, as Hadamard requires). The Hadamard variant applies the rotation as O(n log n)
// FWHT; origin/main did it as two dense O(rows·n^2) matmuls against the materialized R/Rᵀ.
func BenchmarkSpinQuantizeActivations(b *testing.B) {
	const dim, rows, levels = 4096, 8, 16
	s, err := nn.NewHadamardRotation(tensor.F64, dim, nn.WithSpinSeed(7))
	if err != nil {
		b.Fatal(err)
	}
	x := tensor.New(tensor.F64, tensor.Shape{rows, dim})
	xf := x.Storage().F64()
	for i := range xf {
		xf[i] = float64((i*131+7)%1000) / 500.0
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.QuantizeActivations(x, levels); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSpinFoldWeightFWHT measures the offline weight fold Rᵀ·W for the Hadamard variant, a
// column-wise O(out·n log n) FWHT on a non-recording context. The columns are independent, so the
// fast path fans them across cores and gathers/scatters through typed storage.
func BenchmarkSpinFoldWeightFWHT(b *testing.B) {
	const dim, out = 4096, 2048
	s, err := nn.NewHadamardRotation(tensor.F64, dim, nn.WithSpinSeed(7))
	if err != nil {
		b.Fatal(err)
	}
	w := tensor.New(tensor.F64, tensor.Shape{dim, out})
	wf := w.Storage().F64()
	for i := range wf {
		wf[i] = float64((i*131+7)%1000) / 500.0
	}
	ctx := backend.NewContext() // non-recording ⇒ FWHT fold path
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.FoldWeight(ctx, w); err != nil {
			b.Fatal(err)
		}
	}
}
