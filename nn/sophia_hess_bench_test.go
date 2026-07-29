package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkSophiaUpdateHessian covers Sophia's per-step Hessian-EMA update over the
// standard 2-D param fixture.
func BenchmarkSophiaUpdateHessian(b *testing.B) {
	params, hess := stepOnlyFixture2D(tensor.F64)
	s := nn.NewSophia(params, 1e-3)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := s.UpdateHessian(hess); err != nil {
			b.Fatal(err)
		}
	}
}
