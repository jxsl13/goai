//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/tensor"
)

func BenchmarkExecuteMetalBinaryDispatch(b *testing.B) {
	if !metal.Available() {
		b.Skip("Metal device is unavailable")
	}
	a := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	c := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	inputs := []*tensor.Tensor{a, c}
	ctx := &backend.Context{Backend: metal.Backend{}}
	if _, err := backend.Execute(ctx, backend.OpAdd, inputs, nil); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpAdd, inputs, nil); err != nil {
			b.Fatal(err)
		}
	}
}
