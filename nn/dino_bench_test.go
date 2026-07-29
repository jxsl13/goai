package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkDINOCenterUpdate covers the DINO teacher-center EMA update over a realistic
// prototype head: batch B=256, K=16384 prototypes, F64.
func BenchmarkDINOCenterUpdate(b *testing.B) {
	const B, K = 256, 16384
	teach := tensor.New(tensor.F64, tensor.Shape{B, K})
	ts := teach.Storage().F64()
	for i := range ts {
		ts[i] = math.Sin(float64(i) * 0.0001)
	}
	center := tensor.New(tensor.F64, tensor.Shape{1, K})
	cs := center.Storage().F64()
	for i := range cs {
		cs[i] = math.Cos(float64(i) * 0.001)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.DINOCenterUpdate(center, teach, 0.9); err != nil {
			b.Fatal(err)
		}
	}
}
