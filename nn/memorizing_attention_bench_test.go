package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// BenchmarkMemGather covers memory retrieval + neighbour gather (retrieveHead sort
// + gather typed copy) for one head: memory=2048 rows, dim=512, headDim=64,
// query segment T=128, topM=32, F64.
func BenchmarkMemGather(b *testing.B) {
	const n, dim, headDim, tSeg, topM = 2048, 512, 64, 128, 32
	m := &MemMemory{dim: dim, cap: n}
	m.keys = make([][]float64, n)
	m.vals = make([][]float64, n)
	for i := range m.keys {
		kr := make([]float64, dim)
		vr := make([]float64, dim)
		for d := range kr {
			kr[d] = math.Sin(float64(i*dim+d) * 0.001)
			vr[d] = math.Cos(float64(i*dim+d) * 0.0013)
		}
		m.keys[i], m.vals[i] = kr, vr
	}
	qh := tensor.New(tensor.F64, tensor.Shape{tSeg, headDim})
	qs := qh.Storage().F64()
	for i := range qs {
		qs[i] = math.Sin(float64(i) * 0.007)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		kg, vg := m.gather(tensor.F64, qh, 0, headDim, topM)
		_, _ = kg, vg
	}
}
