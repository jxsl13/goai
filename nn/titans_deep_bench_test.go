package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// NeuralMemory.Scan (deep 2-layer MLP memory, the default/headline mode) dispatches ~33 backend
// ops per token. The inference path (nil Recorder) is the fused-path target.
func benchTitansScanDeep(b *testing.B, seq, dim, hidden int) {
	mk := func(rows, cols int, f func(i int) float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		s := t.Storage().F64()
		for i := range s {
			s[i] = f(i)
		}
		return t
	}
	m, err := nn.NewNeuralMemory(tensor.F64, dim, hidden, 7)
	if err != nil {
		b.Fatal(err)
	}
	q := mk(seq, dim, func(i int) float64 { return math.Sin(float64(i) * 0.01) })
	k := mk(seq, dim, func(i int) float64 { return math.Cos(float64(i) * 0.013) })
	v := mk(seq, dim, func(i int) float64 { return math.Sin(float64(i) * 0.017) })
	sig := func(i int) float64 { return 1 / (1 + math.Exp(-math.Sin(float64(i)*0.03))) }
	eta, theta, alpha := mk(seq, 1, sig), mk(seq, 1, sig), mk(seq, 1, sig)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.Scan(backend.NewContext(), q, k, v, eta, theta, alpha); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTitansScanDeep_256x64h128(b *testing.B) { benchTitansScanDeep(b, 256, 64, 128) }
func BenchmarkTitansScanDeep_512x96h192(b *testing.B) { benchTitansScanDeep(b, 512, 96, 192) }
