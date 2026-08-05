package autograd_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchSSMBackward times the OpSSM VJP (selective-scan backward) at Mamba dims
// via a recording tape. D = d_inner (e·d_model) is the independent-channel axis;
// N = d_state. The backward dominates the forward here (reverse scan + forward
// recompute), so the forward+backward wall-clock tracks the VJP.
func benchSSMBackward(b *testing.B, L, D, N int) {
	mk := func(shape tensor.Shape, f func(i int) float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = f(i)
		}
		return t
	}
	u := mk(tensor.Shape{L, D}, func(i int) float64 { return math.Sin(float64(i) * 0.3) })
	delta := mk(tensor.Shape{L, D}, func(i int) float64 { return 0.5 + 0.25*math.Cos(float64(i)*0.1) })
	A := mk(tensor.Shape{D, N}, func(i int) float64 { return -math.Exp(-0.1 * float64(i%N)) }) // A<0
	B := mk(tensor.Shape{L, N}, func(i int) float64 { return math.Cos(float64(i) * 0.2) })
	C := mk(tensor.Shape{L, N}, func(i int) float64 { return math.Sin(float64(i) * 0.15) })
	skip := mk(tensor.Shape{D}, func(i int) float64 { return 0.01 * float64(i%7) })

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tape := autograd.NewTape()
		out, err := backend.Execute(tape.Context(), backend.OpSSM,
			[]*tensor.Tensor{u, delta, A, B, C, skip}, nil)
		if err != nil {
			b.Fatal(err)
		}
		if err := tape.Backward(out[0]); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSSMBackward_L512_D2048_N16(b *testing.B) { benchSSMBackward(b, 512, 2048, 16) }
func BenchmarkSSMBackward_L256_D1536_N16(b *testing.B) { benchSSMBackward(b, 256, 1536, 16) }
