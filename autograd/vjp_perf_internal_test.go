package autograd

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// benchUnaryVJP times a registered unary VJP over a 512×512 F32 tensor — the
// training backward path for activation gradients. Guards the §T632 devirtualization
// (typed flat loop, no per-element Unravel/AtF64/SetF64) against regressing back to
// the per-element path.
func benchUnaryVJP(b *testing.B, op backend.Op) {
	vjp := vjps[op]
	if vjp == nil {
		b.Skipf("no vjp for %v", op)
	}
	mk := func() *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape{512, 512})
		s := t.Storage().F32()
		for i := range s {
			s[i] = float32((i%17)-8) * 0.1
		}
		return t
	}
	x, y, g := mk(), mk(), mk()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := vjp(nil, []*tensor.Tensor{x}, []*tensor.Tensor{y}, nil, g); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUnaryVJPTanh(b *testing.B)    { benchUnaryVJP(b, backend.OpTanh) }
func BenchmarkUnaryVJPExp(b *testing.B)     { benchUnaryVJP(b, backend.OpExp) }
func BenchmarkUnaryVJPSigmoid(b *testing.B) { benchUnaryVJP(b, backend.OpSigmoid) }
