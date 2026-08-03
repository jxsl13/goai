package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchDyTForward times DyT inference (ctx.Recorder == nil) at typical norm-layer
// shapes. DyT is a LayerNorm/RMSNorm drop-in applied ~2× per block, so its
// elementwise cost is on the per-token hot path.
func benchDyTForward(b *testing.B, dt tensor.Dtype, rows, d int) {
	l, err := nn.NewDyT(dt, d, 0)
	if err != nil {
		b.Fatal(err)
	}
	x := tensor.New(dt, tensor.Shape{rows, d})
	xs := x.Storage()
	switch dt {
	case tensor.F64:
		s := xs.F64()
		for i := range s {
			s[i] = float64((i*2654435761)&0xffff)/32768.0 - 1.0
		}
	case tensor.F32:
		s := xs.F32()
		for i := range s {
			s[i] = float32((i*2654435761)&0xffff)/32768.0 - 1.0
		}
	}
	ctx := backend.NewContext() // no recorder → inference path
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := l.Forward(ctx, x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDyTForward_F64_256x2048(b *testing.B) { benchDyTForward(b, tensor.F64, 256, 2048) }
func BenchmarkDyTForward_F32_512x4096(b *testing.B) { benchDyTForward(b, tensor.F32, 512, 4096) }
