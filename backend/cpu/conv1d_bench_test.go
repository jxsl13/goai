package cpu

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// conv1dScalarControlKernel freezes the pre-tile traversal for same-binary policy
// benchmarks. It deliberately mirrors the old kernel's allocation, parallelism,
// accumulation order, and single output rounding.
func conv1dScalarControlKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	x, w := in[0], in[1]
	var bias *tensor.Tensor
	if len(in) == 3 {
		bias = in[2]
	}
	L, D, K := x.Shape()[0], x.Shape()[1], w.Shape()[1]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	switch x.Dtype() {
	case tensor.F64:
		xs, ws, os := x.Contiguous().Storage().F64(), w.Contiguous().Storage().F64(), out.Storage().F64()
		var bs []float64
		if bias != nil {
			bs = bias.Contiguous().Storage().F64()
		}
		parallelWork(L, D*K, func(loT, hiT int) {
			for t := loT; t < hiT; t++ {
				kStart := 0
				if lo := (K - 1) - t; lo > 0 {
					kStart = lo
				}
				for c := range D {
					var acc float64
					for k := kStart; k < K; k++ {
						acc += ws[c*K+k] * xs[(t-(K-1)+k)*D+c]
					}
					if bias != nil {
						acc += bs[c]
					}
					os[t*D+c] = acc
				}
			}
		})
	case tensor.F32:
		xs, ws, os := x.Contiguous().Storage().F32(), w.Contiguous().Storage().F32(), out.Storage().F32()
		var bs []float32
		if bias != nil {
			bs = bias.Contiguous().Storage().F32()
		}
		parallelWork(L, D*K, func(loT, hiT int) {
			for t := loT; t < hiT; t++ {
				kStart := 0
				if lo := (K - 1) - t; lo > 0 {
					kStart = lo
				}
				for c := range D {
					var acc float64
					for k := kStart; k < K; k++ {
						acc += float64(ws[c*K+k]) * float64(xs[(t-(K-1)+k)*D+c])
					}
					if bias != nil {
						acc += float64(bs[c])
					}
					os[t*D+c] = float32(acc)
				}
			}
		})
	default:
		return nil, fmt.Errorf("control: unsupported dtype %v", x.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

var conv1dScalarControl = func() *Backend {
	b := &Backend{table: make(map[kernelKey]backend.Kernel)}
	b.add(backend.OpConv1D, tensor.F32, conv1dScalarControlKernel)
	b.add(backend.OpConv1D, tensor.F64, conv1dScalarControlKernel)
	return b
}()

func benchConv1DCPU(b *testing.B, dt tensor.Dtype, L, D, K int, bias, control bool) {
	rng := rand.New(rand.NewSource(1))
	fill := func(t *tensor.Tensor) *tensor.Tensor {
		if dt == tensor.F64 {
			s := t.Storage().F64()
			for i := range s {
				s[i] = rng.Float64() - 0.5
			}
		} else {
			s := t.Storage().F32()
			for i := range s {
				s[i] = rng.Float32() - 0.5
			}
		}
		return t
	}
	x := fill(tensor.New(dt, tensor.Shape{L, D}))
	w := fill(tensor.New(dt, tensor.Shape{D, K}))
	in := []*tensor.Tensor{x, w}
	if bias {
		in = append(in, fill(tensor.New(dt, tensor.Shape{D})))
	}
	be := std
	if control {
		be = conv1dScalarControl
	}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpConv1D, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConv1D_cpu_f64(b *testing.B) {
	benchConv1DCPU(b, tensor.F64, 2048, 1024, 4, true, false)
}
func BenchmarkConv1D_control_f64(b *testing.B) {
	benchConv1DCPU(b, tensor.F64, 2048, 1024, 4, true, true)
}
func BenchmarkConv1D_cpu_f32(b *testing.B) {
	benchConv1DCPU(b, tensor.F32, 2048, 1024, 4, true, false)
}
func BenchmarkConv1D_control_f32(b *testing.B) {
	benchConv1DCPU(b, tensor.F32, 2048, 1024, 4, true, true)
}

func BenchmarkConv1D_cpu_f32_L1_D1024_K4(b *testing.B) {
	benchConv1DCPU(b, tensor.F32, 1, 1024, 4, true, false)
}
func BenchmarkConv1D_control_f32_L1_D1024_K4(b *testing.B) {
	benchConv1DCPU(b, tensor.F32, 1, 1024, 4, true, true)
}
func BenchmarkConv1D_cpu_f32_L64_D1023_K4_no_bias(b *testing.B) {
	benchConv1DCPU(b, tensor.F32, 64, 1023, 4, false, false)
}
func BenchmarkConv1D_control_f32_L64_D1023_K4_no_bias(b *testing.B) {
	benchConv1DCPU(b, tensor.F32, 64, 1023, 4, false, true)
}
func BenchmarkConv1D_cpu_f32_L512_D256_K8(b *testing.B) {
	benchConv1DCPU(b, tensor.F32, 512, 256, 8, true, false)
}
func BenchmarkConv1D_control_f32_L512_D256_K8(b *testing.B) {
	benchConv1DCPU(b, tensor.F32, 512, 256, 8, true, true)
}
