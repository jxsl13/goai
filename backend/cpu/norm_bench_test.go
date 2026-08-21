package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Guards for the norm backward kernels (§V22 A/B): the dγ/dβ column pass is
// row-tiled — each per-column sum keeps ref's ascending-row order (bit-identical
// values) but memory is walked row-major instead of one full stride-d column at
// a time.
func benchNormBwd(b *testing.B, op backend.Op) {
	be, _ := backend.Get(backend.CPU)
	rows, d := 512, 1024
	x := bench.RandF32(tensor.Shape{rows, d}, 1)
	gamma := bench.RandF32(tensor.Shape{d}, 2)
	g := bench.RandF32(tensor.Shape{rows, d}, 3)
	ctx := backend.NewContext().WithBackend(be)
	ins := []*tensor.Tensor{x, gamma, g}
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLayerNormBwdF32_512x1024(b *testing.B) {
	benchNormBwd(b, backend.OpLayerNormBackward)
}
func BenchmarkRMSNormBwdF32_512x1024(b *testing.B) { benchNormBwd(b, backend.OpRMSNormBackward) }

func benchNormFwd(b *testing.B, op backend.Op, nin, rows, d int) {
	be, _ := backend.Get(backend.CPU)
	x := bench.RandF32(tensor.Shape{rows, d}, 1)
	gamma := bench.RandF32(tensor.Shape{d}, 2)
	beta := bench.RandF32(tensor.Shape{d}, 3)
	ins := []*tensor.Tensor{x, gamma, beta}[:nin]
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, op, ins, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLayerNormFwdF32_512x1024(b *testing.B) {
	benchNormFwd(b, backend.OpLayerNorm, 3, 512, 1024)
}
func BenchmarkRMSNormFwdF32_512x1024(b *testing.B) {
	benchNormFwd(b, backend.OpRMSNorm, 2, 512, 1024)
}
func BenchmarkLayerNormFwdF32_512x4096(b *testing.B) {
	benchNormFwd(b, backend.OpLayerNorm, 3, 512, 4096)
}
func BenchmarkRMSNormFwdF32_512x4096(b *testing.B) {
	benchNormFwd(b, backend.OpRMSNorm, 2, 512, 4096)
}
