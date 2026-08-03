package cpu_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkCholeskyCPU exists because OpCholesky had no cpu benchmark at all — the only
// existing one pins itself to the reference backend on purpose, so nothing measured the path
// every real caller takes.
func benchCholesky(b *testing.B, n int, ref bool) {
	a := spd(n, tensor.F64)
	ctx := backend.NewContext()
	if ref {
		ctx = ctx.WithBackend(backend.Reference())
	}
	in := []*tensor.Tensor{a}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpCholesky, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCholeskyCPU_256(b *testing.B) { benchCholesky(b, 256, false) }
func BenchmarkCholeskyCPU_512(b *testing.B) { benchCholesky(b, 512, false) }
func BenchmarkCholeskyRef_512(b *testing.B) { benchCholesky(b, 512, true) }
