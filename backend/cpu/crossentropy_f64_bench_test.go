package cpu_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func benchCEF64On(b *testing.B, name backend.Name, batch, c int) {
	be, _ := backend.Get(name)
	rng := rand.New(rand.NewSource(1))
	z := tensor.New(tensor.F64, tensor.Shape{batch, c})
	zsl := z.Storage().F64()
	for i := range zsl {
		zsl[i] = rng.NormFloat64() * 3
	}
	tgt := tensor.New(tensor.F64, tensor.Shape{batch})
	for i := 0; i < batch; i++ {
		tgt.SetF64(float64(rng.Intn(c)), i)
	}
	in := []*tensor.Tensor{z, tgt}
	attr := backend.CrossEntropyAttrs{Reduction: backend.ReductionMean}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpCrossEntropy, in, attr); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCEF64Ref_256x32000(b *testing.B) { benchCEF64On(b, backend.Ref, 256, 32000) }
func BenchmarkCEF64CPU_256x32000(b *testing.B) { benchCEF64On(b, backend.CPU, 256, 32000) }
func BenchmarkCEF64Ref_512x50000(b *testing.B) { benchCEF64On(b, backend.Ref, 512, 50000) }
func BenchmarkCEF64CPU_512x50000(b *testing.B) { benchCEF64On(b, backend.CPU, 512, 50000) }
