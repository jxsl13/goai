package cpu_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func benchCEBwdF64On(b *testing.B, name backend.Name, batch, c int) {
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
	gT := tensor.New(tensor.F64, tensor.Shape{})
	gT.SetF64(1)
	in := []*tensor.Tensor{z, tgt, gT}
	attr := backend.CrossEntropyAttrs{Reduction: backend.ReductionMean}
	ctx := backend.NewContext().WithBackend(be)
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpCrossEntropyBackward, in, attr); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCEBwdF64Ref_256x32000(b *testing.B) { benchCEBwdF64On(b, backend.Ref, 256, 32000) }
func BenchmarkCEBwdF64CPU_256x32000(b *testing.B) { benchCEBwdF64On(b, backend.CPU, 256, 32000) }
func BenchmarkCEBwdF64Ref_512x50000(b *testing.B) { benchCEBwdF64On(b, backend.Ref, 512, 50000) }
func BenchmarkCEBwdF64CPU_512x50000(b *testing.B) { benchCEBwdF64On(b, backend.CPU, 512, 50000) }
