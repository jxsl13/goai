package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BenchmarkTIESMerge measures TIES merging of 3 fine-tunes over a critic-sized base
// (~131k params). The dominant cost is trimTopK's stable sort of every parameter by
// |task-vector|, run once per (model × param); slices.SortStableFunc replaces the
// reflection-based sort.SliceStable, and the base/task/output passes take the typed
// contiguous path instead of per-element AtF64/SetF64.
func BenchmarkTIESMerge(b *testing.B) {
	shapes := []tensor.Shape{{256, 256}, {256, 256}, {256, 1}}
	base := make([]*tensor.Tensor, len(shapes))
	for i, sh := range shapes {
		base[i] = tensor.New(tensor.F64, sh)
		bf := base[i].Storage().F64()
		for e := range bf {
			bf[e] = 0.01 * float64((i+e)%17-8)
		}
	}
	const M = 3
	models := make([][]*tensor.Tensor, M)
	for t := range models {
		models[t] = make([]*tensor.Tensor, len(shapes))
		for i, sh := range shapes {
			models[t][i] = tensor.New(tensor.F64, sh)
			mf := models[t][i].Storage().F64()
			for e := range mf {
				mf[e] = 0.01 * float64((t+i+e)%13-6)
			}
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := nn.TIESMerge(base, models, 0.5, 1.0); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTIESMergeManyTensors merges 3 fine-tunes over a realistic tensor count (32
// tensors of 128×128). Each tensor's top-k magnitude sort is compute-bound, so fanning
// the per-tensor work across cores scales better than a memory-bound merge.
func BenchmarkTIESMergeManyTensors(b *testing.B) {
	const M, nTensors = 3, 32
	base := make([]*tensor.Tensor, nTensors)
	for i := range base {
		base[i] = tensor.New(tensor.F64, tensor.Shape{128, 128})
		bf := base[i].Storage().F64()
		for e := range bf {
			bf[e] = 0.01 * float64((i+e)%17-8)
		}
	}
	models := make([][]*tensor.Tensor, M)
	for t := range models {
		models[t] = make([]*tensor.Tensor, nTensors)
		for i := range models[t] {
			models[t][i] = tensor.New(tensor.F64, tensor.Shape{128, 128})
			mf := models[t][i].Storage().F64()
			for e := range mf {
				mf[e] = 0.01 * float64((t+i+e)%13-6)
			}
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := nn.TIESMerge(base, models, 0.5, 1.0); err != nil {
			b.Fatal(err)
		}
	}
}
