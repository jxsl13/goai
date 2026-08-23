package gguf_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

var qmatmulPairApplySink *tensor.Tensor

func BenchmarkQMatMulPairApplyQ4K_M1_TinyLlamaFFN(b *testing.B) {
	const n, k = 5632, 2048
	weightData := make([]float32, n*k)
	for i := range weightData {
		weightData[i] = float32((i%251)-125) / 512
	}
	w := tensor.FromFloat32(tensor.Shape{n * k}, weightData)
	raw0, err := gguf.Quantize(w, gguf.Q4_K)
	if err != nil {
		b.Fatal(err)
	}
	for i := range weightData {
		weightData[i] += 0.03125
	}
	w = tensor.FromFloat32(tensor.Shape{n * k}, weightData)
	raw1, err := gguf.Quantize(w, gguf.Q4_K)
	if err != nil {
		b.Fatal(err)
	}
	xData := make([]float32, k)
	for i := range xData {
		xData[i] = float32((i%127)-63) / 256
	}
	x := tensor.FromFloat32(tensor.Shape{1, k}, xData)
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("CPU backend is not registered")
	}
	whole := be.(backend.SwiGLUInPlaceFuser)
	chunked := be.(backend.SwiGLUF32ChunkFuser)

	b.Run("composed", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			gate, up, err := gguf.QMatMulPair(x, raw0, raw1, gguf.Q4_K, n, k)
			if err != nil {
				b.Fatal(err)
			}
			if !whole.FuseSwiGLUInPlace(gate, up) {
				b.Fatal("CPU fuser rejected projections")
			}
			qmatmulPairApplySink = gate
		}
	})
	b.Run("producer_apply", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			qmatmulPairApplySink, err = gguf.QMatMulPairApply(x, raw0, raw1, gguf.Q4_K, n, k, chunked.FuseSwiGLUF32Chunk)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
