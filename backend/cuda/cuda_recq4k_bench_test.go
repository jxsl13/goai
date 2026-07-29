//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// benchRecQ4K times the RECORDER Q4_K projection (QMatMulResidentQ4K — the path llamagpu's decoder
// actually calls) at batch M. Before the routing fix this ran the per-row GEMV for all M; after, M>=8
// routes to the weight-read-once MT GEMM.
func benchRecQ4K(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(5))
	w := tensor.New(tensor.F32, tensor.Shape{k, n})
	wf := w.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64())
	}
	blocks, err := gguf.Quantize(transpose2D(w), gguf.Q4_K)
	if err != nil {
		b.Fatal(err)
	}
	rq, err := cuda.NewResidentBQ4KFromBlocks(blocks, k, n)
	if err != nil {
		b.Fatal(err)
	}
	defer rq.Free()
	rec, err := cuda.NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	defer rec.Free()
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	da, _ := cuda.NewDeviceF32(m, k)
	defer da.Free()
	da.UploadF32(a)
	out, _ := cuda.NewDeviceF32(m, n)
	defer out.Free()
	if err := rec.QMatMulResidentQ4K(da, rq, out, m); err != nil {
		b.Fatal(err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		rec.QMatMulResidentQ4K(da, rq, out, m)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(k)*float64(n)*0.5625/(b.Elapsed().Seconds()/float64(b.N))/1e9, "wGB/s")
	b.ReportMetric(2*float64(m)*float64(k)*float64(n)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

func BenchmarkRecQ4K128_2048x2048(b *testing.B) { benchRecQ4K(b, 128, 2048, 2048) }
func BenchmarkRecQ4K128_2048x5632(b *testing.B) { benchRecQ4K(b, 128, 2048, 5632) }
func BenchmarkRecQ4K64_2048x2048(b *testing.B)  { benchRecQ4K(b, 64, 2048, 2048) }
