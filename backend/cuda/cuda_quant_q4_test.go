//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Asymmetric-Q4 GEMV must approximate a host f32 matmul within Q4 tolerance (much looser
// than Q8, but far tighter than symmetric Q4_0 thanks to the per-block min) — the
// weight-bandwidth lever for decode. End-to-end token quality is validated separately.
func TestCUDAQ4GemvAccuracy(t *testing.T) {
	skipNoGPU(t)
	const m, k, n = 1, 2048, 2048 // K multiple of 256
	dx, x := dev(t, m, k, 11)
	defer dx.Free()
	wten := bench.RandF32(tensor.Shape{k, n}, 12)
	wf := wten.Contiguous().Storage().F32()
	w4, err := cuda.NewResidentBQ4(wten)
	must(t, err)
	defer w4.Free()
	do, err := cuda.NewDeviceF32(m, n)
	must(t, err)
	defer do.Free()

	must(t, w4.QMatMulInto(dx, do))
	got, err := do.ToHost()
	must(t, err)
	gf := got.Contiguous().Storage().F32()

	var maxRel, sumRel float64
	cnt := 0
	for j := 0; j < n; j++ {
		var s float64
		for p := 0; p < k; p++ {
			s += float64(x[p]) * float64(wf[p*n+j])
		}
		if math.Abs(s) > 1 {
			rel := math.Abs(float64(gf[j])-s) / math.Abs(s)
			if rel > maxRel {
				maxRel = rel
			}
			sumRel += rel
			cnt++
		}
	}
	meanRel := sumRel / float64(cnt)
	t.Logf("Q4 GEMV vs host f32: mean rel err %.2f%%, max %.2f%% (over |out|>1)", meanRel*100, maxRel*100)
	// Sanity: asymmetric Q4 should land in the single-digit-percent mean range — well under
	// 20%, confirming the dequant (min + nibble·scale) and packing are correct.
	if meanRel > 0.20 {
		t.Fatalf("Q4 mean rel err %.1f%% too high — dequant/packing likely wrong", meanRel*100)
	}
}

func benchQMatmul(b *testing.B, k, n int, q4 bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	a := bench.RandF32(tensor.Shape{1, k}, 1)
	w := bench.RandF32(tensor.Shape{k, n}, 2)
	da, _ := cuda.UploadF32(a)
	defer da.Free()
	out, _ := cuda.NewDeviceF32(1, n)
	defer out.Free()
	if q4 {
		r, _ := cuda.NewResidentBQ4(w)
		defer r.Free()
		r.QMatMulInto(da, out)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			r.QMatMulInto(da, out)
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(float64(k)*float64(n)*0.75/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
	} else {
		r, _ := cuda.NewResidentBQ8(w)
		defer r.Free()
		r.QMatMulInto(da, out)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			r.QMatMulInto(da, out)
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(float64(k)*float64(n)*1.125/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
	}
}

func BenchmarkGemvQ8_2048(b *testing.B) { benchQMatmul(b, 2048, 2048, false) }
func BenchmarkGemvQ4_2048(b *testing.B) { benchQMatmul(b, 2048, 2048, true) }
