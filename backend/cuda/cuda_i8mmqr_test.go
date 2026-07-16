//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// per-ROW symmetric int8 quant: one scale per row over all K.
func quantRowsQ8PerRow(x []float32, rows, k int) ([]int8, []float32) {
	q := make([]int8, rows*k)
	sc := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var amax float32
		for j := 0; j < k; j++ {
			v := float32(math.Abs(float64(x[r*k+j])))
			if v > amax {
				amax = v
			}
		}
		s := amax / 127.0
		sc[r] = s
		inv := float32(0)
		if s > 0 {
			inv = 1.0 / s
		}
		for j := 0; j < k; j++ {
			qv := int32(math.Round(float64(x[r*k+j] * inv)))
			if qv > 127 {
				qv = 127
			}
			if qv < -128 {
				qv = -128
			}
			q[r*k+j] = int8(qv)
		}
	}
	return q, sc
}

func TestZZI8MMQRCorrectAndAccuracy(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 128, 256, 128
	rng := rand.New(rand.NewSource(31))
	af := make([]float32, M*K)
	wf := make([]float32, N*K)
	for i := range af {
		af[i] = float32(rng.NormFloat64())
	}
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.5
	}
	a8, aSc := quantRowsQ8PerRow(af, M, K) // per-row activation
	w8, wSc := quantRowsQ8Block(wf, N, K)  // per-block weight (from mmq test)

	got := i8MMQR(a8, w8, aSc, wSc, M, K, N)
	if got == nil {
		t.Fatal("i8MMQR failed")
	}
	nb := K / 32
	var maxAbs, maxMag, sqErr, sqRef float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var refBlock float64
			for b := 0; b < nb; b++ {
				var idot int32
				for j := 0; j < 32; j++ {
					idot += int32(a8[m*K+b*32+j]) * int32(w8[n*K+b*32+j])
				}
				refBlock += float64(idot) * float64(wSc[n*nb+b])
			}
			refBlock *= float64(aSc[m]) // per-row activation scale applied once
			var refTrue float64
			for k := 0; k < K; k++ {
				refTrue += float64(af[m*K+k]) * float64(wf[n*K+k])
			}
			g := float64(got[m*N+n])
			if e := math.Abs(g - refBlock); e > maxAbs {
				maxAbs = e
			}
			if a := math.Abs(refBlock); a > maxMag {
				maxMag = a
			}
			d := g - refTrue
			sqErr += d * d
			sqRef += refTrue * refTrue
		}
	}
	rel := maxAbs / maxMag
	quantRMS := math.Sqrt(sqErr / sqRef)
	t.Logf("MMQ-perrow kernel-vs-ref maxAbs %.4g (rel %.3g); quant-vs-f32 RMS %.4g", maxAbs, rel, quantRMS)
	if rel > 1e-5 {
		t.Fatalf("MMQ_r kernel arithmetic WRONG: rel %.3g", rel)
	}
	if quantRMS > 3e-2 {
		t.Fatalf("per-row-act int8 accuracy too poor: RMS %.3g", quantRMS)
	}
	t.Log("MMQ_r CORRECT + per-row-act/per-block-wt int8 accuracy OK")
}

func BenchmarkI8MMQR_prefill(b *testing.B) {
	if !Available() {
		b.Skip("no gpu")
	}
	const M, K, N = 128, 2048, 2048
	nb := K / 32
	a8 := make([]int8, M*K)
	w8 := make([]int8, N*K)
	aSc := make([]float32, M)
	wSc := make([]float32, N*nb)
	for i := range aSc {
		aSc[i] = 0.01
	}
	for i := range wSc {
		wSc[i] = 0.01
	}
	dA := i8uploadRaw(a8)
	dW := i8uploadRaw(w8)
	dAs := f32uploadRaw(aSc)
	dWs := f32uploadRaw(wSc)
	dC := f32allocRaw(M * N)
	i8mmqrRaw(dA, dW, dAs, dWs, dC, M, K, N)
	i8syncRaw()
	b.ResetTimer()
	for range b.N {
		i8mmqrRaw(dA, dW, dAs, dWs, dC, M, K, N)
	}
	i8syncRaw()
	b.StopTimer()
	b.ReportMetric(2.0*float64(M)*float64(K)*float64(N)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GOP/s")
}
