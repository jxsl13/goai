//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// Full quantized-prefill pipeline: f32 activations -> DEVICE per-row int8 quant -> MMQ vs per-block
// int8 weights -> f32. Validates the device quantizer + the e2e accuracy vs a true f32 GEMM.
func TestZZMMQPrefillPipeline(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 128, 512, 128
	rng := rand.New(rand.NewSource(37))
	af := make([]float32, M*K)
	wf := make([]float32, N*K)
	for i := range af {
		af[i] = float32(rng.NormFloat64())
	}
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.5
	}
	w8, wSc := quantRowsQ8Block(wf, N, K) // weights per-block (resident)

	got := mmqPrefill(af, w8, wSc, M, K, N) // activations quantized ON DEVICE
	if got == nil {
		t.Fatal("mmqPrefill failed")
	}

	// Cross-check the device quant matches a host per-row quant + reference the true f32 GEMM.
	a8h, aSch := quantRowsQ8PerRow(af, M, K)
	nb := K / 32
	var sqErr, sqRef, maxHostDev, maxMag float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var refBlock float64
			for b := 0; b < nb; b++ {
				var idot int32
				for j := 0; j < 32; j++ {
					idot += int32(a8h[m*K+b*32+j]) * int32(w8[n*K+b*32+j])
				}
				refBlock += float64(idot) * float64(wSc[n*nb+b])
			}
			refBlock *= float64(aSch[m])
			var refTrue float64
			for k := 0; k < K; k++ {
				refTrue += float64(af[m*K+k]) * float64(wf[n*K+k])
			}
			g := float64(got[m*N+n])
			if e := math.Abs(g - refBlock); e > maxHostDev {
				maxHostDev = e
			}
			if a := math.Abs(refBlock); a > maxMag {
				maxMag = a
			}
			d := g - refTrue
			sqErr += d * d
			sqRef += refTrue * refTrue
		}
	}
	relHostDev := maxHostDev / maxMag
	quantRMS := math.Sqrt(sqErr / sqRef)
	t.Logf("device-quant vs host-quant rel %.3g; full-pipeline vs true-f32 GEMM norm-rel-RMS %.4g", relHostDev, quantRMS)
	if relHostDev > 1e-5 {
		t.Fatalf("device quantizer disagrees with host per-row quant: rel %.3g", relHostDev)
	}
	if quantRMS > 3e-2 {
		t.Fatalf("quantized-prefill accuracy too poor: RMS %.3g", quantRMS)
	}
	t.Log("QUANTIZED-PREFILL PIPELINE CORRECT (device quant matches host; e2e accuracy within tol)")
}
