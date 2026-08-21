//go:build !(amd64 && goexperiment.simd) && !(arm64 && goexperiment.simd)

package simd

import "math"

// ExpSumF64 sets dst[i] = exp(src[i]-bias) and returns Σ dst[i] — the scalar
// (non-SIMD build) form. Architecture SIMD builds override this definition.
func ExpSumF64(dst, src []float64, bias float64) float64 {
	var sum float64
	for i, v := range src {
		e := math.Exp(v - bias)
		dst[i] = e
		sum += e
	}
	return sum
}

// ExpScaledF64 writes dst[i] = exp(scale·src[i]). Portable scalar fallback.
func ExpScaledF64(dst, src []float64, scale float64) {
	for i, v := range src {
		dst[i] = math.Exp(scale * v)
	}
}

// SigmoidF64 sets dst[i] = 1/(1+e^(−src[i])) (overflow-safe).
func SigmoidF64(dst, src []float64) {
	for i, x := range src {
		if x >= 0 {
			dst[i] = 1 / (1 + math.Exp(-x))
		} else {
			z := math.Exp(x)
			dst[i] = z / (1 + z)
		}
	}
}

// SoftplusNegLLSumF64 returns Σ softplus((1−2·y[i])·f[i]).
func SoftplusNegLLSumF64(f, y []float64) float64 {
	var s float64
	for i := range f {
		x := (1 - 2*y[i]) * f[i]
		if x > 0 {
			s += x + math.Log1p(math.Exp(-x))
		} else {
			s += math.Log1p(math.Exp(x))
		}
	}
	return s
}
