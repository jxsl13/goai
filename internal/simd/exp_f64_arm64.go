//go:build arm64 && goexperiment.simd

package simd

import "math"

const expF64NeonLo = -708.0

//go:noescape
func expNegPairsNeonF64(dst, src *float64, pairs int)

func expNegF64Safe(x float64) bool {
	return !math.IsNaN(x) && x <= 0 && (x >= expF64NeonLo || math.IsInf(x, -1))
}

func expNegSliceF64(x []float64) {
	n := len(x) &^ 1
	if n != 0 {
		expNegPairsNeonF64(&x[0], &x[0], n>>1)
	}
	if n != len(x) {
		x[n] = math.Exp(x[n])
	}
}

func expSumF64Scalar(dst, src []float64, bias float64) float64 {
	var sum float64
	for i, v := range src {
		e := math.Exp(v - bias)
		dst[i] = e
		sum += e
	}
	return sum
}

// ExpSumF64 uses the shared two-lane exp body for the softmax domain x-bias≤0.
func ExpSumF64(dst, src []float64, bias float64) float64 {
	dst = dst[:len(src)]
	for _, v := range src {
		if !expNegF64Safe(v - bias) {
			return expSumF64Scalar(dst, src, bias)
		}
	}
	for i, v := range src {
		dst[i] = v - bias
	}
	expNegSliceF64(dst)
	var sum float64
	for _, v := range dst {
		sum += v
	}
	return sum
}

func expScaledF64Scalar(dst, src []float64, scale float64) {
	for i, v := range src {
		dst[i] = math.Exp(scale * v)
	}
}

// ExpScaledF64 vectorizes the non-positive SSM/Mamba decay domain.
func ExpScaledF64(dst, src []float64, scale float64) {
	dst = dst[:len(src)]
	for _, v := range src {
		if !expNegF64Safe(scale * v) {
			expScaledF64Scalar(dst, src, scale)
			return
		}
	}
	for i, v := range src {
		dst[i] = scale * v
	}
	expNegSliceF64(dst)
}

func sigmoidF64Scalar(dst, src []float64) {
	for i, x := range src {
		if x >= 0 {
			dst[i] = 1 / (1 + math.Exp(-x))
		} else {
			z := math.Exp(x)
			dst[i] = z / (1 + z)
		}
	}
}

// SigmoidF64 remains scalar until the shared leaf's composed route is gated.
func SigmoidF64(dst, src []float64) {
	dst = dst[:len(src)]
	sigmoidF64Scalar(dst, src)
}

func softplusNegLLSumF64Scalar(f, y []float64) float64 {
	var sum float64
	for i := range f {
		x := (1 - 2*y[i]) * f[i]
		if x > 0 {
			sum += x + math.Log1p(math.Exp(-x))
		} else {
			sum += math.Log1p(math.Exp(x))
		}
	}
	return sum
}

// SoftplusNegLLSumF64 remains scalar until the shared leaf is composed and gated.
func SoftplusNegLLSumF64(f, y []float64) float64 {
	y = y[:len(f)]
	return softplusNegLLSumF64Scalar(f, y)
}
