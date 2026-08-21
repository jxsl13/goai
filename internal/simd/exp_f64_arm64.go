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

// SigmoidF64 composes the shared exp body with the stable scalar recombination.
func SigmoidF64(dst, src []float64) {
	dst = dst[:len(src)]
	if len(src) != 0 && &dst[0] == &src[0] {
		sigmoidF64Scalar(dst, src)
		return
	}
	for _, x := range src {
		if !expNegF64Safe(-math.Abs(x)) {
			sigmoidF64Scalar(dst, src)
			return
		}
	}
	for i, x := range src {
		dst[i] = -math.Abs(x)
	}
	expNegSliceF64(dst)
	for i, x := range src {
		z := dst[i]
		if x >= 0 {
			dst[i] = 1 / (1 + z)
		} else {
			dst[i] = z / (1 + z)
		}
	}
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

// SoftplusNegLLSumF64 amortizes the shared exp leaf over stack-resident blocks;
// log1p and the reduction retain scalar order and therefore need no heap scratch.
func SoftplusNegLLSumF64(f, y []float64) float64 {
	y = y[:len(f)]
	for i := range f {
		x := (1 - 2*y[i]) * f[i]
		if !expNegF64Safe(-math.Abs(x)) {
			return softplusNegLLSumF64Scalar(f, y)
		}
	}
	const blockSize = 256
	var z [blockSize]float64
	var sum float64
	for base := 0; base < len(f); base += blockSize {
		end := min(base+blockSize, len(f))
		block := z[:end-base]
		for i := range block {
			x := (1 - 2*y[base+i]) * f[base+i]
			block[i] = -math.Abs(x)
		}
		expNegSliceF64(block)
		for i, e := range block {
			x := (1 - 2*y[base+i]) * f[base+i]
			if x > 0 {
				sum += x + math.Log1p(e)
			} else {
				sum += math.Log1p(e)
			}
		}
	}
	return sum
}
