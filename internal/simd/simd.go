//go:build !(amd64 && goexperiment.simd)

package simd

import "math"

// Portable SIMD-class primitives: tight, allocation-free elementwise loops the
// Go compiler can auto-vectorize. dst, a, b must have equal length. These are the
// fallback used on every arch EXCEPT amd64+GOEXPERIMENT=simd, where simd_avx.go
// provides archsimd (AVX/FMA) overrides of the same API (§T11b, ADR-0005).
//
// Each loop re-slices a and b to len(dst) up front: that single hint lets the
// compiler prove a[i]/b[i] are in range inside the loop and drop the two
// per-element bounds checks (measured ~1.45× on M2 Pro, see
// docs/perf-notes-lowlevel.md). Callers passing shorter a/b panic at the
// re-slice instead of mid-loop — same contract, earlier check.

func AddF64(dst, a, b []float64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
func SubF64(dst, a, b []float64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}
func MulF64(dst, a, b []float64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}
func DivF64(dst, a, b []float64) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func AddF32(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}
func SubF32(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}
func MulF32(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}
func DivF32(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

// ExpSumF64 sets dst[i] = exp(src[i]-bias) and returns Σ dst[i] — the scalar
// (non-SIMD build) form. The amd64+simd build overrides it with a 4-wide AVX2
// version (simd_avx.go). Used by the large-vocab softmax in nlp sampling.
func ExpSumF64(dst, src []float64, bias float64) float64 {
	var sum float64
	for i, v := range src {
		e := math.Exp(v - bias)
		dst[i] = e
		sum += e
	}
	return sum
}

// ExpScaledF64 writes dst[i] = exp(scale·src[i]). Portable scalar fallback; the
// amd64 SIMD build vectorizes the body. See the AVX variant for the parity note.
func ExpScaledF64(dst, src []float64, scale float64) {
	for i, v := range src {
		dst[i] = math.Exp(scale * v)
	}
}

// SigmoidF64 sets dst[i] = 1/(1+e^(−src[i])) (overflow-safe). Portable scalar
// fallback; the amd64 SIMD build vectorizes the body.
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

// SoftplusNegLLSumF64 returns Σ softplus((1−2·y[i])·f[i]) — binary cross-entropy sum
// for y∈{0,1}. Portable scalar fallback; the amd64 SIMD build vectorizes the body.
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

// WKVScanF64 runs the RWKV-4 WKV recurrence (see the AVX variant). Portable scalar
// fallback; the amd64 SIMD build vectorizes over channels.
func WKVScanF64(k, v, w, u, out []float64, seq, d int) {
	wkvScanScalar(k, v, w, u, out, seq, d, 0, d)
}
