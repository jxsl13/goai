//go:build !(goexperiment.simd && (arm64 || amd64))

package cpu

import "math"

// Every build that is NOT a SIMD perf build (arm64 → vexp_arm64.go, amd64 →
// vexp_amd64.go) keeps the f64 math.Exp softmax bit-for-bit: vexpNeon gates the
// whole vexp activation path off, vexpF32Fast gates the softmax-family exp+sum
// off, and this vexpF32 exists only so the driver type-checks (dead code at run
// time here — same pattern as gemm_rows_default.go).
const (
	vexpNeon    = false
	vexpF32Fast = false
	vexpF64Fast = false // no F64 vector SiLU off the amd64 SIMD build; scalar path stays exact
)

// vexpF32 computes p[i] = exp(p[i]-m) in place and returns Σ p[i].
// len(p) must be a multiple of 4 (the NEON kernel's quad contract).
func vexpF32(p []float32, m float32) float32 {
	var sum float32
	for i, v := range p {
		e := expF32(v - m)
		p[i] = e
		sum += e
	}
	return sum
}

// vgeluF32 computes dst[i] = gelu(src[i]) via the scalar poly — exists only so
// the OpGELU F32 fast path type-checks (dead code at run time here: vexpNeon
// is false, so geluKernelCPU keeps the exact f64 math.Erf path).
func vgeluF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = geluF32(v)
	}
}

// vgeluGradF32 computes dst[i] = g[i]·gelu'(x[i]) via the scalar poly — exists
// only so the OpGELUBackward F32 fast path type-checks (dead code at run time
// here: vexpNeon is false, so the kernel is never registered and gelu-backward
// keeps the exact ref f64 path).
func vgeluGradF32(dst, x, g []float32) {
	for i, v := range x {
		dst[i] = geluGradF32(v, g[i])
	}
}

// vsiluF32 / vsiluGradF32 / vsigmoidF32 compute the SiLU / SiLU-backward /
// sigmoid scalar polys — they exist only so the arm64-perf fast paths
// type-check (dead code at run time here: vexpNeon is false, so the kernels
// keep the exact scalar-f64 / ref paths bit-for-bit).
func vsiluF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = siluF32(v)
	}
}

func vsiluGradF32(dst, x, g []float32) {
	for i, v := range x {
		dst[i] = siluGradF32(v, g[i])
	}
}

func vsigmoidF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = sigmoidF32(v)
	}
}

// vexpFullF32 / vtanhF32 / vlogF32 compute the standalone-unary scalar polys
// (§T666) — they exist only so the arm64-perf fast paths type-check (dead code
// at run time here: vexpNeon is false, so expKernelCPU / tanhKernelCPU /
// logKernelCPU keep the exact scalar-f64 math.Exp/Tanh/Log paths bit-for-bit).
func vexpFullF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = expFullF32(v)
	}
}

func vtanhF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = tanhF32(v)
	}
}

func vlogF32(dst, src []float32) {
	for i, v := range src {
		dst[i] = logF32(v)
	}
}

// vsiluF64 exists only so siluKernelCPU type-checks on non-amd64-SIMD builds;
// vexpF64Fast is false here, so it is dead at run time (the scalar exact path runs).
func vsiluF64(dst, src []float64) {
	for i, v := range src {
		dst[i] = v / (1 + math.Exp(-v))
	}
}

// vsigmoidF64 exists only so sigmoidKernelCPU type-checks off the amd64 SIMD build; vexpF64Fast is
// false here, so it is dead at run time (the scalar exact path runs).
func vsigmoidF64(dst, src []float64) {
	for i, v := range src {
		if v >= 0 {
			dst[i] = 1 / (1 + math.Exp(-v))
		} else {
			z := math.Exp(v)
			dst[i] = z / (1 + z)
		}
	}
}

// vsoftplusF64 exists only so softplusKernelCPU type-checks off the amd64 SIMD
// build; vexpF64Fast is false here, so it is dead at run time.
func vsoftplusF64(dst, src []float64) {
	for i, v := range src {
		if v > 0 {
			dst[i] = v + math.Log1p(math.Exp(-v))
		} else {
			dst[i] = math.Log1p(math.Exp(v))
		}
	}
}

// vsoftcapF64 exists only so softCapKernelCPU type-checks off the amd64 SIMD
// build; vexpF64Fast is false here, so it is dead at run time.
func vsoftcapF64(dst, src []float64, cap float64) {
	for i, v := range src {
		dst[i] = cap * math.Tanh(v/cap)
	}
}

// vsoftplusGradF64 computes dst[i] = g[i]·softplus'(x[i]) = g[i]·σ(x[i]) scalar (the amd64
// twin vectorizes this 4-wide). Overflow-safe σ = num/(1+z) with z=e^(−|x|), num=(x≥0?1:z).
func vsoftplusGradF64(dst, x, g []float64) {
	for i, v := range x {
		z := math.Exp(-math.Abs(v))
		num := z
		if v >= 0 {
			num = 1
		}
		dst[i] = g[i] * (num / (1 + z))
	}
}
