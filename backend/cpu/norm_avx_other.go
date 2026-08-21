//go:build !(goexperiment.simd && amd64)

package cpu

// Non-(amd64 SIMD) builds keep the existing scalar-f64 backward write pass.
// The arm64 SIMD build is included deliberately: its NEON specialization is
// forward-only, so backward remains a performance and numerical negative control.
const normF32BackwardFast = false

func rmsNormDxF32(u, g, x, dx []float32, inv, c float32) {
	for j := range dx {
		dx[j] = inv * (u[j]*g[j] - x[j]*c)
	}
}

func layerNormDxF32(u, g, x, dx []float32, mu, inv, meanA, meanAX float32) {
	for j := range dx {
		xhat := (x[j] - mu) * inv
		dx[j] = inv * (u[j]*g[j] - meanA - xhat*meanAX)
	}
}

// sumF32 returns Σx accumulated in f64, 4-way unrolled.
func sumF32(x []float32) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+3 < len(x); i += 4 {
		//perfscan:ignore PS3019 non-amd64-SIMD fallback (dead on amd64 build); already 4-way unrolled
		s0 += float64(x[i])
		s1 += float64(x[i+1])
		s2 += float64(x[i+2])
		s3 += float64(x[i+3])
	}
	//perfscan:ignore PS3010 scalar remainder tail of already-4-way sum; amd64 uses vectorized norm
	for ; i < len(x); i++ {
		s0 += float64(x[i])
	}
	return (s0 + s1) + (s2 + s3)
}

// sumSqF32 returns Σx² accumulated in f64, 4-way unrolled.
func sumSqF32(x []float32) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+3 < len(x); i += 4 {
		//perfscan:ignore PS3019 non-amd64-SIMD fallback; already 4-way unrolled f64
		v0, v1, v2, v3 := float64(x[i]), float64(x[i+1]), float64(x[i+2]), float64(x[i+3])
		s0 += v0 * v0
		s1 += v1 * v1
		s2 += v2 * v2
		s3 += v3 * v3
	}
	for ; i < len(x); i++ {
		v := float64(x[i])
		s0 += v * v
	}
	return (s0 + s1) + (s2 + s3)
}

// varSumF32 returns Σ(x−mu)² accumulated in f64, 4-way unrolled.
func varSumF32(x []float32, mu float64) float64 {
	var s0, s1, s2, s3 float64
	i := 0
	for ; i+3 < len(x); i += 4 {
		//perfscan:ignore PS3019 non-amd64-SIMD fallback; already 4-way unrolled f64
		d0 := float64(x[i]) - mu
		d1 := float64(x[i+1]) - mu
		d2 := float64(x[i+2]) - mu
		d3 := float64(x[i+3]) - mu
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(x); i++ {
		d := float64(x[i]) - mu
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}
