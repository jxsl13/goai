//go:build !(goexperiment.simd && amd64)

package cpu

// Non-(amd64 SIMD) builds keep the scalar f64 normalize passes bit-for-bit: normF32Fast is false so
// layerNormFwd/rmsNormFwd never call the functions below (they exist only so the driver type-checks,
// same dead-code pattern as vexp_default.go). arm64's SIMD perf build is included here deliberately —
// vectorizing these passes is an amd64-only change; arm64 keeps its exact current path.
const normF32Fast = false

func layerNormNormalizeF32(x, gamma, beta, out []float32, mean, inv float32) {
	for j, g := range gamma {
		out[j] = (x[j]-mean)*inv*g + beta[j]
	}
}

func rmsNormNormalizeF32(x, gamma, out []float32, inv float32) {
	for j, g := range gamma {
		out[j] = x[j] * inv * g
	}
}

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
		s0 += float64(x[i])
		s1 += float64(x[i+1])
		s2 += float64(x[i+2])
		s3 += float64(x[i+3])
	}
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
