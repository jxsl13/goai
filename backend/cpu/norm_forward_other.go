//go:build !(goexperiment.simd && (amd64 || arm64))

package cpu

// Default builds keep the scalar-f64 normalize pass. The functions remain so
// the generic driver type-checks even though the false constant makes them dead.
const normF32ForwardFast = false

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
