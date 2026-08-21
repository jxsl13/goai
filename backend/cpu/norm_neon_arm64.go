//go:build goexperiment.simd && arm64

package cpu

// The row statistics remain f64-accumulated. Only the bandwidth-bound output
// pass narrows once per row and runs four F32 lanes per NEON instruction.
const normF32ForwardFast = true

//go:noescape
func rmsNormNormalizeF32BlocksNeon(out, x, gamma *float32, blocks int, inv float32)

//go:noescape
func layerNormNormalizeF32BlocksNeon(out, x, gamma, beta *float32, blocks int, mean, inv float32)

func rmsNormNormalizeF32(x, gamma, out []float32, inv float32) {
	n := len(gamma)
	nv := n &^ 15
	if nv != 0 {
		rmsNormNormalizeF32BlocksNeon(&out[0], &x[0], &gamma[0], nv>>4, inv)
	}
	for i := nv; i < n; i++ {
		out[i] = x[i] * inv * gamma[i]
	}
}

func layerNormNormalizeF32(x, gamma, beta, out []float32, mean, inv float32) {
	n := len(gamma)
	nv := n &^ 15
	if nv != 0 {
		layerNormNormalizeF32BlocksNeon(&out[0], &x[0], &gamma[0], &beta[0], nv>>4, mean, inv)
	}
	for i := nv; i < n; i++ {
		out[i] = (x[i]-mean)*inv*gamma[i] + beta[i]
	}
}
