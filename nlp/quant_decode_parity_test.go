package nlp_test

import "math"

// quantQ8DecodeParity preserves the historical bit-identity gate on portable
// builds. Architectures that register the tolerance-gated Q8_0 SIMD selector
// use the same bound as gguf's fused-vs-general QMatMul gate.
func quantQ8DecodeParity(got, want float64) bool {
	if !q8DecodeUsesSIMDTolerance {
		return got == want
	}
	return math.Abs(got-want) <= 5e-5*math.Abs(want)+1e-6
}
