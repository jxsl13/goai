package llamagpu

// SetGPTDecoderResidualFusionForTest selects the residual epilogue path in same-binary
// benchmarks. It is compiled only into tests and is not part of the library API.
func SetGPTDecoderResidualFusionForTest(d *GPTDecoder, enabled bool) {
	d.fuseResidual = enabled
}
