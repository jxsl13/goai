//go:build darwin && cgo

package metal

// SetConvUseMPS toggles the conv2d forward path between Apple's native MPSGraph
// convolution (true, the live default) and the im2col+MPS-GEMM fallback (false),
// returning the previous value. Test-only (§T620 A/B) — not part of the public API.
func SetConvUseMPS(v bool) bool {
	old := convUseMPS
	convUseMPS = v
	return old
}
