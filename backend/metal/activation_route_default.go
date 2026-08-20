//go:build darwin && cgo && !(arm64 && goexperiment.simd)

package metal

// Default and unmeasured architecture builds retain the incumbent Metal route.
const hostSIMDActivationRouteEnabled = false
