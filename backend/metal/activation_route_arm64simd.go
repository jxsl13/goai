//go:build darwin && cgo && arm64 && goexperiment.simd

package metal

// hostSIMDActivationRouteEnabled is compile-time true only for the measured
// Apple-Silicon SIMD build. The optimizer removes the route from other builds.
const hostSIMDActivationRouteEnabled = true
