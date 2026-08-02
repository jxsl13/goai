//go:build race

package autograd

// raceBuild reports whether this binary was built with the race detector. See racebuild_off.go
// for why an exact-bits golden has to skip under it.
const raceBuild = true
