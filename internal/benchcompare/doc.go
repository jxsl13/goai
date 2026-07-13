// Package benchcompare holds cross-backend comparison micro-benchmarks (cpu vs
// metal vs vulkan vs the pure-Go reference) for the accelerated ops. The
// benchmarks live in compare_test.go behind `//go:build darwin && cgo && vulkan`
// so a single test binary can blank-import every host-supported backend at once;
// run them with `make bench-compare`. This file exists (untagged) only so the
// package is non-empty on the default build path.
package benchcompare
