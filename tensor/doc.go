// Package tensor is layer L0: the core data model.
//
// It defines Dtype, Shape, strides, Device, Allocator, and the Tensor type
// with zero-copy views (reshape/slice/transpose as stride operations). No cgo
// in this package (§I3). Row-major layout with explicit strides (§C5).
package tensor
