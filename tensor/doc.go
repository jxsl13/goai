// Package tensor is layer L0: the core data model every other layer builds on.
//
// # For the AI professional
//
// It defines Dtype (f32, f64, f16, bf16), Shape, strides, Device, Allocator, and
// the Tensor type with zero-copy views: Reshape, Slice, Transpose, and Permute
// are pure stride/offset operations that share the backing Storage (§C5). Layout
// is row-major with explicit strides; there is no cgo in this package (§I3).
// AtF64/SetF64 are the dtype-agnostic accessors used by the reference kernels and
// tests — they widen through float64, so correctness is uniform across dtypes
// (§V9). Cast materializes a new contiguous tensor in another dtype (e.g. f32→f16
// for mixed-precision, §T41). Tensors carry no autograd state; differentiation is
// layer L2 (package autograd), and compute is layer L1 (package backend).
//
// # For the newcomer
//
// A "tensor" is just an n-dimensional array of numbers — a single value is 0-D, a
// list is 1-D, a table/matrix is 2-D, and so on. A [Shape] like {2, 3} means "2
// rows, 3 columns". This package lets you make tensors, read and write their
// elements, and re-view them cheaply: reshaping, slicing a row, or transposing a
// matrix does NOT copy the numbers — it just changes how the same underlying data
// is read (a "view"). That is what makes big-array math fast. You build tensors
// here; you do math on them with the backend package and take gradients with the
// autograd package. Start with [New], [FromFloat64], and the runnable examples
// below.
//
// Further reading: Harris et al. 2020, "Array Programming with NumPy" (Nature), for the strided-view model this tensor follows; Golub & Van Loan, "Matrix Computations" (4th ed.), for the numerical foundations.
//
// In plain terms: this package is the library's notion of an n-dimensional array — the box of numbers every model reads and writes — plus the bookkeeping (shapes, strides, views) that lets many operations share one block of memory instead of copying it.
package tensor
