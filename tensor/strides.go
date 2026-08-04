package tensor

// Strides holds the per-dimension step, in elements (not bytes), to move one
// index along each axis of a Tensor. A Tensor's flat offset for a multi-index
// idx is base + sum(idx[i]*strides[i]). Views (reshape/slice/transpose) are
// expressed purely as changes to Strides and base, never copies (§C5, §T3).
type Strides []int

// RowMajorStrides returns the contiguous row-major strides for shape: the last
// axis is the fastest-varying (stride 1). strides[i] = product(shape[i+1:]).
// Scalar shape → empty strides. Computed structurally even when a dimension is
// 0, so the result stays well-defined for empty tensors.
func RowMajorStrides(shape Shape) Strides {
	n := len(shape)
	if n == 0 {
		return Strides{}
	}
	st := make(Strides, n)
	acc := 1
	for i := n - 1; i >= 0; i-- {
		st[i] = acc
		acc *= shape[i]
	}
	return st
}

// Offset returns the flat element offset for the multi-index idx into a tensor
// with the given base offset and strides. It panics if len(idx) != len(strides),
// mirroring Go slice-index discipline — a rank mismatch is a programming error,
// not a runtime condition to tolerate.
func Offset(base int, strides Strides, idx []int) int {
	if len(idx) != len(strides) {
		panic("tensor: index rank does not match strides rank")
	}
	off := base
	//perfscan:ignore PS3010 Offset loop bounded by ndim (2-4), inherent index math
	for i, ix := range idx {
		off += ix * strides[i]
	}
	return off
}

// IsContiguous reports whether strides are exactly the row-major strides for
// shape, i.e. the tensor's elements are laid out densely with no gaps.
func (st Strides) IsContiguous(shape Shape) bool {
	if len(st) != len(shape) {
		return false
	}
	acc := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if st[i] != acc {
			return false
		}
		acc *= shape[i]
	}
	return true
}

// cloneShapeStrides returns an independent copy of shape together with its
// row-major strides, carved out of ONE backing allocation (§base-perf: halves
// the slice allocs on every tensor creation vs Clone+RowMajorStrides). Both
// slices are full-capacity (three-index) so an append on one can never bleed
// into the other.
func cloneShapeStrides(shape Shape) (Shape, Strides) {
	n := len(shape)
	if n == 0 {
		// Bit-exact with Clone()+RowMajorStrides(): nil stays nil, empty stays empty.
		if shape == nil {
			return nil, Strides{}
		}
		return Shape{}, Strides{}
	}
	buf := make([]int, 2*n)
	sh := Shape(buf[:n:n])
	st := Strides(buf[n : 2*n : 2*n])
	acc := 1
	//perfscan:ignore PS4004 rank-sized loop; stride arith iterates anyway, copy-split no win
	for i := n - 1; i >= 0; i-- {
		sh[i] = shape[i]
		st[i] = acc
		acc *= shape[i]
	}
	return sh, st
}

// copyShapeStrides copies an existing shape+strides pair into ONE backing
// allocation (view-creation hot path: Slice/Transpose/Permute made two slice
// allocs each, §base-perf). Full-capacity slicing prevents append bleed, same
// as cloneShapeStrides.
func copyShapeStrides(shape Shape, strides Strides) (Shape, Strides) {
	n := len(shape)
	if n == 0 {
		if shape == nil {
			return nil, Strides{}
		}
		return Shape{}, Strides{}
	}
	buf := make([]int, 2*n)
	sh := Shape(buf[:n:n])
	st := Strides(buf[n : 2*n : 2*n])
	copy(sh, shape)
	copy(st, strides)
	return sh, st
}

// Unravel converts a flat position in [0, shape.Numel()) into its row-major
// multi-index for shape. It is the inverse of Offset with row-major strides and
// base 0. Scalar shape → empty index. The returned slice is freshly allocated.
func Unravel(pos int, shape Shape) []int {
	n := len(shape)
	idx := make([]int, n)
	for i := n - 1; i >= 0; i-- {
		d := shape[i]
		if d == 0 {
			// Empty axis: no valid positions; leave zeros.
			continue
		}
		idx[i] = pos % d
		pos /= d
	}
	return idx
}
