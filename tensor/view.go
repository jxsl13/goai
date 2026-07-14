package tensor

import "fmt"

// Reshape returns a zero-copy view with newShape. It requires this tensor to be
// contiguous with offset 0 (so element order is well-defined) and the element
// count to match; otherwise it returns an error. Call Contiguous first to
// reshape a non-contiguous view.
func (t *Tensor) Reshape(newShape Shape) (*Tensor, error) {
	if !newShape.IsValid() {
		return nil, fmt.Errorf("tensor: invalid reshape target %v", newShape)
	}
	if newShape.Numel() != t.Numel() {
		return nil, fmt.Errorf("tensor: reshape %v→%v changes element count %d→%d",
			t.shape, newShape, t.Numel(), newShape.Numel())
	}
	if !t.IsContiguous() || t.offset != 0 {
		return nil, fmt.Errorf("tensor: reshape requires contiguous offset-0 view (call Contiguous)")
	}
	sh, st := cloneShapeStrides(newShape)
	return &Tensor{
		storage: t.storage,
		shape:   sh,
		strides: st,
		offset:  t.offset,
		dev:     t.dev,
	}, nil
}

// Slice narrows axis dim to the half-open range [start, stop). It returns a
// zero-copy view sharing storage: the offset advances by start*stride and the
// axis length becomes stop-start. Errors on an out-of-range axis or bounds.
func (t *Tensor) Slice(dim, start, stop int) (*Tensor, error) {
	if dim < 0 || dim >= len(t.shape) {
		return nil, fmt.Errorf("tensor: slice axis %d out of range for %v", dim, t.shape)
	}
	if start < 0 || stop > t.shape[dim] || start > stop {
		return nil, fmt.Errorf("tensor: slice [%d:%d) out of range for axis %d size %d",
			start, stop, dim, t.shape[dim])
	}
	sh, st := copyShapeStrides(t.shape, t.strides)
	sh[dim] = stop - start
	return &Tensor{
		storage: t.storage,
		shape:   sh,
		strides: st,
		offset:  t.offset + start*t.strides[dim],
		dev:     t.dev,
	}, nil
}

// Transpose returns a zero-copy view with axes i and j swapped (shape and
// strides swapped at those positions). The result is generally non-contiguous.
func (t *Tensor) Transpose(i, j int) (*Tensor, error) {
	n := len(t.shape)
	if i < 0 || i >= n || j < 0 || j >= n {
		return nil, fmt.Errorf("tensor: transpose axes (%d,%d) out of range for %v", i, j, t.shape)
	}
	sh, st := copyShapeStrides(t.shape, t.strides)
	sh[i], sh[j] = sh[j], sh[i]
	st[i], st[j] = st[j], st[i]
	return &Tensor{storage: t.storage, shape: sh, strides: st, offset: t.offset, dev: t.dev}, nil
}

// Permute returns a zero-copy view whose axes are reordered by perm, where
// perm[k] is the source axis placed at position k. perm must be a permutation of
// 0..Ndim-1.
func (t *Tensor) Permute(perm ...int) (*Tensor, error) {
	n := len(t.shape)
	if len(perm) != n {
		return nil, fmt.Errorf("tensor: permute needs %d axes, got %d", n, len(perm))
	}
	// Permutation check without a heap-allocated seen-slice: a bitmask covers
	// every realistic rank (§base-perf); ranks beyond 64 fall back to a slice.
	if n <= 64 {
		var seen uint64
		for _, p := range perm {
			if p < 0 || p >= n || seen&(1<<uint(p)) != 0 {
				return nil, fmt.Errorf("tensor: permute %v is not a permutation of 0..%d", perm, n-1)
			}
			seen |= 1 << uint(p)
		}
	} else {
		seen := make([]bool, n)
		for _, p := range perm {
			if p < 0 || p >= n || seen[p] {
				return nil, fmt.Errorf("tensor: permute %v is not a permutation of 0..%d", perm, n-1)
			}
			seen[p] = true
		}
	}
	buf := make([]int, 2*n)
	sh := Shape(buf[:n:n])
	st := Strides(buf[n : 2*n : 2*n])
	for k, p := range perm {
		sh[k] = t.shape[p]
		st[k] = t.strides[p]
	}
	return &Tensor{storage: t.storage, shape: sh, strides: st, offset: t.offset, dev: t.dev}, nil
}
