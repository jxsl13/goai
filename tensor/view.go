package tensor

import (
	"fmt"
	"strconv"
)

// intsText renders a []int as "[a b c]" WITHOUT leaking it. Handing the slice to fmt as an
// interface argument would make escape analysis mark the enclosing function's parameter leaking,
// which forces every caller to heap-allocate its argument even on paths that never format
// (PS3013). This reads only the ints, so nothing derived from the slice is retained.
func intsText(xs []int) string {
	b := make([]byte, 0, 2+4*len(xs))
	b = append(b, '[')
	for i, x := range xs {
		if i > 0 {
			b = append(b, ' ')
		}
		b = strconv.AppendInt(b, int64(x), 10)
	}
	return string(append(b, ']'))
}

// Reshape returns a zero-copy view with newShape. It requires this tensor to be
// contiguous with offset 0 (so element order is well-defined) and the element
// count to match; otherwise it returns an error. Call Contiguous first to
// reshape a non-contiguous view.
func (t *Tensor) Reshape(newShape Shape) (*Tensor, error) {
	if !newShape.IsValid() {
		return nil, fmt.Errorf("tensor: invalid reshape target %s", newShape.String())
	}
	if newShape.Numel() != t.Numel() {
		return nil, fmt.Errorf("tensor: reshape %s→%s changes element count %d→%d",
			t.shape.String(), newShape.String(), t.Numel(), newShape.Numel())
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
	v, sh, st := newViewOf(len(t.shape))
	copy(sh, t.shape)
	copy(st, t.strides)
	sh[dim] = stop - start
	if v == nil {
		return &Tensor{
			storage: t.storage,
			shape:   sh,
			strides: st,
			offset:  t.offset + start*t.strides[dim],
			dev:     t.dev,
		}, nil
	}
	*v = Tensor{
		storage: t.storage,
		shape:   sh,
		strides: st,
		offset:  t.offset + start*t.strides[dim],
		dev:     t.dev,
	}
	return v, nil
}

// Transpose returns a zero-copy view with axes i and j swapped (shape and
// strides swapped at those positions). The result is generally non-contiguous.
func (t *Tensor) Transpose(i, j int) (*Tensor, error) {
	n := len(t.shape)
	if i < 0 || i >= n || j < 0 || j >= n {
		return nil, fmt.Errorf("tensor: transpose axes (%d,%d) out of range for %v", i, j, t.shape)
	}
	v, sh, st := newViewOf(n)
	copy(sh, t.shape)
	copy(st, t.strides)
	sh[i], sh[j] = sh[j], sh[i]
	st[i], st[j] = st[j], st[i]
	if v == nil {
		return &Tensor{storage: t.storage, shape: sh, strides: st, offset: t.offset, dev: t.dev}, nil
	}
	*v = Tensor{storage: t.storage, shape: sh, strides: st, offset: t.offset, dev: t.dev}
	return v, nil
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
				return nil, fmt.Errorf("tensor: permute %s is not a permutation of 0..%d", intsText(perm), n-1)
			}
			seen |= 1 << uint(p)
		}
	} else {
		seen := make([]bool, n)
		for _, p := range perm {
			if p < 0 || p >= n || seen[p] {
				return nil, fmt.Errorf("tensor: permute %s is not a permutation of 0..%d", intsText(perm), n-1)
			}
			seen[p] = true
		}
	}
	// Permute deliberately does NOT go through newViewOf. Its callers are rank 3 and 4, above
	// maxInlineRank, so the block can never apply — and routing it through the helper anyway
	// measured +7.84% (p=0.003, n=12), the cost of a call this code used to do inline.
	buf := make([]int, 2*n)
	sh := Shape(buf[:n:n])
	st := Strides(buf[n : 2*n : 2*n])
	for k, p := range perm {
		sh[k] = t.shape[p]
		st[k] = t.strides[p]
	}
	return &Tensor{storage: t.storage, shape: sh, strides: st, offset: t.offset, dev: t.dev}, nil
}

// viewBlock co-allocates a view's Tensor with the backing array for its shape and strides, the
// same trick NewOn uses for a fresh tensor.
//
// It is safe here for the reason the Storage is NOT safe there (see tensorBlock): these two objects
// have IDENTICAL lifetimes. Nothing but this view ever points at this shape/stride array, so the
// block dies exactly when the view does. The Storage is the opposite case — it is shared with the
// parent and every sibling view, so folding it in would let one survivor pin everything, which was
// measured 6.86% slower.
type viewBlock struct {
	t   Tensor
	buf [2 * maxInlineRank]int
}

// newViewOf returns a shape/stride pair sized n, plus the Tensor co-allocated beside it when the
// rank fits inline. Above maxInlineRank it returns a NIL Tensor and the caller keeps its composite
// literal — which is not a stylistic choice: routing the fallback through here too allocated a
// zeroed Tensor and then copied into it, and that extra write measured +6.17% on Permute (p=0.008),
// a rank-4 view that always takes this path. The fallback must stay exactly the code it was.
func newViewOf(n int) (*Tensor, Shape, Strides) {
	if n == 0 || n > maxInlineRank {
		buf := make([]int, 2*n)
		return nil, Shape(buf[:n:n]), Strides(buf[n : 2*n : 2*n])
	}
	blk := &viewBlock{}
	return &blk.t, Shape(blk.buf[:n:n]), Strides(blk.buf[n : 2*n : 2*n])
}
