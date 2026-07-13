package backend

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// ConcatPlan validates a set of input shapes for concatenation along axis and
// returns the output shape, the per-input start offsets along the (normalized)
// axis, and the normalized axis. It is the single source of truth shared by the
// concat forward kernel and its VJP so they cannot disagree. All inputs must have
// the same rank and equal extents in every dimension except the concat axis; the
// output extent along the axis is the sum of the inputs'. A negative axis counts
// from the end (numpy.concatenate convention).
func ConcatPlan(shapes []tensor.Shape, axis int) (out tensor.Shape, offsets []int, ax int, err error) {
	if len(shapes) == 0 {
		return nil, nil, 0, fmt.Errorf("backend: concat needs ≥1 input")
	}
	nd := len(shapes[0])
	if nd == 0 {
		return nil, nil, 0, fmt.Errorf("backend: concat needs rank≥1 inputs")
	}
	ax = axis
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return nil, nil, 0, fmt.Errorf("backend: concat axis %d out of range for rank %d", axis, nd)
	}
	out = append(tensor.Shape(nil), shapes[0]...)
	offsets = make([]int, len(shapes))
	sum := 0
	for i, s := range shapes {
		if len(s) != nd {
			return nil, nil, 0, fmt.Errorf("backend: concat input %d rank %d != %d", i, len(s), nd)
		}
		for d := range nd {
			if d != ax && s[d] != shapes[0][d] {
				return nil, nil, 0, fmt.Errorf("backend: concat input %d shape %v mismatches %v on axis %d", i, s, shapes[0], d)
			}
		}
		offsets[i] = sum
		sum += s[ax]
	}
	out[ax] = sum
	return out, offsets, ax, nil
}

// SlicePlan validates a slice of shape along axis over the half-open range
// [start,end) and returns the normalized axis and the output shape (same as the
// input except the axis extent becomes end−start). Shared by the slice forward
// kernel and its VJP. A negative axis counts from the end (numpy convention).
func SlicePlan(shape tensor.Shape, axis, start, end int) (ax int, out tensor.Shape, err error) {
	nd := len(shape)
	if nd == 0 {
		return 0, nil, fmt.Errorf("backend: slice needs a rank≥1 tensor")
	}
	ax = axis
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return 0, nil, fmt.Errorf("backend: slice axis %d out of range for rank %d", axis, nd)
	}
	if start < 0 || end > shape[ax] || start >= end {
		return 0, nil, fmt.Errorf("backend: slice range [%d,%d) invalid for axis %d extent %d", start, end, ax, shape[ax])
	}
	out = append(tensor.Shape(nil), shape...)
	out[ax] = end - start
	return ax, out, nil
}

// BroadcastShape returns the common broadcast shape of a and b (numpy rules: the
// shapes are right-aligned; each pair of dims must be equal or one must be 1; the
// result dim is their max), or an error if they are incompatible.
func BroadcastShape(a, b tensor.Shape) (tensor.Shape, error) {
	n := max(len(a), len(b))
	out := make(tensor.Shape, n)
	for i := range n {
		da, db := 1, 1
		if j := i - (n - len(a)); j >= 0 {
			da = a[j]
		}
		if j := i - (n - len(b)); j >= 0 {
			db = b[j]
		}
		switch {
		case da == db, db == 1:
			out[i] = da
		case da == 1:
			out[i] = db
		default:
			return nil, fmt.Errorf("backend: shapes %v and %v not broadcast-compatible (axis %d: %d vs %d)", a, b, i, da, db)
		}
	}
	return out, nil
}

// CumsumPlan normalizes a cumsum/scan axis for a shape and returns it together with
// the shape minus that axis (used to enumerate the lines along the axis). A negative
// axis counts from the end. Shared by the cumsum forward kernel and its VJP.
func CumsumPlan(shape tensor.Shape, axis int) (ax int, reduced tensor.Shape, err error) {
	nd := len(shape)
	if nd == 0 {
		return 0, nil, fmt.Errorf("backend: cumsum needs a rank≥1 tensor")
	}
	ax = axis
	if ax < 0 {
		ax += nd
	}
	if ax < 0 || ax >= nd {
		return 0, nil, fmt.Errorf("backend: cumsum axis %d out of range for rank %d", axis, nd)
	}
	reduced = make(tensor.Shape, 0, nd-1)
	for d, s := range shape {
		if d != ax {
			reduced = append(reduced, s)
		}
	}
	return ax, reduced, nil
}

// FillLineCoord writes into coord (len == rank) the full coordinates of the line
// indexed by l in the reduced (axis-removed) shape, leaving coord[ax] for the caller
// to walk along the axis.
func FillLineCoord(coord []int, l int, reduced tensor.Shape, ax int) {
	base := tensor.Unravel(l, reduced)
	bi := 0
	for d := range coord {
		if d == ax {
			continue
		}
		coord[d] = base[bi]
		bi++
	}
}

// BroadcastCoords fills dst (len == len(inShape)) with the input coordinates that
// output coordinate oc maps to under broadcasting: a size-1 input axis reads index
// 0, otherwise the aligned output coordinate. offset = len(oc) − len(inShape).
func BroadcastCoords(dst, oc []int, inShape tensor.Shape, offset int) {
	for a, d := range inShape {
		if d == 1 {
			dst[a] = 0
		} else {
			dst[a] = oc[a+offset]
		}
	}
}

// BroadcastPlan validates that inShape can be broadcast to target (numpy rules:
// right-align the shapes; each input dim must be 1 or equal to the target dim; the
// input may have fewer dims, treated as leading 1s) and returns the alignment
// offset = len(target) − len(inShape). Shared by the broadcast forward kernel and
// its VJP so they map coordinates identically.
func BroadcastPlan(inShape, target tensor.Shape) (offset int, err error) {
	offset = len(target) - len(inShape)
	if offset < 0 {
		return 0, fmt.Errorf("backend: cannot broadcast rank %d to smaller rank %d", len(inShape), len(target))
	}
	for a, d := range inShape {
		td := target[a+offset]
		if td < 0 {
			return 0, fmt.Errorf("backend: broadcast target %v has a negative dimension", target)
		}
		if d != td && d != 1 {
			return 0, fmt.Errorf("backend: cannot broadcast shape %v to %v (axis %d: %d vs %d)", inShape, target, a, d, td)
		}
	}
	return offset, nil
}
