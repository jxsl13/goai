package tensor

import (
	"strconv"
	"strings"
)

// Shape is the ordered list of dimension sizes of a Tensor, outermost first
// (row-major, §C5). The empty Shape (len 0) is a scalar: it has one element.
// A dimension of size 0 is valid and makes the tensor empty (zero elements).
type Shape []int

// Ndim returns the number of dimensions (rank). Scalar → 0.
func (s Shape) Ndim() int { return len(s) }

// IsValid reports whether every dimension is non-negative. Negative dims are
// meaningless and rejected by constructors.
func (s Shape) IsValid() bool {
	for _, d := range s {
		if d < 0 {
			return false
		}
	}
	return true
}

// IsScalar reports whether s is the zero-dimensional (scalar) shape.
func (s Shape) IsScalar() bool { return len(s) == 0 }

// Numel returns the total number of elements: the product of all dimensions.
// The scalar shape (no dims) has product 1. Any dimension of size 0 makes the
// product 0. Assumes s is valid (see IsValid).
func (s Shape) Numel() int {
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

// IsEmpty reports whether the tensor has zero elements (some dim is 0).
func (s Shape) IsEmpty() bool { return s.Numel() == 0 }

// Equal reports whether s and o have identical dimensions.
func (s Shape) Equal(o Shape) bool {
	if len(s) != len(o) {
		return false
	}
	for i := range s {
		if s[i] != o[i] {
			return false
		}
	}
	return true
}

// Clone returns an independent copy of s (nil for a nil receiver).
func (s Shape) Clone() Shape {
	if s == nil {
		return nil
	}
	c := make(Shape, len(s))
	copy(c, s)
	return c
}

// String renders the shape as e.g. "(2, 3, 4)"; the scalar shape as "()".
func (s Shape) String() string {
	if len(s) == 0 {
		return "()"
	}
	var b strings.Builder
	b.WriteByte('(')
	//perfscan:ignore PS2002 cold debug String(); loop=ndim ~2-5, no Grow gain
	for i, d := range s {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(strconv.Itoa(d))
	}
	b.WriteByte(')')
	return b.String()
}
