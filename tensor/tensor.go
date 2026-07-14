package tensor

import "fmt"

// Tensor is an n-dimensional view over a shared Storage (§I.L0). shape and
// strides describe this view; offset is its base element position in the
// storage. Views (reshape/slice/transpose) produce new Tensors that share the
// same Storage with no data copy (§C5).
//
// The Shape/Strides returned by accessors are the tensor's own slices — callers
// must treat them as read-only (numpy/gonum convention) and use Clone to mutate.
type Tensor struct {
	storage *Storage
	shape   Shape
	strides Strides
	offset  int
	dev     Device
}

// New allocates a zeroed contiguous tensor on the default CPU device.
func New(dtype Dtype, shape Shape) *Tensor { return NewOn(CPU(), dtype, shape) }

// NewOn allocates a zeroed contiguous tensor on dev, using its allocator. It
// panics on an invalid dtype or shape (programming error).
func NewOn(dev Device, dtype Dtype, shape Shape) *Tensor {
	if !shape.IsValid() {
		panic(fmt.Sprintf("tensor: invalid shape %v", shape))
	}
	return &Tensor{
		storage: newStorageWith(dev.Allocator(), dtype, shape.Numel()),
		shape:   shape.Clone(),
		strides: RowMajorStrides(shape),
		offset:  0,
		dev:     dev,
	}
}

// FromFloat64 builds a contiguous F64 tensor from data laid out row-major. len
// (data) must equal shape.Numel(). The slice is copied.
func FromFloat64(shape Shape, data []float64) *Tensor {
	t := New(F64, shape)
	if len(data) != t.storage.n {
		panic(fmt.Sprintf("tensor: data len %d != numel %d", len(data), t.storage.n))
	}
	copy(t.storage.F64(), data)
	return t
}

// FromFloat32 builds a contiguous F32 tensor from row-major data.
func FromFloat32(shape Shape, data []float32) *Tensor {
	t := New(F32, shape)
	if len(data) != t.storage.n {
		panic(fmt.Sprintf("tensor: data len %d != numel %d", len(data), t.storage.n))
	}
	copy(t.storage.F32(), data)
	return t
}

// Storage returns the shared backing buffer.
func (t *Tensor) Storage() *Storage { return t.storage }

// Device returns the device this tensor lives on.
func (t *Tensor) Device() Device { return t.dev }

// Dtype returns the element type.
func (t *Tensor) Dtype() Dtype { return t.storage.dtype }

// Shape returns the tensor's shape (read-only).
func (t *Tensor) Shape() Shape { return t.shape }

// Strides returns the tensor's strides (read-only).
func (t *Tensor) Strides() Strides { return t.strides }

// Offset returns the base element offset into the storage.
func (t *Tensor) Offset() int { return t.offset }

// Ndim returns the rank.
func (t *Tensor) Ndim() int { return len(t.shape) }

// Numel returns the element count of this view.
func (t *Tensor) Numel() int { return t.shape.Numel() }

// IsContiguous reports whether this view is densely row-major laid out.
func (t *Tensor) IsContiguous() bool { return t.strides.IsContiguous(t.shape) }

// flatOffset maps a multi-index to a flat storage position, including the view's
// base offset. It panics on rank mismatch (see Offset).
func (t *Tensor) flatOffset(idx []int) int {
	return Offset(t.offset, t.strides, idx)
}

// AtF64 reads the element at multi-index idx as float64 (widening from dtype).
func (t *Tensor) AtF64(idx ...int) float64 {
	return t.storage.atF64(t.flatOffset(idx))
}

// SetF64 writes v at multi-index idx (narrowing to dtype). Writes go through the
// shared storage, so they are visible via every view of the same buffer.
func (t *Tensor) SetF64(v float64, idx ...int) {
	t.storage.setF64(t.flatOffset(idx), v)
}

// Cast returns a fresh contiguous tensor of the given dtype with t's values
// (widened through f64 then narrowed). Returns a contiguous copy even when the
// dtype is unchanged. Used for mixed-precision (§T41) and f32-only accel
// backends (metal, §T20).
func (t *Tensor) Cast(dtype Dtype) *Tensor {
	out := NewOn(t.dev, dtype, t.shape)
	n := t.Numel()
	if n == 0 {
		return out
	}
	// Fast typed path for CONTIGUOUS float casts — the hot F64↔F32 GPU-prep
	// conversion (§base-perf): a direct flat loop, no per-element Unravel heap alloc
	// or AtF64/SetF64 dtype dispatch.
	if t.IsContiguous() {
		b := t.offset
		switch {
		case t.Dtype() == F64 && dtype == F32:
			src, dst := t.storage.F64(), out.storage.F32()
			for i := 0; i < n; i++ {
				dst[i] = float32(src[b+i])
			}
			return out
		case t.Dtype() == F32 && dtype == F64:
			src, dst := t.storage.F32(), out.storage.F64()
			for i := 0; i < n; i++ {
				dst[i] = float64(src[b+i])
			}
			return out
		case t.Dtype() == dtype && dtype == F32:
			copy(out.storage.F32(), t.storage.F32()[b:b+n])
			return out
		case t.Dtype() == dtype && dtype == F64:
			copy(out.storage.F64(), t.storage.F64()[b:b+n])
			return out
		}
	}
	// Generic fallback (strided, or int/other dtypes): widen-through-f64 with an
	// INCREMENTAL source offset — no per-element Unravel alloc.
	nd := len(t.shape)
	if nd == 0 {
		out.storage.setF64(0, t.storage.atF64(t.offset))
		return out
	}
	idx := make([]int, nd)
	off := t.offset
	for pos := 0; pos < n; pos++ {
		out.storage.setF64(pos, t.storage.atF64(off))
		for d := nd - 1; d >= 0; d-- {
			idx[d]++
			off += t.strides[d]
			if idx[d] < t.shape[d] {
				break
			}
			idx[d] = 0
			off -= t.strides[d] * t.shape[d]
		}
	}
	return out
}

// Contiguous returns t if already contiguous, else a fresh contiguous row-major
// copy with the same values. This is an explicit data copy (not a view) used
// before operations that require dense layout (e.g. Reshape).
func (t *Tensor) Contiguous() *Tensor {
	if t.IsContiguous() && t.offset == 0 {
		return t
	}
	out := NewOn(t.dev, t.Dtype(), t.shape)
	n := t.Numel()
	nd := len(t.shape)
	if n == 0 {
		return out
	}
	if nd == 0 { // rank-0 scalar carrying an offset
		out.storage.setF64(0, t.storage.atF64(t.offset))
		return out
	}
	// Strided gather WITHOUT a per-element Unravel heap alloc (§base-perf): walk a
	// running row-major multi-index and keep the source flat offset INCREMENTALLY,
	// and use typed []T access instead of the per-element atF64/setF64 dtype dispatch.
	// Same traversal order as the old Unravel path → identical result. Every op that
	// materializes a transposed/permuted view (attention Q/K/V reshapes) hits this.
	idx := make([]int, nd)
	switch t.Dtype() {
	case F32:
		src, dst := t.storage.F32(), out.storage.F32()
		off := t.offset
		for pos := 0; pos < n; pos++ {
			dst[pos] = src[off]
			for d := nd - 1; d >= 0; d-- {
				idx[d]++
				off += t.strides[d]
				if idx[d] < t.shape[d] {
					break
				}
				idx[d] = 0
				off -= t.strides[d] * t.shape[d]
			}
		}
	case F64:
		src, dst := t.storage.F64(), out.storage.F64()
		off := t.offset
		for pos := 0; pos < n; pos++ {
			dst[pos] = src[off]
			for d := nd - 1; d >= 0; d-- {
				idx[d]++
				off += t.strides[d]
				if idx[d] < t.shape[d] {
					break
				}
				idx[d] = 0
				off -= t.strides[d] * t.shape[d]
			}
		}
	default: // int and any other dtype: keep the generic accessor
		off := t.offset
		for pos := 0; pos < n; pos++ {
			out.storage.setF64(pos, t.storage.atF64(off))
			for d := nd - 1; d >= 0; d-- {
				idx[d]++
				off += t.strides[d]
				if idx[d] < t.shape[d] {
					break
				}
				idx[d] = 0
				off -= t.strides[d] * t.shape[d]
			}
		}
	}
	return out
}
