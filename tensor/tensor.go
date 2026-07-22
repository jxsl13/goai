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
	sh, st := cloneShapeStrides(shape)
	return &Tensor{
		storage: newStorageWith(dev.Allocator(), dtype, shape.Numel()),
		shape:   sh,
		strides: st,
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
// base offset. It panics on rank mismatch (see Offset). Ranks 1 and 2 — the
// overwhelmingly common At/Set shapes — are unrolled (§base-perf).
func (t *Tensor) flatOffset(idx []int) int {
	st := t.strides
	if len(idx) != len(st) {
		panic("tensor: index rank does not match strides rank")
	}
	switch len(idx) {
	case 1:
		return t.offset + idx[0]*st[0]
	case 2:
		return t.offset + idx[0]*st[0] + idx[1]*st[1]
	}
	off := t.offset
	for i, ix := range idx {
		off += ix * st[i]
	}
	return off
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

// gatherElem is the element type set the devirtualized strided-gather helpers
// operate on. uint16 covers the RAW BIT patterns of F16/BF16 — same-dtype
// copies of half floats are exact bit moves, no f64 round-trip needed.
type gatherElem interface{ ~float32 | ~float64 | ~uint16 }

// gatherCast walks the strided source (shape/strides/base offset off) in
// row-major order and writes D(src)-converted elements into contiguous dst.
// len(dst) must be shape.Numel() > 0 and len(shape) > 0. Only instantiate with
// S/D pairs where the plain Go conversion IS the desired numeric cast
// (float32/float64 combos, or S==D for raw uint16 half bits) — for f16↔f32
// value conversion use the generic setF64 path instead.
// gatherBlocked2D gathers a rank-2 strided view into a contiguous dst in 32×32 tiles.
// For a transpose (innermost stride = #rows) the naive row-major walk reads src down a
// full column between consecutive writes, thrashing the cache; tiling keeps each block's
// source footprint cache-resident. Same (i,j)→dst[i*cols+j] mapping as gatherCast, just
// reordered — bit-identical values.
func gatherBlocked2D[S, D gatherElem](dst []D, src []S, rows, cols, s0, s1, off int) {
	const blk = 16
	for i0 := 0; i0 < rows; i0 += blk {
		iMax := min(i0+blk, rows)
		for j0 := 0; j0 < cols; j0 += blk {
			jMax := min(j0+blk, cols)
			for i := i0; i < iMax; i++ {
				d := dst[i*cols+j0 : i*cols+jMax]
				s := off + i*s0 + j0*s1
				for p := range d { // increment the source offset (no per-element multiply)
					d[p] = D(src[s])
					s += s1
				}
			}
		}
	}
}

func gatherCast[S, D gatherElem](dst []D, src []S, shape Shape, strides Strides, off int) {
	nd := len(shape)
	idx := make([]int, nd)
	for pos := range dst {
		dst[pos] = D(src[off])
		for d := nd - 1; d >= 0; d-- {
			idx[d]++
			off += strides[d]
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
			off -= strides[d] * shape[d]
		}
	}
}

// gatherRows materializes a strided view whose INNERMOST stride is 1 by copying
// whole contiguous rows (length shape[last]) instead of per-element gathers —
// the layout produced by Slice on any non-last axis. Same traversal order as
// gatherCast. Preconditions: len(shape) > 0, strides[last] == 1, shape[last] > 0,
// len(dst) == shape.Numel() > 0.
func gatherRows[T gatherElem](dst, src []T, shape Shape, strides Strides, off int) {
	nd := len(shape)
	row := shape[nd-1]
	rows := len(dst) / row
	idx := make([]int, nd-1)
	for r := 0; r < rows; r++ {
		copy(dst[r*row:(r+1)*row], src[off:off+row])
		for d := nd - 2; d >= 0; d-- {
			idx[d]++
			off += strides[d]
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
			off -= strides[d] * shape[d]
		}
	}
}

// NOTE(§C3 rejected): a 32×32 cache-tiled rank-2 gather variant was measured
// against this walk for transposed-view Contiguous/Cast (12 interleaved,
// core-pinned rounds) and landed within noise (p≈1.0) on this machine, so it
// was reverted. Re-try when a quiet benchmark box is available — the naive walk
// of a transposed matrix is a textbook cache-thrash pattern.

// gatherGeneric is the dtype-agnostic strided fallback: per-element
// atF64/setF64 through the storage accessors. Kept for dtypes without a typed
// fast path (future int/quantized types) and for f16↔f32 value conversions.
func gatherGeneric(out, t *Tensor, n int) {
	nd := len(t.shape)
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
	sd := t.Dtype()
	// Fast typed paths for CONTIGUOUS casts — the hot F64↔F32 GPU-prep conversion
	// and the F16/BF16 (de)quantization casts (§base-perf): direct flat loops, no
	// per-element atF64/setF64 dtype dispatch. Each loop reproduces the generic
	// path's numerics exactly: the generic path is setF64(atF64(x)), i.e. widen to
	// f64 then narrow — for these dtype pairs that composition equals the direct
	// conversion written here (float widening is exact; f64→f16/bf16 in setF64
	// already goes through float32 first).
	if t.IsContiguous() {
		b := t.offset
		switch {
		case sd == F64 && dtype == F32:
			src := t.storage.F64()[b : b+n]
			dst := out.storage.F32()[:n]
			for i, v := range src {
				dst[i] = float32(v)
			}
			return out
		case sd == F32 && dtype == F64:
			src := t.storage.F32()[b : b+n]
			dst := out.storage.F64()[:n]
			for i, v := range src {
				dst[i] = float64(v)
			}
			return out
		case sd == dtype && dtype == F32:
			copy(out.storage.F32(), t.storage.F32()[b:b+n])
			return out
		case sd == dtype && dtype == F64:
			copy(out.storage.F64(), t.storage.F64()[b:b+n])
			return out
		case sd == dtype && (dtype == F16 || dtype == BF16):
			// Same 16-bit dtype: raw bit copy (f16→f64→f16 round-trips exactly).
			copy(out.storage.U16(), t.storage.U16()[b:b+n])
			return out
		case sd == F16 && dtype == F32:
			src := t.storage.U16()[b : b+n]
			dst := out.storage.F32()[:n]
			for i, v := range src {
				dst[i] = f16ToF32(v)
			}
			return out
		case sd == F16 && dtype == F64:
			src := t.storage.U16()[b : b+n]
			dst := out.storage.F64()[:n]
			for i, v := range src {
				dst[i] = float64(f16ToF32(v))
			}
			return out
		case sd == BF16 && dtype == F32:
			src := t.storage.U16()[b : b+n]
			dst := out.storage.F32()[:n]
			for i, v := range src {
				dst[i] = bf16ToF32(v)
			}
			return out
		case sd == BF16 && dtype == F64:
			src := t.storage.U16()[b : b+n]
			dst := out.storage.F64()[:n]
			for i, v := range src {
				dst[i] = float64(bf16ToF32(v))
			}
			return out
		case sd == F32 && dtype == F16:
			src := t.storage.F32()[b : b+n]
			dst := out.storage.U16()[:n]
			for i, v := range src {
				dst[i] = f32ToF16(v)
			}
			return out
		case sd == F64 && dtype == F16:
			src := t.storage.F64()[b : b+n]
			dst := out.storage.U16()[:n]
			for i, v := range src {
				dst[i] = f32ToF16(float32(v))
			}
			return out
		case sd == F32 && dtype == BF16:
			src := t.storage.F32()[b : b+n]
			dst := out.storage.U16()[:n]
			for i, v := range src {
				dst[i] = f32ToBF16(v)
			}
			return out
		case sd == F64 && dtype == BF16:
			src := t.storage.F64()[b : b+n]
			dst := out.storage.U16()[:n]
			for i, v := range src {
				dst[i] = f32ToBF16(float32(v))
			}
			return out
		case sd == F16 && dtype == BF16:
			src := t.storage.U16()[b : b+n]
			dst := out.storage.U16()[:n]
			for i, v := range src {
				dst[i] = f32ToBF16(f16ToF32(v))
			}
			return out
		case sd == BF16 && dtype == F16:
			src := t.storage.U16()[b : b+n]
			dst := out.storage.U16()[:n]
			for i, v := range src {
				dst[i] = f32ToF16(bf16ToF32(v))
			}
			return out
		}
	}
	nd := len(t.shape)
	if nd == 0 {
		out.storage.setF64(0, t.storage.atF64(t.offset))
		return out
	}
	// Strided typed paths for the f32/f64 combos (transposed GPU-prep casts):
	// incremental-offset walk, direct conversion, no dtype dispatch per element. A
	// rank-2 non-row-run source (a transposed view) is tiled like Contiguous so the
	// cast doesn't thrash the cache sweeping a source column per output row.
	blk2d := nd == 2 && t.strides[nd-1] != 1
	switch {
	case sd == F64 && dtype == F32:
		if blk2d {
			gatherBlocked2D(out.storage.F32()[:n], t.storage.F64(), t.shape[0], t.shape[1], t.strides[0], t.strides[1], t.offset)
		} else {
			gatherCast(out.storage.F32()[:n], t.storage.F64(), t.shape, t.strides, t.offset)
		}
	case sd == F32 && dtype == F64:
		if blk2d {
			gatherBlocked2D(out.storage.F64()[:n], t.storage.F32(), t.shape[0], t.shape[1], t.strides[0], t.strides[1], t.offset)
		} else {
			gatherCast(out.storage.F64()[:n], t.storage.F32(), t.shape, t.strides, t.offset)
		}
	case sd == dtype && dtype == F32:
		if blk2d {
			gatherBlocked2D(out.storage.F32()[:n], t.storage.F32(), t.shape[0], t.shape[1], t.strides[0], t.strides[1], t.offset)
		} else {
			gatherCast(out.storage.F32()[:n], t.storage.F32(), t.shape, t.strides, t.offset)
		}
	case sd == dtype && dtype == F64:
		if blk2d {
			gatherBlocked2D(out.storage.F64()[:n], t.storage.F64(), t.shape[0], t.shape[1], t.strides[0], t.strides[1], t.offset)
		} else {
			gatherCast(out.storage.F64()[:n], t.storage.F64(), t.shape, t.strides, t.offset)
		}
	case sd == dtype && (dtype == F16 || dtype == BF16):
		if blk2d {
			gatherBlocked2D(out.storage.U16()[:n], t.storage.U16(), t.shape[0], t.shape[1], t.strides[0], t.strides[1], t.offset)
		} else {
			gatherCast(out.storage.U16()[:n], t.storage.U16(), t.shape, t.strides, t.offset)
		}
	default:
		// Strided cross-dtype half casts: widen-through-f64 generic walk.
		gatherGeneric(out, t, n)
	}
	return out
}

// Contiguous returns t if already contiguous with offset 0, else a fresh
// contiguous row-major copy with the same values. This is an explicit data copy
// (not a view) used before operations that require dense layout (e.g. Reshape).
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
	// Densely laid out but carrying a base offset (e.g. a dim-0 Slice of a
	// contiguous tensor): a single flat copy, no stride walk (§base-perf).
	if t.IsContiguous() {
		b := t.offset
		switch t.Dtype() {
		case F32:
			copy(out.storage.F32(), t.storage.F32()[b:b+n])
			return out
		case F64:
			copy(out.storage.F64(), t.storage.F64()[b:b+n])
			return out
		case F16, BF16:
			copy(out.storage.U16(), t.storage.U16()[b:b+n])
			return out
		}
	}
	// Strided gather WITHOUT a per-element Unravel heap alloc (§base-perf): walk a
	// running row-major multi-index and keep the source flat offset INCREMENTALLY,
	// with typed []T access instead of per-element atF64/setF64 dtype dispatch.
	// When the innermost stride is 1 (Slice on a non-last axis) whole rows are
	// contiguous runs and are moved with copy(). Same traversal order as the old
	// Unravel path → identical result. Every op that materializes a transposed/
	// permuted view (attention Q/K/V reshapes) hits this.
	rowRuns := t.strides[nd-1] == 1 && t.shape[nd-1] > 0
	// A rank-2 non-row-run view is a transpose/column-strided gather — tile it so the
	// source footprint stays cache-resident (blocked transpose), instead of the naive
	// column-sweep gatherCast that thrashes the cache on large matrices.
	blk2d := nd == 2 && !rowRuns
	switch t.Dtype() {
	case F32:
		switch {
		case rowRuns:
			gatherRows(out.storage.F32()[:n], t.storage.F32(), t.shape, t.strides, t.offset)
		case blk2d:
			gatherBlocked2D(out.storage.F32()[:n], t.storage.F32(), t.shape[0], t.shape[1], t.strides[0], t.strides[1], t.offset)
		default:
			gatherCast(out.storage.F32()[:n], t.storage.F32(), t.shape, t.strides, t.offset)
		}
	case F64:
		switch {
		case rowRuns:
			gatherRows(out.storage.F64()[:n], t.storage.F64(), t.shape, t.strides, t.offset)
		case blk2d:
			gatherBlocked2D(out.storage.F64()[:n], t.storage.F64(), t.shape[0], t.shape[1], t.strides[0], t.strides[1], t.offset)
		default:
			gatherCast(out.storage.F64()[:n], t.storage.F64(), t.shape, t.strides, t.offset)
		}
	case F16, BF16:
		// Same-dtype raw bit moves — exact, no f64 round-trip.
		switch {
		case rowRuns:
			gatherRows(out.storage.U16()[:n], t.storage.U16(), t.shape, t.strides, t.offset)
		case blk2d:
			gatherBlocked2D(out.storage.U16()[:n], t.storage.U16(), t.shape[0], t.shape[1], t.strides[0], t.strides[1], t.offset)
		default:
			gatherCast(out.storage.U16()[:n], t.storage.U16(), t.shape, t.strides, t.offset)
		}
	default: // any future dtype: keep the generic accessor
		gatherGeneric(out, t, n)
	}
	return out
}
