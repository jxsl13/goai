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

// maxInlineRank is the rank up to which a fresh tensor carries its shape and strides inside its
// own allocation block.
//
// TWO, and the value was measured rather than picked. The block reserves 2*maxInlineRank ints
// whatever the actual rank, so anything above the real rank is waste on every tensor. Swept on
// BenchmarkJambaDecode: rank 2, 3 and 4 all give the same 957 allocs/op, but B/op runs 542519,
// 544834 and 547159 against a 542517 baseline — so 2 buys the whole allocation win for nothing,
// while 3 and 4 pay bytes for coverage no tensor in these paths needs.
//
// Identical allocation counts across the three also say something worth recording: nothing on a
// decode path here is rank 3 or above. A workload that is would take the fallback below, which is
// the old behavior, so raising this is a byte-for-coverage trade and never a regression.
const maxInlineRank = 2

// tensorBlock co-allocates TWO of the objects NewOn always creates together: the Tensor and the
// backing array for shape and strides.
//
// A fresh tensor cost FIVE allocations — Tensor, Storage, the shape/stride array, the data slice,
// and the interface box Storage.data pays to hold that slice — and a Jamba decode step creates
// about 158 of them, which was over half its allocation count. Folding the shape/stride array into
// the Tensor drops that to four.
//
// THE STORAGE IS DELIBERATELY NOT IN HERE, and that is the whole point of this paragraph, because
// putting it here is the obvious next step and it has already been tried and MEASURED WORSE. Views
// are why. Reshape and friends build a new Tensor pointing at the SAME Storage, so if the Storage
// lived inside this block, one surviving view would pin the entire block — the original Tensor and
// its shape/stride buffer included — instead of a bare 48-byte Storage. The A/B was unambiguous:
// allocations fell about 23%, and the decode step ran 6.86% SLOWER, because retention beat the
// allocation saving. An allocation count is not the objective; it is a proxy that fails exactly
// when the removed object outlives the one it was folded into.
//
// So: fold objects whose lifetimes are IDENTICAL, never objects one of which can outlive the other.
// The shape/stride array qualifies — nothing but this Tensor ever points at it.
type tensorBlock struct {
	t   Tensor
	buf [2 * maxInlineRank]int
}

// NewOn allocates a zeroed contiguous tensor on dev, using its allocator. It
// panics on an invalid dtype or shape (programming error).
func NewOn(dev Device, dtype Dtype, shape Shape) *Tensor {
	if !shape.IsValid() {
		// shape.String() rather than a %v: passing the slice to Sprintf as an interface makes escape
		// analysis mark shape as LEAKING, which forces every caller to heap-allocate its shape
		// literal even though this panic never fires. String is proven non-escaping.
		panic("tensor: invalid shape " + shape.String())
	}
	n := len(shape)
	if n == 0 || n > maxInlineRank {
		// Rank 0 keeps the nil/empty distinction cloneShapeStrides is careful about; high ranks
		// are rare enough that the block would be waste.
		sh, st := cloneShapeStrides(shape)
		return &Tensor{
			storage: newStorageWith(dev.Allocator(), dtype, shape.Numel()),
			shape:   sh,
			strides: st,
			offset:  0,
			dev:     dev,
		}
	}
	blk := &tensorBlock{}
	sh := Shape(blk.buf[:n:n])
	st := Strides(blk.buf[n : 2*n : 2*n])
	acc := 1
	for i := n - 1; i >= 0; i-- {
		sh[i] = shape[i]
		st[i] = acc
		acc *= shape[i]
	}
	blk.t = Tensor{storage: newStorageWith(dev.Allocator(), dtype, acc), shape: sh, strides: st, offset: 0, dev: dev}
	return &blk.t
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
	// Hoist the INNERMOST axis out of the odometer: its stride is constant across a
	// run, so the run is a straight strided walk and the odometer ticks once per run
	// instead of once per element (the same trick gatherBlocked2D uses inline).
	// Traversal order and the per-element D() conversion are unchanged, and over a
	// full run the innermost axis contributes inner*sInner - sInner*inner = 0 to off,
	// so the outer tick is identical — bit-identical values.
	inner, sInner := shape[nd-1], strides[nd-1]
	idx := make([]int, nd)
	for pos := 0; pos < len(dst); pos += inner {
		run := dst[pos : pos+inner]
		s := off
		for p := range run {
			run[p] = D(src[s])
			s += sInner
		}
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
	// Hoist the INNERMOST axis out of the odometer, exactly as gatherCast does: its
	// stride is constant across a run, so the run is a straight strided walk and the
	// odometer ticks once per RUN instead of once per element. Bit-identical by
	// construction — pure traversal reordering, every destination still reads the same
	// source offset through the same atF64/setF64 pair, and over a full run the
	// innermost axis contributes inner*sInner - sInner*inner = 0 to off.
	//
	// The per-element accessor dispatch is NOT addressed here; that is the separate
	// (and probably larger) PS1001 half, which needs a typed switch reproducing this
	// path's widen-through-float64 semantics exactly.
	if gatherHalfTyped(out, t, n) {
		return
	}
	if nd > 0 && t.shape[nd-1] > 0 && n%t.shape[nd-1] == 0 {
		inner, sInner := t.shape[nd-1], t.strides[nd-1]
		for pos := 0; pos < n; pos += inner {
			s := off
			for p := 0; p < inner; p++ {
				out.storage.setF64(pos+p, t.storage.atF64(s))
				s += sInner
			}
			for d := nd - 2; d >= 0; d-- {
				idx[d]++
				off += t.strides[d]
				if idx[d] < t.shape[d] {
					break
				}
				idx[d] = 0
				off -= t.strides[d] * t.shape[d]
			}
		}
		return
	}
	//perfscan:ignore PS4005 fallback for shapes the strip-mine above declines
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
	// Fold axes that are already dense with each other before deciding how to walk: a rank-4 slice
	// of a middle axis is usually a rank-2 walk in disguise (see coalesceDims).
	cshape, cstrides := coalesceDims(t.shape, t.strides)
	cnd := len(cshape)
	blk2d := cnd == 2 && cstrides[cnd-1] != 1
	switch {
	case sd == F64 && dtype == F32:
		if blk2d {
			gatherBlocked2D(out.storage.F32()[:n], t.storage.F64(), cshape[0], cshape[1], cstrides[0], cstrides[1], t.offset)
		} else {
			gatherCast(out.storage.F32()[:n], t.storage.F64(), cshape, cstrides, t.offset)
		}
	case sd == F32 && dtype == F64:
		if blk2d {
			gatherBlocked2D(out.storage.F64()[:n], t.storage.F32(), cshape[0], cshape[1], cstrides[0], cstrides[1], t.offset)
		} else {
			gatherCast(out.storage.F64()[:n], t.storage.F32(), cshape, cstrides, t.offset)
		}
	case sd == dtype && dtype == F32:
		if blk2d {
			gatherBlocked2D(out.storage.F32()[:n], t.storage.F32(), cshape[0], cshape[1], cstrides[0], cstrides[1], t.offset)
		} else {
			gatherCast(out.storage.F32()[:n], t.storage.F32(), cshape, cstrides, t.offset)
		}
	case sd == dtype && dtype == F64:
		if blk2d {
			gatherBlocked2D(out.storage.F64()[:n], t.storage.F64(), cshape[0], cshape[1], cstrides[0], cstrides[1], t.offset)
		} else {
			gatherCast(out.storage.F64()[:n], t.storage.F64(), cshape, cstrides, t.offset)
		}
	case sd == dtype && (dtype == F16 || dtype == BF16):
		if blk2d {
			gatherBlocked2D(out.storage.U16()[:n], t.storage.U16(), cshape[0], cshape[1], cstrides[0], cstrides[1], t.offset)
		} else {
			gatherCast(out.storage.U16()[:n], t.storage.U16(), cshape, cstrides, t.offset)
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
	cshape, cstrides := coalesceDims(t.shape, t.strides) // see coalesceDims: fold dense axis pairs
	cnd := len(cshape)
	rowRuns := cstrides[cnd-1] == 1 && cshape[cnd-1] > 0
	// A rank-2 non-row-run view is a transpose/column-strided gather — tile it so the
	// source footprint stays cache-resident (blocked transpose), instead of the naive
	// column-sweep gatherCast that thrashes the cache on large matrices.
	blk2d := cnd == 2 && !rowRuns
	switch t.Dtype() {
	case F32:
		switch {
		case rowRuns:
			gatherRows(out.storage.F32()[:n], t.storage.F32(), cshape, cstrides, t.offset)
		case blk2d:
			gatherBlocked2D(out.storage.F32()[:n], t.storage.F32(), cshape[0], cshape[1], cstrides[0], cstrides[1], t.offset)
		default:
			gatherCast(out.storage.F32()[:n], t.storage.F32(), cshape, cstrides, t.offset)
		}
	case F64:
		switch {
		case rowRuns:
			gatherRows(out.storage.F64()[:n], t.storage.F64(), cshape, cstrides, t.offset)
		case blk2d:
			gatherBlocked2D(out.storage.F64()[:n], t.storage.F64(), cshape[0], cshape[1], cstrides[0], cstrides[1], t.offset)
		default:
			gatherCast(out.storage.F64()[:n], t.storage.F64(), cshape, cstrides, t.offset)
		}
	case F16, BF16:
		// Same-dtype raw bit moves — exact, no f64 round-trip.
		switch {
		case rowRuns:
			gatherRows(out.storage.U16()[:n], t.storage.U16(), cshape, cstrides, t.offset)
		case blk2d:
			gatherBlocked2D(out.storage.U16()[:n], t.storage.U16(), cshape[0], cshape[1], cstrides[0], cstrides[1], t.offset)
		default:
			gatherCast(out.storage.U16()[:n], t.storage.U16(), cshape, cstrides, t.offset)
		}
	default: // any future dtype: keep the generic accessor
		gatherGeneric(out, t, n)
	}
	return out
}

// gatherHalfTyped is gatherGeneric's devirtualized arm for the STRIDED HALF CASTS that
// actually reach that function: a f16/bf16 view widened to f32/f64, or the reverse.
//
// The generic walk pays an interface type-switch on Storage.data for EVERY element,
// twice — once to decode the source and once to encode the destination. Hoisting both
// switches out of the loop leaves a typed strided walk. Everything else still routes
// through the accessor path, which is why this returns a bool rather than handling all
// sixteen dtype combinations.
//
// BIT-IDENTITY IS BY EXPRESSION, NOT BY ARGUMENT: each arm performs the SAME
// conversion the accessors perform, written out — decode through float64 exactly as
// Storage.atF64 does, encode exactly as Storage.setF64 does. The float64 intermediate
// is kept even where it is provably redundant (f32->f64->f32 round-trips exactly),
// because the point is to mirror the original expression rather than to reason about
// when a widening may be elided. TestGatherGenericMatchesOdometerExact holds every arm
// to the frozen per-element oracle at tolerance 0, across all six dtype pairs — which
// is exactly the proof PS6004 exists to demand, so the finding is accepted here rather
// than left open. Every arm is mutation-probed by CORRUPTION, not by disabling it:
// disabling an arm falls back to the accessor path, which IS the oracle, so it would
// stay green and prove nothing.
//
//perfscan:ignore PS6004 bit-identity proven by TestGatherGenericMatchesOdometerExact
func gatherHalfTyped(out, t *Tensor, n int) bool {
	nd := len(t.shape)
	if nd == 0 || t.shape[nd-1] <= 0 || n%t.shape[nd-1] != 0 {
		return false
	}
	inner, sInner := t.shape[nd-1], t.strides[nd-1]

	srcU16, srcIsU16 := t.storage.u16, t.storage.u16 != nil
	dstU16, dstIsU16 := out.storage.u16, out.storage.u16 != nil
	srcF32, srcIsF32 := t.storage.f32, t.storage.f32 != nil
	dstF32, dstIsF32 := out.storage.f32, out.storage.f32 != nil
	srcF64, srcIsF64 := t.storage.f64, t.storage.f64 != nil
	dstF64, dstIsF64 := out.storage.f64, out.storage.f64 != nil
	srcBF, dstBF := t.storage.dtype == BF16, out.storage.dtype == BF16

	// decode picks the source reader once; encode picks the destination writer once.
	// Only the half-cast combinations are claimed here.
	var run func(dstOff, srcOff int)
	switch {
	case srcIsU16 && dstIsF32:
		run = func(dstOff, s int) {
			for p := 0; p < inner; p++ {
				if srcBF {
					dstF32[dstOff+p] = float32(float64(bf16ToF32(srcU16[s])))
				} else {
					dstF32[dstOff+p] = float32(float64(f16ToF32(srcU16[s])))
				}
				s += sInner
			}
		}
	case srcIsU16 && dstIsF64:
		run = func(dstOff, s int) {
			for p := 0; p < inner; p++ {
				if srcBF {
					dstF64[dstOff+p] = float64(bf16ToF32(srcU16[s]))
				} else {
					dstF64[dstOff+p] = float64(f16ToF32(srcU16[s]))
				}
				s += sInner
			}
		}
	case srcIsF32 && dstIsU16:
		run = func(dstOff, s int) {
			for p := 0; p < inner; p++ {
				v := float64(srcF32[s])
				if dstBF {
					dstU16[dstOff+p] = f32ToBF16(float32(v))
				} else {
					dstU16[dstOff+p] = f32ToF16(float32(v))
				}
				s += sInner
			}
		}
	case srcIsF64 && dstIsU16:
		run = func(dstOff, s int) {
			for p := 0; p < inner; p++ {
				v := srcF64[s]
				if dstBF {
					dstU16[dstOff+p] = f32ToBF16(float32(v))
				} else {
					dstU16[dstOff+p] = f32ToF16(float32(v))
				}
				s += sInner
			}
		}
	default:
		return false
	}

	idx := make([]int, nd)
	off := t.offset
	for pos := 0; pos < n; pos += inner {
		run(pos, off)
		for d := nd - 2; d >= 0; d-- {
			idx[d]++
			off += t.strides[d]
			if idx[d] < t.shape[d] {
				break
			}
			idx[d] = 0
			off -= t.strides[d] * t.shape[d]
		}
	}
	return true
}

// coalesceDims merges adjacent axes that are already dense with respect to each other, returning an
// equivalent shape/stride pair of lower rank. Axes d and d+1 can merge exactly when
// strides[d] == strides[d+1]*shape[d+1], which is what it means for the pair to describe one
// contiguous run in the source.
//
// Every strided gather below walks the view at its DECLARED rank, which for the common case — a
// Slice on a non-innermost axis of a rank-4 attention tensor — means thousands of short runs where
// a handful of long ones describe the identical traversal. On the shape this package's own
// benchmark builds, {4,6,64,64} over strides {32768,4096,64,1}, the walk is 1536 runs of 64
// elements; coalesced it is 4 runs of 24576. Nothing about the ORDER changes — this is a
// re-description of the same offset sequence, not a different one — so every gather stays
// bit-identical and only the call overhead per run disappears.
func coalesceDims(shape, strides []int) ([]int, []int) {
	nd := len(shape)
	if nd < 2 {
		return shape, strides
	}
	// Scan first, allocate only if something actually merges. Contiguous and Cast run on every
	// materialization, most of which cannot be folded at all, and two allocations there cost more
	// than the walk they were meant to shorten — measured as an 18% REGRESSION on an
	// offset-view benchmark before this early-out went in.
	merges := false
	for d := 1; d < nd; d++ {
		if strides[d-1] == strides[d]*shape[d] {
			merges = true
			break
		}
	}
	if !merges {
		return shape, strides
	}
	sh := make([]int, 0, nd)
	st := make([]int, 0, nd)
	sh = append(sh, shape[0])
	st = append(st, strides[0])
	for d := 1; d < nd; d++ {
		last := len(sh) - 1
		if st[last] == strides[d]*shape[d] {
			sh[last] *= shape[d] // the pair is one dense run: fold it
			st[last] = strides[d]
			continue
		}
		sh = append(sh, shape[d])
		st = append(st, strides[d])
	}
	return sh, st
}
