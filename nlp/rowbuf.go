package nlp

import "github.com/jxsl13/goai/tensor"

// rowBuf is an amortized-growth row buffer for KV caches. Where the old
// concatRows append reallocated a fresh [t+1, width] tensor and copied all t
// existing rows on EVERY decode step — O(T²) copy traffic per layer over a
// T-token decode — rowBuf keeps a backing [cap, width] tensor plus a length,
// writes each new token's row in place, and only reallocates when full
// (doubling, so each row is copied O(1) times overall): amortized O(T) total.
//
// Append returns buf.Slice(0, 0, n) — a zero-copy view of the first n rows.
// Because the backing tensor is contiguous with offset 0 and the slice is on
// axis 0 starting at 0, the view has row-major strides and offset 0, so
// IsContiguous() is true and Contiguous() (which every kernel calls on its
// inputs) returns the view itself without copying. Kernels index storage by
// shape-derived geometry, never by len(storage), so the backing capacity
// beyond n rows is invisible to them. Rows already handed out in a view are
// append-only — later Appends write only row n and beyond (or a fresh backing
// on growth) — so previously returned views stay valid and unchanged.
type rowBuf struct {
	buf *tensor.Tensor // backing [cap, width] contiguous tensor; nil until first append
	n   int            // rows in use; the handed-out view is buf[0:n]
}

// Append appends row [r, width] (r is 1 in the decode paths) after the cur
// view and returns the refreshed [n+r, width] view, value-identical to
// concatRows(cur, row) but without recopying the existing rows.
//
// cur is the cache's current public view. When it is not the view this buffer
// handed out — a fresh cache (nil), a struct-literal cache, or a view some
// caller replaced, truncated, or evicted — the buffer resynchronizes by
// adopting cur as the new base (at most one O(rows) copy via Contiguous),
// after which growth is amortized O(1) per token again. Mixed or exotic
// dtypes fall back to concatRows and adopt its result.
func (r *rowBuf) Append(cur, row *tensor.Tensor) *tensor.Tensor {
	if row == nil {
		return cur
	}
	if cur == nil {
		// Truncate(0) deliberately returns nil while retaining the allocation. Keep
		// that empty backing; an externally cleared non-empty view still resets it.
		if r.n != 0 {
			r.buf, r.n = nil, 0
		}
	} else if !r.owns(cur) {
		// Foreign view: adopt it as the backing. Contiguous() copies at most
		// once (and not at all for a contiguous offset-0 tensor); the adopted
		// backing has cap == rows, so the next Append grows into a fresh
		// allocation instead of writing into storage the caller may share.
		c := cur.Contiguous()
		r.buf, r.n = c, c.Shape()[0]
	}
	rows, width := row.Shape()[0], row.Shape()[1]
	dt := row.Dtype()
	if r.buf != nil && (r.buf.Dtype() != dt || r.buf.Shape()[1] != width) {
		// Dtype/width mismatch: keep concatRows' generic semantics exactly.
		out := concatRows(cur, row)
		r.buf, r.n = out, out.Shape()[0]
		return out
	}
	if !typedRowCopyOK(dt) {
		out := concatRows(cur, row)
		r.buf, r.n = out, out.Shape()[0]
		return out
	}
	need := r.n + rows
	if r.buf == nil || need > r.buf.Shape()[0] {
		// Grow by doubling: each existing row is recopied O(log T) times total
		// across a decode, amortizing to O(1) copies per row.
		newCap := max(2*need, 8)
		nb := tensor.NewOn(row.Device(), dt, tensor.Shape{newCap, width})
		if r.n > 0 {
			copyRows(nb, 0, r.buf, r.n*width)
		}
		r.buf = nb
	}
	copyRows(r.buf, r.n*width, row, rows*width)
	r.n = need
	view, err := r.buf.Slice(0, 0, r.n)
	if err != nil { // bounds are ours by construction; unreachable
		panic(err)
	}
	return view
}

// Truncate discards rows [n:] while retaining the backing allocation for the next
// append. LayerSkip uses it to roll speculative KV rows back after a rejection.
// cur may be a foreign public cache view; as in Append, it is adopted first.
func (r *rowBuf) Truncate(cur *tensor.Tensor, n int) *tensor.Tensor {
	if cur == nil {
		if n != 0 {
			panic("nlp: truncate nil row buffer to nonzero length")
		}
		r.n = 0
		return nil
	}
	if !r.owns(cur) {
		c := cur.Contiguous()
		r.buf, r.n = c, c.Shape()[0]
	}
	if n < 0 || n > r.n {
		panic("nlp: row buffer truncate outside current length")
	}
	r.n = n
	if n == 0 {
		return nil
	}
	view, err := r.buf.Slice(0, 0, n)
	if err != nil { // bounds are ours by construction; unreachable
		panic(err)
	}
	return view
}

// CopyPrefix replaces the logical rows with the first n rows of src while retaining
// a compatible backing allocation. Prefix-pool cloning uses this instead of making
// a temporary src.Slice and then routing it through Append: one typed block copy and
// one destination view are sufficient. src is never adopted or shared.
func (r *rowBuf) CopyPrefix(src *tensor.Tensor, n int) *tensor.Tensor {
	if src == nil || src.Ndim() != 2 || n < 0 || n > src.Shape()[0] {
		panic("nlp: row buffer copy prefix outside source")
	}
	if n == 0 {
		r.n = 0
		return nil
	}
	width, dt := src.Shape()[1], src.Dtype()
	if !typedRowCopyOK(dt) {
		prefix, err := src.Slice(0, 0, n)
		if err != nil { // bounds checked above; unreachable
			panic(err)
		}
		out := prefix.Cast(dt) // Cast always makes an independent contiguous copy
		r.buf, r.n = out, n
		return out
	}
	if r.buf == nil || r.buf.Dtype() != dt || r.buf.Shape()[1] != width || r.buf.Shape()[0] < n {
		capRows := max(2*n, 8)
		r.buf = tensor.NewOn(src.Device(), dt, tensor.Shape{capRows, width})
	}
	copyRows(r.buf, 0, src, n*width)
	r.n = n
	view, err := r.buf.Slice(0, 0, n)
	if err != nil { // capacity is ours by construction; unreachable
		panic(err)
	}
	return view
}

// typedRowCopyOK reports whether dtype has a typed storage slice we can move
// with copy(); anything else routes through concatRows' generic path.
func typedRowCopyOK(dt tensor.Dtype) bool {
	switch dt {
	case tensor.F32, tensor.F64, tensor.F16, tensor.BF16:
		return true
	}
	return false
}

// copyRows moves nEl elements of src (same dtype as dst, read contiguously
// from its start) into dst's storage at flat element offset dstOff — the
// typed row-block copy of the KV append hot path (§base-perf: no per-element
// AtF64/SetF64 dispatch).
func copyRows(dst *tensor.Tensor, dstOff int, src *tensor.Tensor, nEl int) {
	s := src.Contiguous() // no-op for kernel outputs and rowBuf views (offset 0)
	switch dst.Dtype() {
	case tensor.F32:
		copy(dst.Storage().F32()[dstOff:dstOff+nEl], s.Storage().F32()[:nEl])
	case tensor.F64:
		copy(dst.Storage().F64()[dstOff:dstOff+nEl], s.Storage().F64()[:nEl])
	case tensor.F16, tensor.BF16:
		copy(dst.Storage().U16()[dstOff:dstOff+nEl], s.Storage().U16()[:nEl])
	}
}

// owns reports whether cur is the row-prefix view this buffer last handed out:
// same storage, offset 0, rank-2 [n, width], contiguous. Anything else means a
// caller replaced the cache entry and Append must resynchronize.
func (r *rowBuf) owns(cur *tensor.Tensor) bool {
	return r.buf != nil &&
		cur.Storage() == r.buf.Storage() &&
		cur.Offset() == 0 &&
		cur.Ndim() == 2 &&
		cur.Shape()[0] == r.n &&
		cur.Shape()[1] == r.buf.Shape()[1] &&
		cur.IsContiguous()
}

// kvBufs pairs the per-layer rowBufs backing a KV cache's public K, V views.
// Cache structs embed it unexported; the exported K[l], V[l] fields keep
// holding the current [t, width] views (value-identical to the old concatRows
// results), so external readers — Len(), tests, eviction helpers — see the
// same tensors as before while appends run in amortized O(1) per token.
type kvBufs struct {
	k, v []rowBuf
}

// kvAppendViaConcat forces appendKV back onto the old concatRows path. It is a
// BENCH-ONLY fallback (§V22): the e2e old-vs-new decode benchmark flips it so
// both sides run the identical DecodeStep code and differ only in the append.
// Never set outside a benchmark; benchmarks run sequentially so no lock.
var kvAppendViaConcat bool

// appendKV appends the token's k, v rows to layer l and returns the refreshed
// public views. The internal buffer slices grow on demand so caches built as
// struct literals (bypassing NewCache) work identically.
func (b *kvBufs) appendKV(K, V []*tensor.Tensor, l int, k, v *tensor.Tensor) (kNew, vNew *tensor.Tensor) {
	if kvAppendViaConcat {
		return concatRows(K[l], k), concatRows(V[l], v)
	}
	if len(b.k) <= l {
		b.k = growSlice(b.k, max(l+1, len(K)))
	}
	if len(b.v) <= l {
		b.v = growSlice(b.v, max(l+1, len(V)))
	}
	kNew = b.k[l].Append(K[l], k)
	vNew = b.v[l].Append(V[l], v)
	return kNew, vNew
}

// growSlice returns s extended with zero values to length n (state preserved);
// s unchanged if already long enough.
func growSlice[T any](s []T, n int) []T {
	if len(s) >= n {
		return s
	}
	ns := make([]T, n)
	copy(ns, s)
	return ns
}
