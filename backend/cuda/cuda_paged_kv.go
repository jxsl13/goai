//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// §ROADMAP FRONT B / B1 — paged KV cache (the PagedAttention SOSP'23 storage design), the
// foundation for continuous batching. Instead of one contiguous [maxSeq,wkv] buffer per
// sequence (the current KVCache — forces a full maxSeq reservation, fragments, caps concurrency),
// a shared device POOL of fixed-size blocks is carved among many sequences via per-sequence
// BLOCK TABLES (logical block -> physical block id). Near-zero fragmentation; O(1) admit/evict.
// This slice is the storage layer + a gather-to-contiguous bridge (so the existing attention
// kernel still runs); B2 replaces the gather with a paged/ragged attention kernel that reads
// blocks directly.

// PagedKVPool is a device block pool of `numBlocks` blocks, each holding blockSize tokens of
// width wkv (= kvHeads·headDim), for K and V. Blocks are handed out via a host-side free list.
type PagedKVPool struct {
	k, v      *DeviceF32 // [numBlocks*blockSize, wkv] f32
	blockSize int
	wkv       int
	numBlocks int
	free      []int32        // free physical block ids
	kf16      unsafe.Pointer // lazy f16 (u16) shadow of k for the f16 decode path (nil until built)
	vf16      unsafe.Pointer
}

// NewPagedKVPool allocates a pool of numBlocks × blockSize × wkv for K and V.
func NewPagedKVPool(numBlocks, blockSize, wkv int) (*PagedKVPool, error) {
	if numBlocks <= 0 || blockSize <= 0 || wkv <= 0 {
		return nil, fmt.Errorf("cuda: PagedKVPool bad dims blocks=%d blockSize=%d wkv=%d", numBlocks, blockSize, wkv)
	}
	rows := numBlocks * blockSize
	k, err := NewDeviceF32(rows, wkv)
	if err != nil {
		return nil, err
	}
	v, err := NewDeviceF32(rows, wkv)
	if err != nil {
		k.Free()
		return nil, err
	}
	free := make([]int32, numBlocks)
	for i := range free {
		free[i] = int32(i)
	}
	return &PagedKVPool{k: k, v: v, blockSize: blockSize, wkv: wkv, numBlocks: numBlocks, free: free}, nil
}

// FreeBlocks reports how many physical blocks are unallocated.
func (p *PagedKVPool) FreeBlocks() int { return len(p.free) }

func (p *PagedKVPool) alloc() (int32, error) {
	if len(p.free) == 0 {
		return -1, fmt.Errorf("cuda: PagedKVPool out of blocks (%d total)", p.numBlocks)
	}
	b := p.free[len(p.free)-1]
	p.free = p.free[:len(p.free)-1]
	return b, nil
}

func (p *PagedKVPool) release(b int32) { p.free = append(p.free, b) }

// Free releases the pool's device memory.
func (p *PagedKVPool) Free() {
	if p.k != nil {
		p.k.Free()
		p.v.Free()
		p.k, p.v = nil, nil
	}
	if p.kf16 != nil {
		C.cu_free_f32(p.kf16)
		C.cu_free_f32(p.vf16)
		p.kf16, p.vf16 = nil, nil
	}
}

// ensureF16 lazily builds an f16 (u16) shadow of the whole K/V pool for the f16 decode path.
// NOTE: this is a read-optimized shadow that assumes the pool is STATIC after it is built — it is
// not refreshed on Append (a measurement/bench path today; full f16 storage is the productization).
func (p *PagedKVPool) ensureF16() error {
	if p.kf16 != nil {
		return nil
	}
	n := p.numBlocks * p.blockSize * p.wkv
	kf16 := C.cu_alloc_u16(C.int(n))
	vf16 := C.cu_alloc_u16(C.int(n))
	if kf16 == nil || vf16 == nil {
		freeIf(kf16)
		freeIf(vf16)
		return fmt.Errorf("cuda: PagedKVPool f16 shadow alloc failed")
	}
	if rc := C.cu_cvt_f32_to_f16(kf16, p.k.ptr, C.long(n)); rc != 0 {
		C.cu_free_f32(kf16)
		C.cu_free_f32(vf16)
		return fmt.Errorf("cuda: PagedKVPool f16 convert K failed (%d)", int(rc))
	}
	if rc := C.cu_cvt_f32_to_f16(vf16, p.v.ptr, C.long(n)); rc != 0 {
		C.cu_free_f32(kf16)
		C.cu_free_f32(vf16)
		return fmt.Errorf("cuda: PagedKVPool f16 convert V failed (%d)", int(rc))
	}
	p.kf16, p.vf16 = unsafe.Pointer(kf16), unsafe.Pointer(vf16)
	return nil
}

// SeqKV is one sequence's paged view into a pool: a block table (logical block -> physical id)
// plus the logical length. Append grows it a block at a time; Gather materializes it contiguous.
type SeqKV struct {
	pool  *PagedKVPool
	table []int32 // logical block index -> physical block id
	n     int     // logical token count
}

// NewSeqKV starts an empty sequence on the pool.
func (p *PagedKVPool) NewSeqKV() *SeqKV { return &SeqKV{pool: p} }

// Len returns the sequence's token count.
func (s *SeqKV) Len() int { return s.n }

// Advance bumps the sequence's logical length by delta WITHOUT copying K/V — the host-side counterpart
// to a device-side AppendBatchedDev (which writes the K/V but does not touch SeqKV.n). Pair them for a
// continuous-batch decode with a DYNAMIC active set, where the view is re-uploaded each step from
// SeqKV.n (so n must track the device length): view(dsl=n) -> AppendBatchedDev(slot=n) -> Advance(1)
// -> view(dsl=n+1) -> attention. Caller must ensure blocks exist for the advanced positions (Reserve1).
func (s *SeqKV) Advance(delta int) { s.n += delta }

// Append writes ntok=k.rows tokens (k,v are [ntok,wkv] device buffers) at the current logical
// end, allocating physical blocks as logical block boundaries are crossed. K and V are appended
// in lockstep.
func (s *SeqKV) Append(k, v *DeviceF32) error {
	p := s.pool
	if k.cols != p.wkv || v.cols != p.wkv || k.rows != v.rows {
		return fmt.Errorf("cuda: SeqKV.Append shape mismatch k[%d,%d] v[%d,%d] wkv=%d", k.rows, k.cols, v.rows, v.cols, p.wkv)
	}
	ntok := k.rows
	for pos := s.n; pos < s.n+ntok; {
		lb := pos / p.blockSize
		slot := pos % p.blockSize
		for lb >= len(s.table) {
			b, err := p.alloc()
			if err != nil {
				return err
			}
			s.table = append(s.table, b)
		}
		phys := int(s.table[lb])
		cnt := p.blockSize - slot
		if rem := s.n + ntok - pos; rem < cnt {
			cnt = rem
		}
		dstOff := (phys*p.blockSize + slot) * p.wkv
		srcOff := (pos - s.n) * p.wkv
		if err := p.k.Copy2D(dstOff, p.wkv, k, srcOff, p.wkv, cnt, p.wkv); err != nil {
			return err
		}
		if err := p.v.Copy2D(dstOff, p.wkv, v, srcOff, p.wkv, cnt, p.wkv); err != nil {
			return err
		}
		pos += cnt
	}
	s.n += ntok
	return nil
}

// Reserve1 ensures a physical block exists for this sequence's NEXT token, so a device-side append
// can write it. Block allocation is the infrequent host-side bookkeeping paged serving does out of
// band (once per blockSize tokens); the per-step data write stays on the device (see AppendBatched).
func (s *SeqKV) Reserve1() error {
	lb := s.n / s.pool.blockSize
	for lb >= len(s.table) {
		b, err := s.pool.alloc()
		if err != nil {
			return err
		}
		s.table = append(s.table, b)
	}
	return nil
}

// AppendBatched appends ONE token per sequence device-side: dk/dv are [batch,wkv] device buffers (the
// decode step's own freshly-computed K/V), scattered into each sequence's next KV slot by a single
// kernel — no host round-trip. This is the real serving append; SeqKV.Append (host uploads per call)
// is the harness path. `view` must reflect the sequences' CURRENT (pre-append) block tables + lengths,
// and each sequence must have its next block reserved (call Reserve1 then view.Update first). On
// success each sequence's logical length is bumped by one (host-side metadata only — no data copy).
func (p *PagedKVPool) AppendBatched(seqs []*SeqKV, dk, dv *DeviceF32, view *PagedBatchView) error {
	batch := len(seqs)
	if dk.rows != batch || dv.rows != batch || dk.cols != p.wkv || dv.cols != p.wkv {
		return fmt.Errorf("cuda: AppendBatched shape dk[%d,%d] dv[%d,%d] want [%d,%d]", dk.rows, dk.cols, dv.rows, dv.cols, batch, p.wkv)
	}
	if view.batch != batch {
		return fmt.Errorf("cuda: AppendBatched view batch %d != %d", view.batch, batch)
	}
	if rc := C.cu_paged_append_batched(p.k.ptr, p.v.ptr, view.dbt, view.dsl, dk.ptr, dv.ptr,
		C.int(batch), C.int(p.wkv), C.int(p.blockSize), C.int(view.maxBlocks)); rc != 0 {
		return fmt.Errorf("cuda: AppendBatched rc=%d", int(rc))
	}
	for _, s := range seqs {
		s.n++
	}
	return nil
}

// AppendBatchedDev is AppendBatched WITHOUT the host-side SeqKV.n bump — the pure-device (capturable)
// append for GRAPH decode. In a captured graph the append KERNEL replays each step but the Go host
// code runs only at capture time, so the length must advance device-side (BumpLens, in-graph) and
// the host SeqKV.n manually (per replay, so Reserve1 knows where to allocate on block boundaries).
// Writes each row of dk/dv to slot=view.dsl[b] of its sequence — view.dsl must hold the pre-append
// length (=pos); pair with view.BumpLens(1) so the captured attention reads pos+1 keys.
func (p *PagedKVPool) AppendBatchedDev(dk, dv *DeviceF32, view *PagedBatchView) error {
	if dk.rows != view.batch || dv.rows != view.batch || dk.cols != p.wkv || dv.cols != p.wkv {
		return fmt.Errorf("cuda: AppendBatchedDev shape mismatch")
	}
	if rc := C.cu_paged_append_batched(p.k.ptr, p.v.ptr, view.dbt, view.dsl, dk.ptr, dv.ptr,
		C.int(view.batch), C.int(p.wkv), C.int(p.blockSize), C.int(view.maxBlocks)); rc != 0 {
		return fmt.Errorf("cuda: AppendBatchedDev rc=%d", int(rc))
	}
	return nil
}

// GatherK / GatherV materialize the sequence's K / V as fresh contiguous [n,wkv] buffers by
// copying the (possibly non-contiguous) physical blocks in logical order — the bridge that lets
// the existing contiguous attention kernel consume paged storage until B2 lands.
func (s *SeqKV) GatherK() (*DeviceF32, error) { return s.gather(s.pool.k) }
func (s *SeqKV) GatherV() (*DeviceF32, error) { return s.gather(s.pool.v) }

func (s *SeqKV) gather(src *DeviceF32) (*DeviceF32, error) {
	p := s.pool
	out, err := NewDeviceF32(s.n, p.wkv)
	if err != nil {
		return nil, err
	}
	for pos := 0; pos < s.n; {
		lb := pos / p.blockSize
		slot := pos % p.blockSize
		phys := int(s.table[lb])
		cnt := p.blockSize - slot
		if rem := s.n - pos; rem < cnt {
			cnt = rem
		}
		srcOff := (phys*p.blockSize + slot) * p.wkv
		if err := out.Copy2D(pos*p.wkv, p.wkv, src, srcOff, p.wkv, cnt, p.wkv); err != nil {
			out.Free()
			return nil, err
		}
		pos += cnt
	}
	return out, nil
}

// BlockTable returns a copy of the logical->physical block map (for the paged attention kernel).
func (s *SeqKV) BlockTable() []int32 {
	t := make([]int32, len(s.table))
	copy(t, s.table)
	return t
}

// Release returns the sequence's blocks to the pool (O(#blocks), no copies) — the eviction
// path continuous batching needs when a request finishes.
func (s *SeqKV) Release() {
	for _, b := range s.table {
		s.pool.release(b)
	}
	s.table = nil
	s.n = 0
}

// PagedBatchView is a device-resident snapshot of a batch's block tables + sequence lengths,
// uploaded ONCE per decode step and reused across all layers. BatchedDecodeAttn re-uploaded (and
// device-synced) these every call — a 22-layer step paid 22 host↔device round-trips just to ship
// the same integers. Building the view once and calling BatchedDecodeAttnView removes 21 of them.
type PagedBatchView struct {
	pool      *PagedKVPool
	dbt, dsl  unsafe.Pointer
	batch     int
	maxBlocks int
}

// UploadBatchView packs and uploads (async, no device sync) the block tables + seq lengths for
// `seqs`. Valid while the sequences' block tables and lengths are unchanged (a single decode step,
// before Append). Call Free() when the step is done.
func (p *PagedKVPool) UploadBatchView(seqs []*SeqKV) (*PagedBatchView, error) {
	batch := len(seqs)
	if batch == 0 {
		return nil, fmt.Errorf("cuda: UploadBatchView empty batch")
	}
	maxBlocks := 0
	for _, s := range seqs {
		if s.pool != p {
			return nil, fmt.Errorf("cuda: UploadBatchView sequence not from this pool")
		}
		if len(s.table) > maxBlocks {
			maxBlocks = len(s.table)
		}
	}
	if maxBlocks == 0 {
		return nil, fmt.Errorf("cuda: UploadBatchView all sequences empty")
	}
	bt := make([]int32, batch*maxBlocks)
	sl := make([]int32, batch)
	for i, s := range seqs {
		copy(bt[i*maxBlocks:], s.table)
		sl[i] = int32(s.n)
	}
	dbt := C.cu_upload_i32_async((*C.int)(&bt[0]), C.int(len(bt)))
	dsl := C.cu_upload_i32_async((*C.int)(&sl[0]), C.int(len(sl)))
	if dbt == nil || dsl == nil {
		freeIf(dbt)
		freeIf(dsl)
		return nil, fmt.Errorf("cuda: UploadBatchView device alloc failed")
	}
	return &PagedBatchView{pool: p, dbt: unsafe.Pointer(dbt), dsl: unsafe.Pointer(dsl), batch: batch, maxBlocks: maxBlocks}, nil
}

// Update re-packs the block tables + seq lengths for `seqs` into the view's EXISTING device
// buffers in place (no alloc) — for a fixed-buffer graph-decode: capture the decode step reading
// this view, then between launches append K/V and Update(), replay. Requires the sequences still
// fit the view's batch and maxBlocks (allocate the view for the max sequence length up front).
func (v *PagedBatchView) Update(seqs []*SeqKV) error {
	if len(seqs) != v.batch {
		return fmt.Errorf("cuda: PagedBatchView.Update batch %d != view %d", len(seqs), v.batch)
	}
	mb := 0
	for _, s := range seqs {
		if s.pool != v.pool {
			return fmt.Errorf("cuda: PagedBatchView.Update sequence not from this pool")
		}
		if len(s.table) > mb {
			mb = len(s.table)
		}
	}
	if mb > v.maxBlocks {
		return fmt.Errorf("cuda: PagedBatchView.Update needs %d blocks > view capacity %d — pre-size the view", mb, v.maxBlocks)
	}
	bt := make([]int32, v.batch*v.maxBlocks) // padded to the view's fixed stride
	sl := make([]int32, v.batch)
	for i, s := range seqs {
		copy(bt[i*v.maxBlocks:], s.table)
		sl[i] = int32(s.n)
	}
	if rc := C.cu_update_i32(v.dbt, (*C.int)(&bt[0]), C.int(len(bt))); rc != 0 {
		return fmt.Errorf("cuda: PagedBatchView.Update block-table update failed (%d)", int(rc))
	}
	if rc := C.cu_update_i32(v.dsl, (*C.int)(&sl[0]), C.int(len(sl))); rc != 0 {
		return fmt.Errorf("cuda: PagedBatchView.Update seq-len update failed (%d)", int(rc))
	}
	return nil
}

// UpdateLens re-uploads ONLY the sequence lengths (batch int32) — the cheap per-step refresh for a
// steady-state decode where no block boundary was crossed (block tables unchanged). Real paged
// serving allocates a block only every blockSize tokens, so the full block-table rebuild in Update
// is amortized ~1/blockSize; every other step needs just this small length bump.
func (v *PagedBatchView) UpdateLens(seqs []*SeqKV) error {
	if len(seqs) != v.batch {
		return fmt.Errorf("cuda: PagedBatchView.UpdateLens batch %d != view %d", len(seqs), v.batch)
	}
	sl := make([]int32, v.batch)
	for i, s := range seqs {
		sl[i] = int32(s.n)
	}
	if rc := C.cu_update_i32(v.dsl, (*C.int)(&sl[0]), C.int(len(sl))); rc != 0 {
		return fmt.Errorf("cuda: PagedBatchView.UpdateLens failed (%d)", int(rc))
	}
	return nil
}

// BumpLens increments the view's device-resident seq-lengths by delta ON-DEVICE (capturable, no host
// sync) — the enabler for a CORRECT in-graph decode: append writes slot=seqLen, BumpLens(1) advances
// it, then the captured attention reads seqLen+1 keys (including the just-appended current token). The
// off-by-one path (attend before append) is catastrophically wrong (0/24 tokens, see ordering test).
func (v *PagedBatchView) BumpLens(delta int) error {
	if rc := C.cu_bump_i32(v.dsl, C.int(v.batch), C.int(delta)); rc != 0 {
		return fmt.Errorf("cuda: BumpLens failed (%d)", int(rc))
	}
	return nil
}

// Free releases the view's device buffers.
func (v *PagedBatchView) Free() {
	if v.dbt != nil {
		C.cu_free_f32(v.dbt)
		v.dbt = nil
	}
	if v.dsl != nil {
		C.cu_free_f32(v.dsl)
		v.dsl = nil
	}
}

// BatchedDecodeAttnView is BatchedDecodeAttn using a pre-uploaded PagedBatchView (no per-call
// block-table upload/sync). q is [batch, qHeads·hd]; returns o[batch, qHeads·hd].
func (p *PagedKVPool) BatchedDecodeAttnView(q *DeviceF32, view *PagedBatchView, qHeads, kvHeads int) (*DeviceF32, error) {
	if view.pool != p {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnView view not from this pool")
	}
	if q.rows != view.batch {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnView q rows %d != batch %d", q.rows, view.batch)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnView bad head config q=%d kv=%d width=%d", qHeads, kvHeads, q.cols)
	}
	hd := q.cols / qHeads
	if hd != 64 && hd != 128 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnView requires hd==64 or 128 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnView kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	out := C.cu_alloc_f32(C.int(view.batch * q.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnView device alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_paged_decode_attn(q.ptr, p.k.ptr, p.v.ptr, view.dbt, view.dsl, out,
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnView failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: view.batch, cols: q.cols}, nil
}

// BatchedDecodeAttnViewInto writes the decode attention into a caller-provided output buffer (no
// alloc) — the fixed-buffer form a CUDA-graph capture needs. Same math as BatchedDecodeAttnView.
func (p *PagedKVPool) BatchedDecodeAttnViewInto(q *DeviceF32, view *PagedBatchView, qHeads, kvHeads int, out *DeviceF32) error {
	if view.pool != p || q.rows != view.batch || out.rows != view.batch || out.cols != q.cols {
		return fmt.Errorf("cuda: BatchedDecodeAttnViewInto shape/pool mismatch")
	}
	hd := q.cols / qHeads
	if hd != 64 || kvHeads*hd != p.wkv {
		return fmt.Errorf("cuda: BatchedDecodeAttnViewInto bad config")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_paged_decode_attn_gqa(q.ptr, p.k.ptr, p.v.ptr, view.dbt, view.dsl, out.ptr,
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale))
	if rc != 0 {
		return fmt.Errorf("cuda: BatchedDecodeAttnViewInto failed (code %d)", int(rc))
	}
	return nil
}

// BatchedDecodeAttnViewGQA is BatchedDecodeAttnView using the GQA K/V-shared kernel (one block per
// (kv head, sequence), staging each K/V tile into shared memory once and serving all group query
// heads) — cuts the naive kernel's group× redundant K/V traffic. Requires hd==64, group≤8,
// blockSize≤16 (falls back is the caller's choice); otherwise returns an error.
func (p *PagedKVPool) BatchedDecodeAttnViewGQA(q *DeviceF32, view *PagedBatchView, qHeads, kvHeads int) (*DeviceF32, error) {
	if view.pool != p {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQA view not from this pool")
	}
	if q.rows != view.batch {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQA q rows %d != batch %d", q.rows, view.batch)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQA bad head config q=%d kv=%d width=%d", qHeads, kvHeads, q.cols)
	}
	hd := q.cols / qHeads
	if hd != 64 && hd != 128 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQA requires hd==64 or 128 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQA kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	out := C.cu_alloc_f32(C.int(view.batch * q.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQA device alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_paged_decode_attn_gqa(q.ptr, p.k.ptr, p.v.ptr, view.dbt, view.dsl, out,
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQA failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: view.batch, cols: q.cols}, nil
}

// BatchedDecodeAttnViewGQAQio is BatchedDecodeAttnViewGQA with f16 query IN and f16 output OUT (raw
// u16 pointers) over the SAME f32 K/V pool — for the A1 serving decode, which has its query in f16
// and immediately wants the result in f16, so this kills the two per-layer f32<->f16 conversions
// around attention with NO accuracy change (K/V precision unchanged). qCols = qHeads*hd. Caller owns
// the returned DeviceF16 (Free()). q is the f16 [batch,qCols] query.
func (p *PagedKVPool) BatchedDecodeAttnViewGQAQio(q *DeviceF16, view *PagedBatchView, qHeads, kvHeads int) (*DeviceF16, error) {
	if view.pool != p {
		return nil, fmt.Errorf("cuda: GQAQio view not from this pool")
	}
	if q.rows != view.batch {
		return nil, fmt.Errorf("cuda: GQAQio q rows %d != batch %d", q.rows, view.batch)
	}
	qCols := q.cols
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || qCols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: GQAQio bad head config q=%d kv=%d width=%d", qHeads, kvHeads, qCols)
	}
	hd := qCols / qHeads
	if hd != 64 {
		return nil, fmt.Errorf("cuda: GQAQio requires hd==64 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: GQAQio kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	out := C.cu_alloc_u16(C.int(view.batch * qCols))
	if out == nil {
		return nil, fmt.Errorf("cuda: GQAQio device alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_paged_decode_attn_gqa_qio(q.ptr, p.k.ptr, p.v.ptr, view.dbt, view.dsl, unsafe.Pointer(out),
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(unsafe.Pointer(out))
		return nil, fmt.Errorf("cuda: GQAQio failed (code %d)", int(rc))
	}
	return &DeviceF16{ptr: unsafe.Pointer(out), rows: view.batch, cols: qCols}, nil
}

// BatchedDecodeAttnViewGQAf16 is BatchedDecodeAttnViewGQA reading an f16 (u16) shadow of the pool
// K/V — half the global bytes, the lever once the GQA kernel is memory-bound. Builds the shadow
// lazily (see ensureF16: assumes a static pool after build). Requires hd==64, group≤8, blockSize≤16.
func (p *PagedKVPool) BatchedDecodeAttnViewGQAf16(q *DeviceF32, view *PagedBatchView, qHeads, kvHeads int) (*DeviceF32, error) {
	if view.pool != p {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAf16 view not from this pool")
	}
	if q.rows != view.batch {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAf16 q rows %d != batch %d", q.rows, view.batch)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAf16 bad head config q=%d kv=%d width=%d", qHeads, kvHeads, q.cols)
	}
	hd := q.cols / qHeads
	if hd != 64 && hd != 128 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAf16 requires hd==64 or 128 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAf16 kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	if err := p.ensureF16(); err != nil {
		return nil, err
	}
	out := C.cu_alloc_f32(C.int(view.batch * q.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAf16 device alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_paged_decode_attn_gqa_f16(q.ptr, p.kf16, p.vf16, view.dbt, view.dsl, out,
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewGQAf16 failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: view.batch, cols: q.cols}, nil
}

// BatchedDecodeAttnViewGQAf16Qio is BatchedDecodeAttnViewGQAf16 with f16 query IN and f16 output OUT
// (raw u16 pointers) — for the A1 decode, which already has its query in f16 and immediately wants
// the result in f16, so this kills the two per-layer f32<->f16 conversions around attention. qCols =
// qHeads*hd. Caller owns the returned f16 buffer (FreeDev). Bit-identical to the f32-IO path + converts.
func (p *PagedKVPool) BatchedDecodeAttnViewGQAf16Qio(q16 unsafe.Pointer, qCols int, view *PagedBatchView, qHeads, kvHeads int) (unsafe.Pointer, error) {
	if view.pool != p {
		return nil, fmt.Errorf("cuda: GQAf16Qio view not from this pool")
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || qCols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: GQAf16Qio bad head config q=%d kv=%d width=%d", qHeads, kvHeads, qCols)
	}
	hd := qCols / qHeads
	if hd != 64 {
		return nil, fmt.Errorf("cuda: GQAf16Qio requires hd==64 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: GQAf16Qio kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	if err := p.ensureF16(); err != nil {
		return nil, err
	}
	out := C.cu_alloc_u16(C.int(view.batch * qCols))
	if out == nil {
		return nil, fmt.Errorf("cuda: GQAf16Qio device alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_paged_decode_attn_gqa_f16_qio(q16, p.kf16, p.vf16, view.dbt, view.dsl, unsafe.Pointer(out),
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(unsafe.Pointer(out))
		return nil, fmt.Errorf("cuda: GQAf16Qio failed (code %d)", int(rc))
	}
	return unsafe.Pointer(out), nil
}

// BatchedDecodeAttnViewSK is BatchedDecodeAttnView with split-K (FlashDecoding): splitK blocks per
// (kv head, sequence) scan disjoint key chunks into partials, then a merge combines them — parallelizes
// the online-softmax scan (attention is ~34% of the A1 step, latency-bound). splitK 1..32.
func (p *PagedKVPool) BatchedDecodeAttnViewSK(q *DeviceF32, view *PagedBatchView, qHeads, kvHeads, splitK int) (*DeviceF32, error) {
	if view.pool != p {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewSK view not from this pool")
	}
	if q.rows != view.batch {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewSK q rows %d != batch %d", q.rows, view.batch)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewSK bad head config")
	}
	hd := q.cols / qHeads
	if hd != 64 && hd != 128 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewSK requires hd==64 or 128 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewSK kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	out := C.cu_alloc_f32(C.int(view.batch * q.cols))
	if out == nil {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewSK device alloc failed")
	}
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_paged_decode_attn_gqa_sk(q.ptr, p.k.ptr, p.v.ptr, view.dbt, view.dsl, out,
		C.int(view.batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(view.maxBlocks), C.float(scale), C.int(splitK))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: BatchedDecodeAttnViewSK failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: view.batch, cols: q.cols}, nil
}

// BatchedDecodeAttn runs single-query decode attention for a batch of sequences over this pool:
// q is [batch, qHeads·hd] (one query token per sequence, row i for seqs[i]), returning
// o[batch, qHeads·hd]. Each sequence attends over its own paged K/V via its block table — the
// batched, paged decode kernel (FRONT B / B2) that makes continuous batching possible. hd==64.
func (p *PagedKVPool) BatchedDecodeAttn(q *DeviceF32, seqs []*SeqKV, qHeads, kvHeads int) (*DeviceF32, error) {
	batch := len(seqs)
	if batch == 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn empty batch")
	}
	if q.rows != batch {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn q rows %d != batch %d", q.rows, batch)
	}
	if qHeads <= 0 || kvHeads <= 0 || qHeads%kvHeads != 0 || q.cols%qHeads != 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn bad head config q=%d kv=%d width=%d", qHeads, kvHeads, q.cols)
	}
	hd := q.cols / qHeads
	if hd != 64 && hd != 128 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn requires hd==64 or 128 (got %d)", hd)
	}
	if kvHeads*hd != p.wkv {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn kvHeads*hd=%d != pool wkv=%d", kvHeads*hd, p.wkv)
	}
	maxBlocks := 0
	for _, s := range seqs {
		if s.pool != p {
			return nil, fmt.Errorf("cuda: BatchedDecodeAttn sequence not from this pool")
		}
		if len(s.table) > maxBlocks {
			maxBlocks = len(s.table)
		}
	}
	if maxBlocks == 0 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn all sequences empty")
	}
	bt := make([]int32, batch*maxBlocks)
	sl := make([]int32, batch)
	for i, s := range seqs {
		copy(bt[i*maxBlocks:], s.table)
		sl[i] = int32(s.n)
	}
	dbt := C.cu_upload_i32((*C.int)(&bt[0]), C.int(len(bt)))
	dsl := C.cu_upload_i32((*C.int)(&sl[0]), C.int(len(sl)))
	out := C.cu_alloc_f32(C.int(batch * q.cols))
	if dbt == nil || dsl == nil || out == nil {
		freeIf(dbt)
		freeIf(dsl)
		freeIf(out)
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn device alloc failed")
	}
	defer C.cu_free_f32(dbt)
	defer C.cu_free_f32(dsl)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rc := C.cu_paged_decode_attn(q.ptr, p.k.ptr, p.v.ptr, dbt, dsl, out,
		C.int(batch), C.int(qHeads), C.int(kvHeads), C.int(hd), C.int(p.blockSize), C.int(maxBlocks), C.float(scale))
	if rc != 0 {
		C.cu_free_f32(out)
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn failed (code %d)", int(rc))
	}
	return &DeviceF32{ptr: out, rows: batch, cols: q.cols}, nil
}
