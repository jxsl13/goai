//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
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
	free      []int32 // free physical block ids
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
	if hd != 64 {
		return nil, fmt.Errorf("cuda: BatchedDecodeAttn requires hd==64 (got %d)", hd)
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
