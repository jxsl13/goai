//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestPrefixShareCorrectAndSaves validates the RadixAttention/prefix-cache primitive: a sequence that
// SHARES another's prefix KV blocks (NewSeqSharingPrefix, refcounted) must produce IDENTICAL decode
// attention to the full independent sequence (the shared blocks ARE the same physical KV), while using
// FEWER physical blocks (the shared prefix is stored once). Also checks refcount safety: releasing one
// sharer must not free blocks still held by the other.
func TestPrefixShareCorrectAndSaves(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const blockSize = 16
	qHeads, kvHeads, hd := 8, 8, 64
	kvW := kvHeads * hd
	prefixTok, suffixTok := 32, 16 // 2 prefix blocks + 1 suffix block
	pool, err := cuda.NewPagedKVPool(64, blockSize, kvW)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	rng := rand.New(rand.NewSource(5))

	mkKV := func(ntok int) (*cuda.DeviceF32, *cuda.DeviceF32) {
		kf := make([]float32, ntok*kvW)
		vf := make([]float32, ntok*kvW)
		for i := range kf {
			kf[i] = float32(rng.NormFloat64() * 0.3)
			vf[i] = float32(rng.NormFloat64() * 0.3)
		}
		dk, _ := cuda.NewDeviceF32(ntok, kvW)
		dv, _ := cuda.NewDeviceF32(ntok, kvW)
		dk.UploadF32(kf)
		dv.UploadF32(vf)
		return dk, dv
	}

	// seqA: full independent sequence (prefix + suffix), own blocks — the reference.
	seqA := pool.NewSeqKV()
	pk, pv := mkKV(prefixTok)
	seqA.Append(pk, pk) // K==V here is fine; we only test share-equivalence
	sk, sv := mkKV(suffixTok)
	seqA.Append(sk, sk)
	usedAfterA := pool.NumBlocks() - pool.FreeBlocks()

	// seqB: shares seqA's 2 prefix blocks, then appends the SAME suffix KV → logically identical to seqA.
	seqB, err := pool.NewSeqSharingPrefix(seqA, prefixTok/blockSize)
	if err != nil {
		t.Fatal(err)
	}
	seqB.Append(sk, sk)
	usedAfterB := pool.NumBlocks() - pool.FreeBlocks()

	// Memory: seqB must have added only its 1 suffix block (shared the 2 prefix blocks).
	if got := usedAfterB - usedAfterA; got != suffixTok/blockSize {
		t.Fatalf("prefix sharing added %d blocks, want %d (should reuse the %d prefix blocks)", got, suffixTok/blockSize, prefixTok/blockSize)
	}

	// Correctness: decode A and B with the same query → identical output (same physical KV).
	qh := make([]float32, qHeads*hd)
	for i := range qh {
		qh[i] = float32(rng.NormFloat64() * 0.3)
	}
	decode := func(s *cuda.SeqKV) []float32 {
		q, _ := cuda.NewDeviceF32(1, qHeads*hd)
		q.UploadF32(qh)
		defer q.Free()
		view, _ := pool.UploadBatchView([]*cuda.SeqKV{s})
		defer view.Free()
		o, err := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
		if err != nil {
			t.Fatal(err)
		}
		defer o.Free()
		out := make([]float32, qHeads*hd)
		o.DownloadF32(out)
		_ = scale
		return out
	}
	oa, ob := decode(seqA), decode(seqB)
	var maxErr float64
	for i := range oa {
		maxErr = math.Max(maxErr, math.Abs(float64(oa[i]-ob[i])))
	}
	t.Logf("shared-prefix decode vs full-sequence decode: maxErr=%.3g; blocks used %d (vs %d independent) — saved %d",
		maxErr, usedAfterB, 2*(prefixTok+suffixTok)/blockSize, 2*(prefixTok/blockSize)-(prefixTok/blockSize))
	if maxErr != 0 {
		t.Fatalf("shared-prefix attention differs from full sequence (maxErr=%g) — sharing must be exact", maxErr)
	}

	// Refcount safety: releasing seqB must NOT free the shared prefix blocks (seqA still holds them).
	freeBefore := pool.FreeBlocks()
	seqB.Release()
	freedByB := pool.FreeBlocks() - freeBefore
	if freedByB != suffixTok/blockSize {
		t.Fatalf("releasing seqB freed %d blocks, want %d (shared prefix must survive until seqA releases)", freedByB, suffixTok/blockSize)
	}
	seqA.Release()
	if pool.FreeBlocks() != pool.NumBlocks() {
		t.Fatalf("after both released, %d/%d blocks free (leak)", pool.FreeBlocks(), pool.NumBlocks())
	}
	pk.Free()
	pv.Free()
	sk.Free()
	sv.Free()
}
