//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestPrefixCacheAutoShares validates the automatic prefix cache: request 2, arriving with the SAME token
// prefix as request 1, has that prefix's KV blocks discovered by content hash (Match) and shared
// (NewSeqFromBlocks) — no recompute — and decodes BIT-IDENTICALLY to an independent sequence with its own
// copy of the identical prefix+suffix, while adding only its divergent suffix block.
func TestPrefixCacheAutoShares(t *testing.T) {
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
	cache := cuda.NewPrefixBlockCache(pool, 32)
	rng := rand.New(rand.NewSource(9))

	// Deterministic KV per token position (a stand-in for "same tokens -> same KV").
	mkKV := func(ntok int, seed int) *cuda.DeviceF32 {
		r := rand.New(rand.NewSource(int64(seed)))
		f := make([]float32, ntok*kvW)
		for i := range f {
			f[i] = float32(r.NormFloat64() * 0.3)
		}
		d, _ := cuda.NewDeviceF32(ntok, kvW)
		d.UploadF32(f)
		return d
	}
	prefixKV := mkKV(prefixTok, 1) // the shared prefix's KV (seed 1)
	suffix2KV := mkKV(suffixTok, 3)

	prefixToks := make([]int32, prefixTok)
	for i := range prefixToks {
		prefixToks[i] = int32(i + 1)
	}
	suffix1Toks := make([]int32, suffixTok)
	suffix2Toks := make([]int32, suffixTok)
	for i := range suffix1Toks {
		suffix1Toks[i] = int32(1000 + i)
		suffix2Toks[i] = int32(2000 + i)
	}

	// Request 1: build prefix+suffix1, insert its blocks under its tokens.
	seq1 := pool.NewSeqKV()
	seq1.Append(prefixKV, prefixKV)
	s1KV := mkKV(suffixTok, 2)
	seq1.Append(s1KV, s1KV)
	toks1 := append(append([]int32(nil), prefixToks...), suffix1Toks...)
	cache.Insert(toks1, seq1.BlockTable())
	usedAfter1 := pool.NumBlocks() - pool.FreeBlocks()

	// Request 2: SAME prefix tokens, different suffix. The cache must discover the 2 prefix blocks.
	toks2 := append(append([]int32(nil), prefixToks...), suffix2Toks...)
	shared := cache.Match(toks2)
	if len(shared) != prefixTok/blockSize {
		t.Fatalf("cache matched %d prefix blocks, want %d (auto-discovery failed)", len(shared), prefixTok/blockSize)
	}
	seq2 := pool.NewSeqFromBlocks(shared)
	seq2.Append(suffix2KV, suffix2KV)
	usedAfter2 := pool.NumBlocks() - pool.FreeBlocks()
	if got := usedAfter2 - usedAfter1; got != suffixTok/blockSize {
		t.Fatalf("request 2 added %d blocks, want %d (should reuse cached prefix)", got, suffixTok/blockSize)
	}

	// Reference: an independent sequence with its OWN copy of the same prefix KV + suffix2 KV.
	ref := pool.NewSeqKV()
	ref.Append(prefixKV, prefixKV)
	ref.Append(suffix2KV, suffix2KV)

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
		return out
	}
	got, want := decode(seq2), decode(ref)
	var maxErr float64
	for i := range got {
		maxErr = math.Max(maxErr, math.Abs(float64(got[i]-want[i])))
	}
	t.Logf("auto-shared request decode vs independent copy: maxErr=%.3g; cache hits=%d misses=%d; blocks: req2 added %d (shared %d prefix)",
		maxErr, cache.Hits, cache.Misses, usedAfter2-usedAfter1, len(shared))
	if maxErr != 0 {
		t.Fatalf("auto-shared attention differs from independent copy (maxErr=%g) — sharing must be exact", maxErr)
	}

	prefixKV.Free()
	suffix2KV.Free()
	s1KV.Free()
}
