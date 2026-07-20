//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestSeqKVAdvance validates the device-side eager append pattern for continuous batching:
// AppendBatchedDev writes K/V at slot=view.dsl (=SeqKV.n) but does NOT bump SeqKV.n; SeqKV.Advance(1)
// bumps it so a re-uploaded view reflects the new length. Verify the appended K/V lands at the right
// slot and Len advances — i.e. AppendBatchedDev+Advance == a correct device-side SeqKV.Append.
func TestSeqKVAdvance(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const kvW = 256
	pool, _ := cuda.NewPagedKVPool(4, 16, kvW)
	defer pool.Free()
	s := pool.NewSeqKV()
	defer s.Release()
	s.Reserve1() // block for slot 0
	// append two tokens device-side via AppendBatchedDev + Advance, distinct values per token.
	for tokn := 0; tokn < 2; tokn++ {
		dk, _ := cuda.NewDeviceF32(1, kvW)
		dv, _ := cuda.NewDeviceF32(1, kvW)
		kf := make([]float32, kvW)
		vf := make([]float32, kvW)
		for i := range kf {
			kf[i] = float32(tokn*1000 + i)
			vf[i] = float32(tokn*1000+i) + 0.5
		}
		dk.UploadF32(kf)
		dv.UploadF32(vf)
		view, _ := pool.UploadBatchView([]*cuda.SeqKV{s}) // dsl = s.n (=tokn)
		pool.AppendBatchedDev(dk, dv, view)               // writes slot=dsl=tokn
		view.Free()
		s.Advance(1) // host length now tokn+1
		dk.Free()
		dv.Free()
	}
	if s.Len() != 2 {
		t.Fatalf("Len %d != 2 after 2 Advance", s.Len())
	}
	gk, _ := s.GatherK()
	defer gk.Free()
	host := make([]float32, 2*kvW)
	gk.DownloadF32(host)
	for tokn := 0; tokn < 2; tokn++ {
		for i := 0; i < kvW; i++ {
			if got, want := host[tokn*kvW+i], float32(tokn*1000+i); got != want {
				t.Fatalf("token %d ch %d = %v want %v (AppendBatchedDev+Advance misplaced K/V)", tokn, i, got, want)
			}
		}
	}
	t.Logf("AppendBatchedDev + SeqKV.Advance: 2 tokens at correct slots, Len=2 — device-side eager append for continuous batching WORKS")
}
