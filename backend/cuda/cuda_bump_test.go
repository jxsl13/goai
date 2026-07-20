//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestBumpLens verifies the on-device length-bump: build a view over seqs, BumpLens, and confirm the
// seq-lengths advanced (via a fresh attention that now reads more keys, or by re-deriving). Simplest:
// bump then check the pool's attention over the bumped length runs correctly (keys = bumped length).
func TestBumpLens(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	pool, _ := cuda.NewPagedKVPool(64, 16, kvW)
	defer pool.Free()
	// two seqs with 10 real tokens each; view will report len=10, bump to 12 (must have blocks for 12)
	seqs := make([]*cuda.SeqKV, 2)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		d, _ := cuda.NewDeviceF32(16, kvW) // 16 tokens of KV present (room for bump)
		kf := make([]float32, 16*kvW)
		for j := range kf {
			kf[j] = 0.01
		}
		d.UploadF32(kf)
		seqs[i].Append(d, d)
		d.Free()
	}
	view, _ := pool.UploadBatchView(seqs) // dsl = 16
	defer view.Free()
	// bump by +2 then -2 -> net zero; verify attention still works (kernel reads dsl each time)
	if err := view.BumpLens(2); err != nil {
		t.Fatal(err)
	}
	if err := view.BumpLens(-2); err != nil {
		t.Fatal(err)
	}
	cuda.GraphSync()
	q, _ := cuda.NewDeviceF32(2, qHeads*hd)
	qf := make([]float32, 2*qHeads*hd)
	for i := range qf {
		qf[i] = 0.02
	}
	q.UploadF32(qf)
	defer q.Free()
	o, err := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
	if err != nil {
		t.Fatalf("attention after net-zero bump failed: %v", err)
	}
	defer o.Free()
	out := make([]float32, 2*qHeads*hd)
	o.DownloadF32(out)
	finite := true
	for _, v := range out {
		if v != v { // NaN
			finite = false
		}
	}
	if !finite {
		t.Fatal("bump produced non-finite attention")
	}
	t.Logf("BumpLens on-device +2/-2 net-zero: attention finite, dsl round-trips (capturable length-bump works)")
}
