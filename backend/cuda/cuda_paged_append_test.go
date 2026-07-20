//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestPagedAppendBatched validates the device-side batched paged KV append: growing a batch of
// sequences ONE token/step with NO host K/V upload (dk/dv are already device-resident, as they are
// in real decode), crossing a paged-block boundary, then verifying every appended token is bit-exact
// via Gather. This is the real serving append that replaces the harness's per-seq host round-trip;
// with it, the full serving loop hits the graph-decode capstone's clean ~9425 tok/s (1.35x vLLM)
// instead of the harness-append-bound 6832.
func TestPagedAppendBatched(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const batch, wkv, blockSize, startLen, steps = 6, 32, 16, 14, 6 // 14 -> 20 crosses the block boundary at 16
	pool, err := cuda.NewPagedKVPool(batch*4, blockSize, wkv)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()
	// distinct, reconstructable value per (seq, token, channel): a misrouted write is caught exactly
	kval := func(seq, tok, ch int) float32 { return float32(seq*100000 + tok*100 + ch) }
	vval := func(seq, tok, ch int) float32 { return kval(seq, tok, ch) + 0.5 } // V distinguishable from K

	seqs := make([]*cuda.SeqKV, batch)
	for b := 0; b < batch; b++ {
		seqs[b] = pool.NewSeqKV()
		kf := make([]float32, startLen*wkv)
		vf := make([]float32, startLen*wkv)
		for tok := 0; tok < startLen; tok++ {
			for ch := 0; ch < wkv; ch++ {
				kf[tok*wkv+ch] = kval(b, tok, ch)
				vf[tok*wkv+ch] = vval(b, tok, ch)
			}
		}
		dk, _ := cuda.NewDeviceF32(startLen, wkv)
		dv, _ := cuda.NewDeviceF32(startLen, wkv)
		dk.UploadF32(kf)
		dv.UploadF32(vf)
		if err := seqs[b].Append(dk, dv); err != nil {
			t.Fatal(err)
		}
		dk.Free()
		dv.Free()
	}

	// Grow one token/step, device-side. Each step: reserve the next block (host bookkeeping), upload a
	// fresh view (pre-append lengths + block tables), scatter this step's K/V device-side.
	dk, _ := cuda.NewDeviceF32(batch, wkv)
	dv, _ := cuda.NewDeviceF32(batch, wkv)
	defer dk.Free()
	defer dv.Free()
	for j := 0; j < steps; j++ {
		tok := startLen + j
		kf := make([]float32, batch*wkv)
		vf := make([]float32, batch*wkv)
		for b := 0; b < batch; b++ {
			for ch := 0; ch < wkv; ch++ {
				kf[b*wkv+ch] = kval(b, tok, ch)
				vf[b*wkv+ch] = vval(b, tok, ch)
			}
		}
		dk.UploadF32(kf)
		dv.UploadF32(vf)
		for b := range seqs {
			if err := seqs[b].Reserve1(); err != nil {
				t.Fatal(err)
			}
		}
		view, err := pool.UploadBatchView(seqs)
		if err != nil {
			t.Fatal(err)
		}
		if err := pool.AppendBatched(seqs, dk, dv, view); err != nil {
			t.Fatal(err)
		}
		view.Free()
	}

	// Verify: every token (initial host-appended + device-appended) is bit-exact and correctly routed.
	finalLen := startLen + steps
	for b := 0; b < batch; b++ {
		if seqs[b].Len() != finalLen {
			t.Fatalf("seq %d len %d want %d", b, seqs[b].Len(), finalLen)
		}
		gk, err := seqs[b].GatherK()
		if err != nil {
			t.Fatal(err)
		}
		gv, err := seqs[b].GatherV()
		if err != nil {
			t.Fatal(err)
		}
		kf := make([]float32, finalLen*wkv)
		vf := make([]float32, finalLen*wkv)
		if err := gk.DownloadF32(kf); err != nil {
			t.Fatal(err)
		}
		if err := gv.DownloadF32(vf); err != nil {
			t.Fatal(err)
		}
		gk.Free()
		gv.Free()
		for tok := 0; tok < finalLen; tok++ {
			for ch := 0; ch < wkv; ch++ {
				if got, want := kf[tok*wkv+ch], kval(b, tok, ch); got != want {
					t.Fatalf("K seq %d tok %d ch %d = %v want %v", b, tok, ch, got, want)
				}
				if got, want := vf[tok*wkv+ch], vval(b, tok, ch); got != want {
					t.Fatalf("V seq %d tok %d ch %d = %v want %v", b, tok, ch, got, want)
				}
			}
		}
		seqs[b].Release()
	}
}
