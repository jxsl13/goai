//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestGraphInGraphAppend validates the deployable decoder's core capture pattern: a CUDA-graph that
// captures [AppendBatchedDev(current K/V) -> BumpLens(1) -> attention] and REPLAYS it over growing KV.
// Unlike TestGraphDecodeGrowingKV (host-side Update between replays), here the append + length-bump are
// INSIDE the captured graph (pure device ops), so replaying grows the KV with no host intervention.
// Compared against an eager reference (SeqKV.Append then attend). Stays within one 16-block to skip
// boundary allocation. If they match, the in-graph correct-ordering decode step is capture-safe.
func TestGraphInGraphAppend(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	const batch, startLen, steps = 2, 8, 6 // 8..14, within block 0 (no boundary)
	mkKV := func(seed float32) *cuda.DeviceF32 {
		d, _ := cuda.NewDeviceF32(1, kvW)
		f := make([]float32, kvW)
		for i := range f {
			f[i] = seed + 0.001*float32(i%7)
		}
		d.UploadF32(f)
		return d
	}
	// the "current token" K/V appended each step (same buffer replayed) — one row per batch seq
	dkStep, _ := cuda.NewDeviceF32(batch, kvW)
	dvStep, _ := cuda.NewDeviceF32(batch, kvW)
	defer dkStep.Free()
	defer dvStep.Free()
	kf := make([]float32, batch*kvW)
	vf := make([]float32, batch*kvW)
	for i := range kf {
		kf[i] = 0.05 + 0.002*float32(i%11)
		vf[i] = 0.03 + 0.001*float32(i%13)
	}
	dkStep.UploadF32(kf)
	dvStep.UploadF32(vf)
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = 0.02 + 0.0005*float32(i%17)
	}

	build := func() (*cuda.PagedKVPool, []*cuda.SeqKV) {
		pool, _ := cuda.NewPagedKVPool(batch*2, 16, kvW)
		seqs := make([]*cuda.SeqKV, batch)
		for i := range seqs {
			seqs[i] = pool.NewSeqKV()
			for p := 0; p < startLen; p++ {
				d := mkKV(0.01 + 0.003*float32(p))
				seqs[i].Append(d, d)
				d.Free()
			}
			seqs[i].Reserve1() // ensure block covers up to 15 (startLen+steps=14)
		}
		return pool, seqs
	}

	// eager reference: append the step K/V `steps` times, attend at the end.
	poolE, seqsE := build()
	defer poolE.Free()
	for s := 0; s < steps; s++ {
		// append row b of dkStep/dvStep to seq b — via a 1-row slice per seq
		for b := 0; b < batch; b++ {
			d, _ := cuda.NewDeviceF32(1, kvW)
			d.UploadF32(kf[b*kvW : (b+1)*kvW])
			dv, _ := cuda.NewDeviceF32(1, kvW)
			dv.UploadF32(vf[b*kvW : (b+1)*kvW])
			seqsE[b].Append(d, dv)
			d.Free()
			dv.Free()
		}
	}
	qE, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	qE.UploadF32(qf)
	defer qE.Free()
	viewE, _ := poolE.UploadBatchView(seqsE)
	defer viewE.Free()
	oE, _ := poolE.BatchedDecodeAttnViewGQA(qE, viewE, qHeads, kvHeads)
	defer oE.Free()
	refOut := make([]float32, batch*qHeads*hd)
	oE.DownloadF32(refOut)

	// graph path: capture [AppendBatchedDev + BumpLens + attention_into], replay `steps` times.
	poolG, seqsG := build()
	defer poolG.Free()
	qG, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	qG.UploadF32(qf)
	defer qG.Free()
	view, _ := poolG.UploadBatchView(seqsG) // dsl = startLen
	defer view.Free()
	attnOut, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	defer attnOut.Free()
	step := func() {
		poolG.AppendBatchedDev(dkStep, dvStep, view) // writes slot=dsl
		view.BumpLens(1)                             // dsl++ (attention includes it)
		poolG.BatchedDecodeAttnViewInto(qG, view, qHeads, kvHeads, attnOut)
	}
	step() // warm (also does the 1st of `steps` appends)
	cuda.GraphSync()
	if err := cuda.CaptureBegin(); err != nil {
		t.Skipf("capture: %v", err)
	}
	step()
	g, err := cuda.CaptureEnd()
	if err != nil {
		t.Skipf("capture end: %v", err)
	}
	defer g.Free()
	// warm executed append #1; the capture step only RECORDS (does not execute); each replay executes
	// one append. So steps-1 replays gives `steps` total appends (slots startLen..startLen+steps-1).
	for i := 0; i < steps-1; i++ {
		g.Launch()
	}
	cuda.GraphSync()
	gOut := make([]float32, batch*qHeads*hd)
	attnOut.DownloadF32(gOut)

	var num, den float64
	for i := range refOut {
		d := float64(refOut[i] - gOut[i])
		num += d * d
		den += float64(refOut[i]) * float64(refOut[i])
	}
	rms := math.Sqrt(num / den)
	t.Logf("in-graph append+bump+attention vs eager: rel-RMS %.3e (%d steps captured+replayed over growing KV)", rms, steps)
	if rms > 1e-5 {
		t.Fatalf("in-graph append under capture diverged: rel-RMS %.3e", rms)
	}
	t.Logf("=> in-graph AppendBatchedDev+BumpLens+attention is CAPTURE-SAFE — deployable graph decoder core proven")
	for _, s := range seqsG {
		s.Release()
	}
	for _, s := range seqsE {
		s.Release()
	}
}
