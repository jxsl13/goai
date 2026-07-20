//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// Isolate the paged decode attention kernel's wall-clock share of the batched decode step, to
// measure-first whether the GQA-redundant K/V reads (one warp per (seq,q-head) → each KV head
// read `group`× by its query heads) actually cost time or are absorbed by L2. Config mirrors the
// batched-forward bench: qHeads=32, kvHeads=4 (group 8), hd=64, seqLen=128.
func benchPagedDecodeAttn(b *testing.B, batch, seqLen int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(7))
	pool, err := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, seqLen*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(seqLen, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		b.Fatal(err)
	}
	defer view.Free()
	// warm
	o, err := pool.BatchedDecodeAttnView(q, view, qHeads, kvHeads)
	if err != nil {
		b.Fatal(err)
	}
	o.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		o, _ := pool.BatchedDecodeAttnView(q, view, qHeads, kvHeads)
		o.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
	// effective K/V bytes moved if reads were non-redundant (each kv element once per q that needs it)
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "attn/s")
}

func BenchmarkPagedDecodeAttn_b512(b *testing.B) { benchPagedDecodeAttn(b, 512, 128) }
func BenchmarkPagedDecodeAttn_b256(b *testing.B) { benchPagedDecodeAttn(b, 256, 128) }

// TestPagedDecodeGQAParity: the GQA K/V-shared kernel must match the naive per-head kernel
// bit-closely (both f32 online-softmax over the same paged K/V). Ragged batch, GQA group=8.
func TestPagedDecodeGQAParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(3))
	pool, err := cuda.NewPagedKVPool(512, 16, kvW)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()
	lens := []int{20, 47, 5, 128, 1, 63, 64, 33}
	batch := len(lens)
	seqs := make([]*cuda.SeqKV, batch)
	for i, n := range lens {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, n*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(n, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		t.Fatal(err)
	}
	defer view.Free()
	oNaive, err := pool.BatchedDecodeAttnView(q, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer oNaive.Free()
	oGQA, err := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer oGQA.Free()
	a := make([]float32, batch*qHeads*hd)
	bb := make([]float32, batch*qHeads*hd)
	if err := oNaive.DownloadF32(a); err != nil {
		t.Fatal(err)
	}
	if err := oGQA.DownloadF32(bb); err != nil {
		t.Fatal(err)
	}
	var num, den float64
	for i := range a {
		d := float64(a[i] - bb[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	relRMS := 0.0
	if den > 0 {
		relRMS = math.Sqrt(num / den)
	}
	if relRMS > 1e-5 {
		t.Fatalf("GQA vs naive paged decode rel-RMS %.3e too high", relRMS)
	}
	t.Logf("paged decode GQA parity rel-RMS %.3e", relRMS)
}

func BenchmarkPagedDecodeAttnGQA_b512(b *testing.B) { benchPagedDecodeAttnGQA(b, 512, 128) }

func benchPagedDecodeAttnGQA(b *testing.B, batch, seqLen int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(7))
	pool, err := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, seqLen*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(seqLen, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		b.Fatal(err)
	}
	defer view.Free()
	o, err := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
	if err != nil {
		b.Fatal(err)
	}
	o.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		o, _ := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
		o.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "attn/s")
}

// TestPagedDecodeGQAf16Parity: f16-KV decode must match the f32 kernel within f16 rounding
// (~1e-3 rel-RMS) — K/V stored as f16, converted to f32 in shared for identical compute.
func TestPagedDecodeGQAf16Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(3))
	pool, err := cuda.NewPagedKVPool(512, 16, kvW)
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Free()
	lens := []int{20, 47, 5, 128, 1, 63, 64, 33}
	batch := len(lens)
	seqs := make([]*cuda.SeqKV, batch)
	for i, n := range lens {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, n*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(n, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		t.Fatal(err)
	}
	defer view.Free()
	oF32, err := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer oF32.Free()
	oF16, err := pool.BatchedDecodeAttnViewGQAf16(q, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer oF16.Free()
	a := make([]float32, batch*qHeads*hd)
	b := make([]float32, batch*qHeads*hd)
	oF32.DownloadF32(a)
	oF16.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	relRMS := 0.0
	if den > 0 {
		relRMS = math.Sqrt(num / den)
	}
	if relRMS > 3e-3 {
		t.Fatalf("f16 vs f32 paged decode rel-RMS %.3e too high", relRMS)
	}
	t.Logf("paged decode f16-KV parity rel-RMS %.3e", relRMS)
}

func BenchmarkPagedDecodeAttnGQAf16_b512(b *testing.B) { benchPagedDecodeAttnGQAf16(b, 512, 128) }

// benchPagedDecodeAttnGQAf16 measures the f16-KV GQA attention.
//
// MEASURED (b512, 200x, 2026-07-20) — the ctx-512 "should become bandwidth-bound" hypothesis is
// DISPROVEN; attention is per-key COMPUTE-bound at BOTH contexts:
//   - GQA f32 KV : len128 707µs | len512 2698µs
//   - GQA f16 KV : len128 663µs | len512 2619µs
//
// Two independent signals:
//
//	(1) LINEAR in seqLen — 4x the keys (128->512) costs 3.95x the time, so per-key work (QK·, exp,
//	    running max/sum, PV) dominates, not a fixed launch/latency floor.
//	(2) f16 barely beats f32 — halving the K/V BYTES buys only 3-7% (663/707 @128, 2619/2698 @512).
//	    A bandwidth-bound kernel would ~halve. Achieved BW is only ~101-107 GB/s = ~30% of the 3060's
//	    ~360 GB/s peak at BOTH contexts, so bytes are not the limit.
//
// => int8/fp8 KV will NOT speed attention (fewer bytes ≠ the bottleneck; it would give even less than
//
//	f16's 3-7%). f16-KV's real payoff is CAPACITY (half the VRAM footprint => more concurrent seqs =>
//	higher max batch => the aggregate serving throughput we measured), not per-call latency. The only
//	attention speed levers left change the COMPUTE: tensor-core QK/PV (WMMA tried, negative — the M=8
//	GQA group wastes half a 16x16 tile) or fewer keys (sliding-window/sparse — an approximation).
func benchPagedDecodeAttnGQAf16(b *testing.B, batch, seqLen int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(7))
	pool, _ := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, seqLen*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(seqLen, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	o, err := pool.BatchedDecodeAttnViewGQAf16(q, view, qHeads, kvHeads)
	if err != nil {
		b.Fatal(err)
	}
	o.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		o, _ := pool.BatchedDecodeAttnViewGQAf16(q, view, qHeads, kvHeads)
		o.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "attn/s")
}

func BenchmarkPagedDecodeAttnGQAf16_b512_len512(b *testing.B) {
	benchPagedDecodeAttnGQAf16(b, 512, 512)
}

func TestPagedDecodeSKParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(3))
	pool, _ := cuda.NewPagedKVPool(512, 16, kvW)
	defer pool.Free()
	lens := []int{20, 47, 5, 128, 1, 63, 64, 33}
	batch := len(lens)
	seqs := make([]*cuda.SeqKV, batch)
	for i, n := range lens {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, n*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(n, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	oRef, _ := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
	defer oRef.Free()
	oSK, err := pool.BatchedDecodeAttnViewSK(q, view, qHeads, kvHeads, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer oSK.Free()
	a := make([]float32, batch*qHeads*hd)
	b := make([]float32, batch*qHeads*hd)
	oRef.DownloadF32(a)
	oSK.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rms := math.Sqrt(num / den)
	t.Logf("split-K vs GQA paged decode rel-RMS %.3e", rms)
	if rms > 1e-5 {
		t.Fatalf("split-K parity rel-RMS %.3e too high", rms)
	}
}

func benchPagedDecodeSK(b *testing.B, batch, seqLen, splitK int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(7))
	pool, _ := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, seqLen*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(seqLen, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	o, err := pool.BatchedDecodeAttnViewSK(q, view, qHeads, kvHeads, splitK)
	if err != nil {
		b.Skipf("sk: %v", err)
	}
	o.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		o, _ := pool.BatchedDecodeAttnViewSK(q, view, qHeads, kvHeads, splitK)
		o.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "attn/s")
}
func BenchmarkPagedDecodeSK_b512_sk2(b *testing.B) { benchPagedDecodeSK(b, 512, 128, 2) }
func BenchmarkPagedDecodeSK_b512_sk4(b *testing.B) { benchPagedDecodeSK(b, 512, 128, 4) }

func TestPagedDecodeWMMAParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(3))
	pool, _ := cuda.NewPagedKVPool(512, 16, kvW)
	defer pool.Free()
	lens := []int{20, 47, 5, 128, 1, 63, 64, 33}
	batch := len(lens)
	seqs := make([]*cuda.SeqKV, batch)
	for i, n := range lens {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, n*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(n, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	oRef, _ := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
	defer oRef.Free()
	oW, err := pool.BatchedDecodeAttnViewWMMA(q, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer oW.Free()
	a := make([]float32, batch*qHeads*hd)
	b := make([]float32, batch*qHeads*hd)
	oRef.DownloadF32(a)
	oW.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rms := math.Sqrt(num / den)
	t.Logf("WMMA vs warp-GQA paged decode rel-RMS %.3e (f16 attention)", rms)
	if rms > 6e-3 {
		t.Fatalf("WMMA decode parity rel-RMS %.3e too high", rms)
	}
}

func benchPagedDecodeWMMA(b *testing.B, batch, seqLen int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(7))
	pool, _ := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, seqLen*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(seqLen, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	o, err := pool.BatchedDecodeAttnViewWMMA(q, view, qHeads, kvHeads)
	if err != nil {
		b.Skipf("wmma: %v", err)
	}
	o.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		o, _ := pool.BatchedDecodeAttnViewWMMA(q, view, qHeads, kvHeads)
		o.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "attn/s")
}
func BenchmarkPagedDecodeWMMA_b512(b *testing.B) { benchPagedDecodeWMMA(b, 512, 128) }

func TestPagedBatchViewUpdate(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(4))
	pool, _ := cuda.NewPagedKVPool(256, 16, kvW)
	defer pool.Free()
	lens := []int{20, 47, 5, 100}
	batch := len(lens)
	seqs := make([]*cuda.SeqKV, batch)
	for i, n := range lens {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, n*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(n, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, _ := pool.UploadBatchView(seqs) // sized for current (100-tok -> 7 blocks) state
	defer view.Free()
	// append a token to each seq, then Update the view in place
	for i := range seqs {
		kf := make([]float32, kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		d, _ := cuda.NewDeviceF32(1, kvW)
		d.UploadF32(kf)
		seqs[i].Append(d, d)
		d.Free()
	}
	if err := view.Update(seqs); err != nil {
		t.Fatal(err)
	}
	oUpd, err := pool.BatchedDecodeAttnView(q, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer oUpd.Free()
	fresh, _ := pool.UploadBatchView(seqs)
	defer fresh.Free()
	oFresh, _ := pool.BatchedDecodeAttnView(q, fresh, qHeads, kvHeads)
	defer oFresh.Free()
	a := make([]float32, batch*qHeads*hd)
	b := make([]float32, batch*qHeads*hd)
	oUpd.DownloadF32(a)
	oFresh.DownloadF32(b)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("Update view attn != fresh view attn at %d: %v vs %v", i, a[i], b[i])
		}
	}
	t.Logf("PagedBatchView.Update in-place == fresh view (bit-exact), post-append")
}

// TestGraphDecodeGrowingKV: the make-or-break for the fixed-buffer graph decode — a CAPTURED
// attention graph must correctly attend over GROWING KV when the persistent view is updated in
// place between replays (vLLM's pattern). Captures the attention once, then each step appends a
// K/V token + Update()s the view + replays, comparing the graph output to a fresh eager attention.
func TestGraphDecodeGrowingKV(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	const startLen, maxLen, batch = 40, 90, 3
	rng := rand.New(rand.NewSource(6))
	pool, _ := cuda.NewPagedKVPool(batch*(maxLen/16+2), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	appendTok := func(s *cuda.SeqKV) {
		kf := make([]float32, kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		d, _ := cuda.NewDeviceF32(1, kvW)
		d.UploadF32(kf)
		s.Append(d, d)
		d.Free()
	}
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		for k := 0; k < startLen; k++ {
			appendTok(seqs[i])
		}
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	// grow every sequence to maxLen up front so the view's maxBlocks capacity covers the max,
	// then Release+re-seed to startLen (the view is sized for max).
	for i := range seqs {
		for k := startLen; k < maxLen; k++ {
			appendTok(seqs[i])
		}
	}
	view, _ := pool.UploadBatchView(seqs) // sized for maxLen blocks
	defer view.Free()
	// reset to startLen
	for i := range seqs {
		seqs[i].Release()
		seqs[i] = pool.NewSeqKV()
		for k := 0; k < startLen; k++ {
			appendTok(seqs[i])
		}
	}
	view.Update(seqs)
	out, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	defer out.Free()

	// capture the attention (fixed output) reading the persistent view
	pool.BatchedDecodeAttnViewInto(q, view, qHeads, kvHeads, out) // warm
	cuda.GraphSync()
	if err := cuda.CaptureBegin(); err != nil {
		t.Skipf("capture: %v", err)
	}
	pool.BatchedDecodeAttnViewInto(q, view, qHeads, kvHeads, out)
	g, err := cuda.CaptureEnd()
	if err != nil {
		t.Skipf("capture end: %v", err)
	}
	defer g.Free()

	for step := 0; step < 6; step++ {
		for i := range seqs {
			appendTok(seqs[i]) // grow KV
		}
		if err := view.Update(seqs); err != nil {
			t.Fatal(err)
		}
		g.Launch() // replay the captured attention over the grown KV
		cuda.GraphSync()
		gotHost := make([]float32, batch*qHeads*hd)
		out.DownloadF32(gotHost)
		// eager reference at the same state
		ref, _ := pool.BatchedDecodeAttnView(q, view, qHeads, kvHeads)
		refHost := make([]float32, batch*qHeads*hd)
		ref.DownloadF32(refHost)
		ref.Free()
		var num, den float64
		for i := range gotHost {
			d := float64(gotHost[i] - refHost[i])
			num += d * d
			den += float64(refHost[i]) * float64(refHost[i])
		}
		rms := math.Sqrt(num / den)
		if rms > 1e-5 { // f32-reduction-order noise is ~1e-7; a real growing-KV mismatch would be O(1)
			t.Fatalf("step %d (len %d): graph replay vs eager rel-RMS %.3e too high", step, seqs[0].Len(), rms)
		}
		t.Logf("step %d: captured graph attends over grown KV (len %d) == eager (rel-RMS %.2e)", step, seqs[0].Len(), rms)
	}
	for _, s := range seqs {
		s.Release()
	}
}

// Long-context (512) attention isolation: at ctx 128 split-K lost (batch already SM-saturates), but
// the warp-GQA online-softmax scan is 4x longer at ctx 512 — does split-K's shorter parallel chains
// now win, or does the high-batch SM saturation still dominate? Rigor for the "long-context decode
// attention = the real lever vs vLLM FlashDecoding" hypothesis (Iw8).
func BenchmarkPagedDecodeAttnGQA_b512_len512(b *testing.B) { benchPagedDecodeAttnGQA(b, 512, 512) }
func BenchmarkPagedDecodeSK4_b512_len512(b *testing.B)     { benchPagedDecodeSK(b, 512, 512, 4) }
func BenchmarkPagedDecodeSK8_b512_len512(b *testing.B)     { benchPagedDecodeSK(b, 512, 512, 8) }

// TestPagedDecodeWMMAFlashParity validates the tiled FlashDecoding WMMA kernel against the warp-GQA
// reference across RAGGED lengths INCLUDING long ones (300, 512) the non-flash WMMA (seqLen<=128)
// cannot handle. This is the correctness gate for the one remaining perf lever (long-context decode).
func TestPagedDecodeWMMAFlashParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(11))
	lens := []int{20, 47, 512, 1, 300, 128, 64, 33, 256, 511}
	maxN := 0
	for _, n := range lens {
		if n > maxN {
			maxN = n
		}
	}
	pool, _ := cuda.NewPagedKVPool(len(lens)*((maxN+16)/16+1), 16, kvW)
	defer pool.Free()
	batch := len(lens)
	seqs := make([]*cuda.SeqKV, batch)
	for i, n := range lens {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, n*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(n, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	oRef, _ := pool.BatchedDecodeAttnViewGQA(q, view, qHeads, kvHeads)
	defer oRef.Free()
	oF, err := pool.BatchedDecodeAttnViewWMMAFlash(q, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer oF.Free()
	a := make([]float32, batch*qHeads*hd)
	b := make([]float32, batch*qHeads*hd)
	oRef.DownloadF32(a)
	oF.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rms := math.Sqrt(num / den)
	t.Logf("WMMA-Flash vs warp-GQA paged decode rel-RMS %.3e (ragged incl 512/511/300)", rms)
	if rms > 6e-3 {
		t.Fatalf("WMMA-Flash decode parity rel-RMS %.3e too high", rms)
	}
}

// benchPagedDecodeWMMAFlash: tiled FlashDecoding WMMA attention isolation (uniform seqLen).
func benchPagedDecodeWMMAFlash(b *testing.B, batch, seqLen int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(7))
	pool, _ := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, batch)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, seqLen*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(seqLen, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, batch*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(batch, qHeads*hd)
	q.UploadF32(qf)
	defer q.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	o, err := pool.BatchedDecodeAttnViewWMMAFlash(q, view, qHeads, kvHeads)
	if err != nil {
		b.Fatal(err)
	}
	o.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		o, _ := pool.BatchedDecodeAttnViewWMMAFlash(q, view, qHeads, kvHeads)
		o.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "attn/s")
}

func BenchmarkPagedDecodeWMMAFlash_b512_len512(b *testing.B) { benchPagedDecodeWMMAFlash(b, 512, 512) }
func BenchmarkPagedDecodeWMMAFlash_b512_len128(b *testing.B) { benchPagedDecodeWMMAFlash(b, 512, 128) }
