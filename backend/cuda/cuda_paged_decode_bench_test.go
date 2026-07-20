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

func BenchmarkPagedDecodeAttnGQAf16_b512(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	rng := rand.New(rand.NewSource(7))
	pool, _ := cuda.NewPagedKVPool(512*((128+16)/16+1), 16, kvW)
	defer pool.Free()
	seqs := make([]*cuda.SeqKV, 512)
	for i := range seqs {
		seqs[i] = pool.NewSeqKV()
		kf := make([]float32, 128*kvW)
		for j := range kf {
			kf[j] = float32(rng.NormFloat64()) * 0.1
		}
		dk, _ := cuda.NewDeviceF32(128, kvW)
		dk.UploadF32(kf)
		seqs[i].Append(dk, dk)
		dk.Free()
	}
	qf := make([]float32, 512*qHeads*hd)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	q, _ := cuda.NewDeviceF32(512, qHeads*hd)
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
