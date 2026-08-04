//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestPagedDecodeGQAf16QioParity: the f16-Q-in/f16-O-out attention must be bit-identical to the
// f32-IO f16-KV kernel fed the SAME f16-rounded query, then rounded to f16 — i.e. the two
// conversions it removes are exactly the ones it folds in. Validates the +1.5-2.4% qio win is free.
func TestPagedDecodeGQAf16QioParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW := kvHeads * hd
	qW := qHeads * hd
	rng := rand.New(rand.NewSource(4))
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
	// query in f32, then rounded to f16 so BOTH paths see identical query values
	qf := make([]float32, batch*qW)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	qF32 := func() *cuda.DeviceF32 { d, _ := cuda.NewDeviceF32(batch, qW); d.UploadF32(qf); return d }
	qtmp := qF32()
	q16 := cuda.AllocU16(batch * qW)
	cuda.CvtF32ToF16(q16, qtmp.DevPtr(), batch*qW)
	qtmp.Free()
	qRounded, _ := cuda.NewDeviceF32(batch, qW)
	cuda.CvtF16ToF32(qRounded.DevPtr(), q16, batch*qW) // f32(f16(qf)) — the values qio's h2f sees
	defer qRounded.Free()
	defer cuda.FreeDev(q16)

	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		t.Fatal(err)
	}
	defer view.Free()

	// reference: f32-IO f16-KV attention on the f16-rounded query, then round the output to f16
	ref32, err := pool.BatchedDecodeAttnViewGQAf16(qRounded, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer ref32.Free()
	refH := cuda.AllocU16(batch * qW)
	cuda.CvtF32ToF16(refH, ref32.DevPtr(), batch*qW)
	defer cuda.FreeDev(refH)
	refBack, _ := cuda.NewDeviceF32(batch, qW)
	cuda.CvtF16ToF32(refBack.DevPtr(), refH, batch*qW)
	defer refBack.Free()

	// qio: f16 query in, f16 out
	qioH, err := pool.BatchedDecodeAttnViewGQAf16Qio(q16, qW, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.FreeDev(qioH)
	qioBack, _ := cuda.NewDeviceF32(batch, qW)
	cuda.CvtF16ToF32(qioBack.DevPtr(), qioH, batch*qW)
	defer qioBack.Free()

	a := make([]float32, batch*qW)
	b := make([]float32, batch*qW)
	refBack.DownloadF32(a)
	qioBack.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rel := math.Sqrt(num / den)
	if rel > 1e-6 {
		t.Fatalf("qio vs f32-IO+convert rel-RMS %.3e too high", rel)
	}
	t.Logf("qio (f16 Q-in/O-out) vs f32-IO f16-KV + converts: rel-RMS %.3e — bit-identical", rel)
}

// hd=128 variants of the qio parity tests — validate the generalized a0..a3 lane
// packing on the ushort Q-in/O-out convert-elision decode paths (Llama-2/Mistral/Qwen2).
func TestPagedDecodeGQAQioParityHd128(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 8, 2, 128
	kvW := kvHeads * hd
	qW := qHeads * hd
	rng := rand.New(rand.NewSource(14))
	pool, _ := cuda.NewPagedKVPool(256, 16, kvW)
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
	qf := make([]float32, batch*qW)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	qtmp, _ := cuda.NewDeviceF32(batch, qW)
	qtmp.UploadF32(qf)
	q16, _ := cuda.F16FromF32(qtmp)
	qtmp.Free()
	defer q16.Free()
	qRounded, _ := q16.ToF32()
	defer qRounded.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	ref32, err := pool.BatchedDecodeAttnViewGQA(qRounded, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer ref32.Free()
	refH, _ := cuda.F16FromF32(ref32)
	defer refH.Free()
	refBack, _ := refH.ToF32()
	defer refBack.Free()
	qioOut, err := pool.BatchedDecodeAttnViewGQAQio(q16, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer qioOut.Free()
	qioBack, _ := qioOut.ToF32()
	defer qioBack.Free()
	a := make([]float32, batch*qW)
	b := make([]float32, batch*qW)
	refBack.DownloadF32(a)
	qioBack.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rel := math.Sqrt(num / den)
	t.Logf("hd=128 GQA-qio vs f32-IO+convert rel-RMS %.3e", rel)
	if rel > 1e-6 {
		t.Fatalf("hd=128 GQA-qio rel-RMS %.3e too high", rel)
	}
}

func TestPagedDecodeGQAf16QioParityHd128(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 8, 2, 128
	kvW := kvHeads * hd
	qW := qHeads * hd
	rng := rand.New(rand.NewSource(15))
	pool, _ := cuda.NewPagedKVPool(256, 16, kvW)
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
	qf := make([]float32, batch*qW)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	qtmp, _ := cuda.NewDeviceF32(batch, qW)
	qtmp.UploadF32(qf)
	q16 := cuda.AllocU16(batch * qW)
	cuda.CvtF32ToF16(q16, qtmp.DevPtr(), batch*qW)
	qtmp.Free()
	defer cuda.FreeDev(q16)
	qRounded, _ := cuda.NewDeviceF32(batch, qW)
	cuda.CvtF16ToF32(qRounded.DevPtr(), q16, batch*qW)
	defer qRounded.Free()
	view, _ := pool.UploadBatchView(seqs)
	defer view.Free()
	ref32, err := pool.BatchedDecodeAttnViewGQAf16(qRounded, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer ref32.Free()
	refH := cuda.AllocU16(batch * qW)
	cuda.CvtF32ToF16(refH, ref32.DevPtr(), batch*qW)
	defer cuda.FreeDev(refH)
	refBack, _ := cuda.NewDeviceF32(batch, qW)
	cuda.CvtF16ToF32(refBack.DevPtr(), refH, batch*qW)
	defer refBack.Free()
	qioH, err := pool.BatchedDecodeAttnViewGQAf16Qio(q16, qW, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer cuda.FreeDev(qioH)
	qioBack, _ := cuda.NewDeviceF32(batch, qW)
	cuda.CvtF16ToF32(qioBack.DevPtr(), qioH, batch*qW)
	defer qioBack.Free()
	a := make([]float32, batch*qW)
	b := make([]float32, batch*qW)
	refBack.DownloadF32(a)
	qioBack.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rel := math.Sqrt(num / den)
	t.Logf("hd=128 f16-GQA-qio vs f16-IO+convert rel-RMS %.3e", rel)
	if rel > 1e-6 {
		t.Fatalf("hd=128 f16-GQA-qio rel-RMS %.3e too high", rel)
	}
}
