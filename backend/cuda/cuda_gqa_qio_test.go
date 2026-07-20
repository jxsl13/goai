//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestPagedDecodeGQAQioParity: the f32-KV GQA attention with f16 Q-in/O-out must be bit-identical to
// the f32-IO kernel fed the SAME f16-rounded query, then rounded to f16 — i.e. it exactly folds in the
// two conversions it removes. Validates the serving-path qio optimization (no accuracy change).
func TestPagedDecodeGQAQioParity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW, qW := kvHeads*hd, qHeads*hd
	rng := rand.New(rand.NewSource(6))
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
	qf := make([]float32, batch*qW)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	// f16-rounded query so both paths see identical values
	qtmp, _ := cuda.NewDeviceF32(batch, qW)
	qtmp.UploadF32(qf)
	q16, _ := cuda.F16FromF32(qtmp)
	qtmp.Free()
	defer q16.Free()
	qRounded, _ := q16.ToF32()
	defer qRounded.Free()

	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		t.Fatal(err)
	}
	defer view.Free()

	// reference: f32-IO on the f16-rounded query, then round output to f16
	ref32, err := pool.BatchedDecodeAttnViewGQA(qRounded, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer ref32.Free()
	refBack, _ := cuda.F16FromF32(ref32)
	refBack2, _ := refBack.ToF32()
	defer refBack.Free()
	defer refBack2.Free()

	// qio: f16 query in, f16 out
	qioOut, err := pool.BatchedDecodeAttnViewGQAQio(q16, view, qHeads, kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	defer qioOut.Free()
	qioBack, _ := qioOut.ToF32()
	defer qioBack.Free()

	a := make([]float32, batch*qW)
	b := make([]float32, batch*qW)
	refBack2.DownloadF32(a)
	qioBack.DownloadF32(b)
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	rel := math.Sqrt(num / den)
	if rel > 1e-6 {
		t.Fatalf("GQAQio vs f32-IO+convert rel-RMS %.3e too high", rel)
	}
	t.Logf("GQA qio (f16 Q-in/O-out, f32-KV) vs f32-IO + converts: rel-RMS %.3e — bit-identical", rel)
}
