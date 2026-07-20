//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// Pre/post benchmark for the #194 qio serving-attention optimization: the production serving decoders
// feed an f16 query and want an f16 result over an f32-KV pool. PRE = the old path (f16->f32 convert
// in, f32-KV GQA, f32->f16 convert out). POST = cu_paged_decode_attn_gqa_qio (f16 Q-in/O-out, same
// f32-KV). The delta is the two per-layer conversions the optimization removes.
func benchServingAttn(b *testing.B, batch, seqLen int, qio bool) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	const qHeads, kvHeads, hd = 32, 4, 64
	kvW, qW := kvHeads*hd, qHeads*hd
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
	// f16 query (as the serving decode produces it)
	qf := make([]float32, batch*qW)
	for i := range qf {
		qf[i] = float32(rng.NormFloat64()) * 0.1
	}
	qtmp, _ := cuda.NewDeviceF32(batch, qW)
	qtmp.UploadF32(qf)
	q16, _ := cuda.F16FromF32(qtmp)
	qtmp.Free()
	defer q16.Free()
	view, err := pool.UploadBatchView(seqs)
	if err != nil {
		b.Fatal(err)
	}
	defer view.Free()

	// one serving-attention step: the exact op sequence the deploy/continuous decoders run per layer.
	step := func() {
		if qio {
			da16, _ := pool.BatchedDecodeAttnViewGQAQio(q16, view, qHeads, kvHeads) // f16 in, f16 out
			da16.Free()
		} else {
			dqf, _ := q16.ToF32() // f16 -> f32 (removed by qio)
			da, _ := pool.BatchedDecodeAttnViewGQA(dqf, view, qHeads, kvHeads)
			dqf.Free()
			da16, _ := cuda.F16FromF32(da) // f32 -> f16 (removed by qio)
			da.Free()
			da16.Free()
		}
	}
	step()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		step()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "step/s")
}

// PRE (with conversions) vs POST (qio) at serving batches.
func BenchmarkServingAttn_b64_pre(b *testing.B)  { benchServingAttn(b, 64, 128, false) }
func BenchmarkServingAttn_b64_qio(b *testing.B)  { benchServingAttn(b, 64, 128, true) }
func BenchmarkServingAttn_b256_pre(b *testing.B) { benchServingAttn(b, 256, 128, false) }
func BenchmarkServingAttn_b256_qio(b *testing.B) { benchServingAttn(b, 256, 128, true) }
