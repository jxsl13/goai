//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Tw81 measure-first probe: is DECODE attention (single query row vs the full KV cache)
// at the memory-bandwidth roofline? Decode attention is memory-bound — the split-K flash
// kernel reads K+V once each (coalesced), so achieved GB/s vs the RTX 3060's ~360 GB/s
// peak tells us whether there's headroom. Bytes read (GQA reads each KV head once, shared
// by its query group): seqKV * kvHeads * hd * 2(K+V) * dtypeBytes. We report GB/s so the
// roofline gap is directly visible; a bench also guards the kernel against regression.
//
// MEASURED (RTX 3060): gqa8/ctx2048 129 GB/s, gqa8/ctx4096 146, mha/ctx2048 178 — i.e.
// ~35-50% of the ~360 GB/s peak, and achieved BW RISES with more KV bytes. That trend is
// the signature of an OCCUPANCY/LATENCY-bound kernel, not a bandwidth-bound one: the
// per-block 32-row K+V shared tile is ~33KB (+ group*hd*4 for the query), so occupancy is
// pinned near 1 block/SM and memory latency isn't hidden → ~2x headroom. Corroborated by
// the f16-KV path being NO faster (halving bytes doesn't help a non-BW-bound kernel). The
// occupancy pin: even at the max supported group=8 the shared is ~37KB (33KB K/V tile +
// group*hd*4 query) → 1 block/SM on the 48KB budget. Fix: shrink the K/V shared tile so
// >=2 blocks/SM resident → hides latency → toward roofline (Tw82).
// NOTE: the flash path guards group = qHeads/kvHeads > 8 with code -4 (register/warp
// budget); pure MQA / kvHeads<=hd/16 route through the non-flash attention path instead.
//
// Shapes are real decode configs (all group<=8): Mistral/Llama GQA, Llama2 MHA, long-ctx.

func benchFlashDecodeF32(b *testing.B, qHeads, kvHeads, hd, seqKV int) {
	wq, wkv := qHeads*hd, kvHeads*hd
	pos, err := cuda.NewDevicePos()
	if err != nil {
		b.Fatal(err)
	}
	defer pos.Free()
	dq, err := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 12))
	if err != nil {
		b.Fatal(err)
	}
	defer dq.Free()
	dk, err := cuda.UploadF32(bench.RandF32(tensor.Shape{seqKV, wkv}, 13))
	if err != nil {
		b.Fatal(err)
	}
	defer dk.Free()
	dv, err := cuda.UploadF32(bench.RandF32(tensor.Shape{seqKV, wkv}, 14))
	if err != nil {
		b.Fatal(err)
	}
	defer dv.Free()
	out, err := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 16))
	if err != nil {
		b.Fatal(err)
	}
	defer out.Free()
	if err := pos.Set(seqKV - 1); err != nil {
		b.Fatal(err)
	}
	run := func() {
		if err := cuda.GroupedQueryAttentionKVDposFlashInto(dq, dk, dv, qHeads, kvHeads, pos, out); err != nil {
			b.Fatal(err)
		}
	}
	run()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		run()
	}
	cuda.GraphSync()
	b.StopTimer()
	// K+V read once each, f32 = 4 bytes/elem.
	bytesRead := float64(seqKV) * float64(kvHeads) * float64(hd) * 2 * 4
	secPerOp := b.Elapsed().Seconds() / float64(b.N)
	b.ReportMetric(bytesRead/secPerOp/1e9, "GB/s")
	b.ReportMetric(secPerOp*1e6, "us/op")
}

func BenchmarkFlashDecodeF32(b *testing.B) {
	cases := []struct {
		name                     string
		qHeads, kvHeads, hd, ctx int
	}{
		{"Mistral_gqa8_ctx2048", 32, 8, 128, 2048},
		{"Mistral_gqa8_ctx4096", 32, 8, 128, 4096},
		{"Llama2_mha_ctx2048", 32, 32, 128, 2048},
		{"GQA8_ctx8192", 32, 8, 128, 8192}, // group=4, long context — biggest KV read, most decode-relevant
	}
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) { benchFlashDecodeF32(b, c.qHeads, c.kvHeads, c.hd, c.ctx) })
	}
}

// f16-KV is the production decode path (halves KV bytes). Fill the cache row-by-row in
// setup (outside the timer), then time only the flash call.
func BenchmarkFlashDecodeF16(b *testing.B) {
	const qHeads, kvHeads, hd, ctx = 32, 8, 128, 2048
	wq, wkv := qHeads*hd, kvHeads*hd
	pos, err := cuda.NewDevicePos()
	if err != nil {
		b.Fatal(err)
	}
	defer pos.Free()
	dq, err := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 12))
	if err != nil {
		b.Fatal(err)
	}
	defer dq.Free()
	k := bench.RandF32(tensor.Shape{ctx, wkv}, 13)
	v := bench.RandF32(tensor.Shape{ctx, wkv}, 14)
	cf, err := cuda.NewKVCacheF16(ctx, wkv)
	if err != nil {
		b.Fatal(err)
	}
	defer cf.Free()
	if err := cf.ZeroCache(); err != nil {
		b.Fatal(err)
	}
	for i := 0; i < ctx; i++ {
		kr, _ := cuda.UploadF32(rowOf(k, i))
		vr, _ := cuda.UploadF32(rowOf(v, i))
		if err := pos.Set(i); err != nil {
			b.Fatal(err)
		}
		if err := cf.AppendDpos(kr, vr, pos); err != nil {
			b.Fatal(err)
		}
		kr.Free()
		vr.Free()
	}
	out, err := cuda.UploadF32(bench.RandF32(tensor.Shape{1, wq}, 16))
	if err != nil {
		b.Fatal(err)
	}
	defer out.Free()
	if err := pos.Set(ctx - 1); err != nil {
		b.Fatal(err)
	}
	run := func() {
		if err := cuda.GroupedQueryAttentionKVF16DposFlashInto(dq, cf, qHeads, kvHeads, pos, out); err != nil {
			b.Fatal(err)
		}
	}
	run()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		run()
	}
	cuda.GraphSync()
	b.StopTimer()
	bytesRead := float64(ctx) * float64(kvHeads) * float64(hd) * 2 * 2 // f16 = 2 bytes/elem
	secPerOp := b.Elapsed().Seconds() / float64(b.N)
	b.ReportMetric(bytesRead/secPerOp/1e9, "GB/s")
	b.ReportMetric(secPerOp*1e6, "us/op")
}
