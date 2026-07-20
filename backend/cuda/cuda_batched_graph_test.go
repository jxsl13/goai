//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// §ROADMAP FRONT B — kill the per-step overhead on the batched decode forward. B4's eager cut
// (4245 tok/s at batch=64, 0.91× vLLM) re-uploaded + device-synced each sequence's block table
// ONCE PER LAYER (22 full host↔device round-trips per step). This bench compares three modes:
//
//	eager — B4's BatchedDecodeAttn (per-layer upload + sync)
//	view  — UploadBatchView once per step + BatchedDecodeAttnView (no per-layer sync)
//	graph — the view path captured as a CUDA graph and replayed (launch-overhead-free)
//
// The lever to push batch=64 aggregate throughput PAST vLLM's ~4655.
const (
	bgDim, bgQHeads, bgKVHeads, bgHD, bgHidden = 2048, 32, 4, 64, 5632
	bgVocab                                    = 32000 // TinyLlama vocab, for the full-step logits GEMM
)

type bgLayer struct {
	gAttn, gFFN    *cuda.ResidentVec
	wq, wk, wv, wo *cuda.ResidentBF16
	wg, wu, wd     *cuda.ResidentBF16
}

func bgBuild(b testing.TB, batch, seqLen, layers int) ([]*bgLayer, *cuda.PagedKVPool, []*cuda.SeqKV, *cuda.DeviceF32) {
	qW, kvW := bgQHeads*bgHD, bgKVHeads*bgHD
	rng := rand.New(rand.NewSource(9))
	mkW := func(k, n int) *cuda.ResidentBF16 {
		t := tensor.New(tensor.F32, tensor.Shape{k, n})
		f := t.Storage().F32()
		for i := range f {
			f[i] = float32(rng.NormFloat64()) * 0.02
		}
		r, err := cuda.NewResidentBF16(t)
		if err != nil {
			b.Fatal(err)
		}
		return r
	}
	mkVec := func(n int) *cuda.ResidentVec {
		t := tensor.New(tensor.F32, tensor.Shape{n})
		for i := range t.Storage().F32() {
			t.Storage().F32()[i] = 1.0
		}
		r, _ := cuda.NewResidentVec(t)
		return r
	}
	ls := make([]*bgLayer, layers)
	for i := range ls {
		ls[i] = &bgLayer{gAttn: mkVec(bgDim), gFFN: mkVec(bgDim),
			wq: mkW(bgDim, qW), wk: mkW(bgDim, kvW), wv: mkW(bgDim, kvW), wo: mkW(qW, bgDim),
			wg: mkW(bgDim, bgHidden), wu: mkW(bgDim, bgHidden), wd: mkW(bgHidden, bgDim)}
	}
	pool, err := cuda.NewPagedKVPool(batch*((seqLen+16)/16+1), 16, kvW)
	if err != nil {
		b.Fatal(err)
	}
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
	xf := make([]float32, batch*bgDim)
	for i := range xf {
		xf[i] = float32(rng.NormFloat64()) * 0.1
	}
	x0, _ := cuda.NewDeviceF32(batch, bgDim)
	x0.UploadF32(xf)
	return ls, pool, seqs, x0
}

func benchBatchedGraph(b *testing.B, batch, seqLen, layers int, mode string) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	ls, pool, seqs, x0 := bgBuild(b, batch, seqLen, layers)
	defer pool.Free()
	defer x0.Free()
	defer func() { // free resident weights so a batch sweep in one process doesn't leak VRAM
		for _, l := range ls {
			l.gAttn.Free()
			l.gFFN.Free()
			for _, w := range []*cuda.ResidentBF16{l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd} {
				w.Free()
			}
		}
	}()
	rq := backend.RoPEAttrs{Base: 10000, Heads: bgQHeads, PosOffset: seqLen}
	rk := backend.RoPEAttrs{Base: 10000, Heads: bgKVHeads, PosOffset: seqLen}

	// skipAttn isolates attention's true share of the captured step: feed dq straight into wo
	// (same [batch,qHeads*hd] shape), so full-step minus this baseline = attention cost.
	skipAttn := mode == "graph-noattn"
	gqaAttn := mode == "graph-gqa" // use the GQA K/V-shared paged decode kernel
	gqaF16 := mode == "graph-gqa-f16" || mode == "graph-gqa-f16-full"

	// full-step mode adds the decoder HEAD to the captured step (final RMSNorm + logits GEMM
	// [batch,dim]×[dim,vocab]), so the tok/s includes the ~logits cost vLLM's number also pays —
	// closing the "layers-only" caveat. The argmax reduction (one memory-bound pass over the logits)
	// is omitted as <1%. Head weights are synthetic (throughput is weight-value-independent).
	fullStep := mode == "graph-gqa-f16-full"
	var wLogits *cuda.ResidentBF16
	var gFinal *cuda.ResidentVec
	if fullStep {
		rngH := rand.New(rand.NewSource(11))
		lt := tensor.New(tensor.F32, tensor.Shape{bgDim, bgVocab})
		lf := lt.Storage().F32()
		for i := range lf {
			lf[i] = float32(rngH.NormFloat64()) * 0.02
		}
		var err error
		wLogits, err = cuda.NewResidentBF16(lt)
		if err != nil {
			b.Fatal(err)
		}
		defer wLogits.Free()
		gt := tensor.New(tensor.F32, tensor.Shape{bgDim})
		for i := range gt.Storage().F32() {
			gt.Storage().F32()[i] = 1.0
		}
		gFinal, _ = cuda.NewResidentVec(gt)
		defer gFinal.Free()
	}

	// step runs one full batched decode forward. When view != nil it uses the pre-uploaded
	// block table (view mode / graph mode); otherwise it re-uploads per layer (eager mode).
	step := func(view *cuda.PagedBatchView) {
		x, _ := cuda.NewDeviceF32(batch, bgDim)
		x.CopyFrom(x0)
		for _, l := range ls {
			dh, _ := x.RMSNormTo(l.gAttn, 1e-5)
			dq, _ := l.wq.MatMulDevice(dh)
			dk, _ := l.wk.MatMulDevice(dh)
			dv, _ := l.wv.MatMulDevice(dh)
			dh.Free()
			dq.RoPE(rq)
			dk.RoPE(rk)
			dk.Free()
			dv.Free()
			var da *cuda.DeviceF32
			var err error
			if skipAttn {
				da = dq // alias: skip attention, reuse dq as its output
			} else if gqaF16 {
				da, err = pool.BatchedDecodeAttnViewGQAf16(dq, view, bgQHeads, bgKVHeads)
			} else if gqaAttn {
				da, err = pool.BatchedDecodeAttnViewGQA(dq, view, bgQHeads, bgKVHeads)
			} else if view != nil {
				da, err = pool.BatchedDecodeAttnView(dq, view, bgQHeads, bgKVHeads)
			} else {
				da, err = pool.BatchedDecodeAttn(dq, seqs, bgQHeads, bgKVHeads)
			}
			if err != nil {
				b.Fatal(err)
			}
			if !skipAttn {
				dq.Free()
			}
			l.wo.MatMulAccInto(da, x)
			da.Free()
			dh2, _ := x.RMSNormTo(l.gFFN, 1e-5)
			dg, _ := l.wg.MatMulDevice(dh2)
			du, _ := l.wu.MatMulDevice(dh2)
			dh2.Free()
			dg.SwiGLU(du)
			du.Free()
			l.wd.MatMulAccInto(dg, x)
			dg.Free()
		}
		if fullStep {
			dhf, _ := x.RMSNormTo(gFinal, 1e-5)
			lg, err := wLogits.MatMulDevice(dhf)
			if err != nil {
				b.Fatal(err)
			}
			dhf.Free()
			lg.Free()
		}
		x.Free()
	}

	switch mode {
	case "eager":
		step(nil)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			step(nil)
		}
		cuda.GraphSync()
		b.StopTimer()
	case "view":
		view, err := pool.UploadBatchView(seqs)
		if err != nil {
			b.Fatal(err)
		}
		defer view.Free()
		step(view)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			step(view)
		}
		cuda.GraphSync()
		b.StopTimer()
	case "graph", "graph-noattn", "graph-gqa", "graph-gqa-f16", "graph-gqa-f16-full":
		view, err := pool.UploadBatchView(seqs)
		if err != nil {
			b.Fatal(err)
		}
		defer view.Free()
		step(view) // warm up (cuBLAS heuristics, kernel load)
		cuda.GraphSync()
		if err := cuda.CaptureBegin(); err != nil {
			b.Skipf("capture begin: %v", err)
		}
		step(view)
		g, err := cuda.CaptureEnd()
		if err != nil {
			b.Skipf("capture end (cuBLAS/alloc not capturable): %v", err)
		}
		defer g.Free()
		g.Launch()
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			g.Launch()
		}
		cuda.GraphSync()
		b.StopTimer()
	}
	b.ReportMetric(float64(batch)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}

func BenchmarkBatchedGraph_b64_eager(b *testing.B) { benchBatchedGraph(b, 64, 128, 22, "eager") }
func BenchmarkBatchedGraph_b64_view(b *testing.B)  { benchBatchedGraph(b, 64, 128, 22, "view") }
func BenchmarkBatchedGraph_b64_graph(b *testing.B) { benchBatchedGraph(b, 64, 128, 22, "graph") }

// Batch sweep on the graph path: decode streams the full weight set once per step regardless of
// batch, so aggregate tok/s climbs with batch until the GEMMs go compute-bound. Finds our peak
// sustained serving throughput vs vLLM's ~4655.
func BenchmarkBatchedGraph_b128_graph(b *testing.B) { benchBatchedGraph(b, 128, 128, 22, "graph") }
func BenchmarkBatchedGraph_b256_graph(b *testing.B) { benchBatchedGraph(b, 256, 128, 22, "graph") }
func BenchmarkBatchedGraph_b384_graph(b *testing.B) { benchBatchedGraph(b, 384, 128, 22, "graph") }

func BenchmarkBatchedGraph_b512_graph(b *testing.B) { benchBatchedGraph(b, 512, 128, 22, "graph") }

// full vs noattn at b512: the delta is attention's true contribution to the captured step.
func BenchmarkBatchedGraph_b512_noattn(b *testing.B) {
	benchBatchedGraph(b, 512, 128, 22, "graph-noattn")
}

// Moderate-batch noattn (GEMMs+elementwise only, no attention, no logits) to decompose the
// matched-precision gap vs vLLM: full-step minus noattn = attention's share of the step. Run at
// GOAI_CUDA_F16ACC=0 for the like-for-like (f32-acc) comparison.
func BenchmarkBatchedGraph_b64_noattn(b *testing.B) {
	benchBatchedGraph(b, 64, 128, 22, "graph-noattn")
}
func BenchmarkBatchedGraph_b256_noattn(b *testing.B) {
	benchBatchedGraph(b, 256, 128, 22, "graph-noattn")
}

// GQA K/V-shared attention in the full captured step at b512 — the e2e payoff of cutting the
// naive kernel's group× redundant K/V reads (attention was ~45% of the step).
func BenchmarkBatchedGraph_b512_gqa(b *testing.B) { benchBatchedGraph(b, 512, 128, 22, "graph-gqa") }

func BenchmarkBatchedGraph_b768_gqa(b *testing.B) { benchBatchedGraph(b, 768, 128, 22, "graph-gqa") }
func BenchmarkBatchedGraph_b256_gqa(b *testing.B) { benchBatchedGraph(b, 256, 128, 22, "graph-gqa") }

// Low-concurrency points on the production f16-KV path, to compare apples-to-apples with vLLM's
// published 64-concurrent serving number (BENCHMARKS.md) rather than only the saturated peak.
func BenchmarkBatchedGraph_b64_gqaf16(b *testing.B) {
	benchBatchedGraph(b, 64, 128, 22, "graph-gqa-f16")
}
func BenchmarkBatchedGraph_b128_gqaf16(b *testing.B) {
	benchBatchedGraph(b, 128, 128, 22, "graph-gqa-f16")
}

func BenchmarkBatchedGraph_b512_gqaf16(b *testing.B) {
	benchBatchedGraph(b, 512, 128, 22, "graph-gqa-f16")
}

// Full-step (layers + final-norm + logits GEMM) on the production f16-KV path — the honest,
// apples-to-apples serving number vs vLLM's full-pipeline figure (drops the layers-only caveat).
func BenchmarkBatchedGraph_b64_gqaf16_full(b *testing.B) {
	benchBatchedGraph(b, 64, 128, 22, "graph-gqa-f16-full")
}
func BenchmarkBatchedGraph_b256_gqaf16_full(b *testing.B) {
	benchBatchedGraph(b, 256, 128, 22, "graph-gqa-f16-full")
}
func BenchmarkBatchedGraph_b512_gqaf16_full(b *testing.B) {
	benchBatchedGraph(b, 512, 128, 22, "graph-gqa-f16-full")
}
func BenchmarkBatchedGraph_b768_gqaf16_full(b *testing.B) {
	benchBatchedGraph(b, 768, 128, 22, "graph-gqa-f16-full")
}

// Context sweep at 64-concurrent, full-step: attention cost grows with context while the GEMMs stay
// fixed, so the serving lead over vLLM erodes as ctx climbs. Pins where the win holds vs crosses over.
func BenchmarkBatchedGraph_b64_gqaf16_full_ctx256(b *testing.B) {
	benchBatchedGraph(b, 64, 256, 22, "graph-gqa-f16-full")
}
func BenchmarkBatchedGraph_b64_gqaf16_full_ctx512(b *testing.B) {
	benchBatchedGraph(b, 64, 512, 22, "graph-gqa-f16-full")
}
func BenchmarkBatchedGraph_b768_gqaf16(b *testing.B) {
	benchBatchedGraph(b, 768, 128, 22, "graph-gqa-f16")
}

// b1024/b1536 on the f16-KV path (half the KV footprint of MHA) to find the aggregate-throughput
// ceiling — where the decode GEMMs saturate the SMs and adding sequences stops buying tok/s.
// Measured plateau: b768 9195 | b1024 9191 | b1536 9296 tok/s (synthetic bgBuild, LAYERS-ONLY —
// no logits/sampling, so it OVERSTATES a full serving step). Ratio caveat (SPEC Iw8 honesty): the
// fair precision-matched vLLM decode baseline is ~6972 tok/s (both f16 activations, same session),
// NOT the handicapped ~4655 — so the honest standing is ~1.33x fair vLLM, not 2x. These points only
// pin the SM-saturation plateau; they do not move the honest headline.
func BenchmarkBatchedGraph_b1024_gqaf16(b *testing.B) {
	benchBatchedGraph(b, 1024, 128, 22, "graph-gqa-f16")
}
func BenchmarkBatchedGraph_b1536_gqaf16(b *testing.B) {
	benchBatchedGraph(b, 1536, 128, 22, "graph-gqa-f16")
}
