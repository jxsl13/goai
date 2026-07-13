//go:build darwin && cgo && vulkan

// Cross-backend comparison micro-benchmarks. This is the one build configuration
// (darwin + cgo + the vulkan tag) where every host-supported backend — the pure-Go
// reference, the SIMD cpu backend, Metal, and Vulkan — can be blank-imported into a
// single binary, so each op can be timed on all of them side by side.
//
// Run: make bench-compare   (sets the MoltenVK ICD + the vulkan tag)
// Output rows read Benchmark<Op>/<shape>/<backend>, e.g. BenchmarkMatMul/1024/vulkan.
// Backends that do not implement an op fall back to the reference transparently
// (§I4), so their row reflects the effective speed a caller selecting that backend
// gets. GFLOP/s is reported for matmul and conv; ns/op for attention.
package benchcompare

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/metal"
	_ "github.com/jxsl13/goai/backend/ref"
	_ "github.com/jxsl13/goai/backend/vulkan"
)

// compared is the descending-capability set of host backends to time each op on.
var compared = []backend.Name{backend.Ref, backend.CPU, backend.Metal, backend.Vulkan}

// eachBackend runs fn as a sub-benchmark per registered backend in `compared`.
func eachBackend(b *testing.B, fn func(b *testing.B, ctx *backend.Context)) {
	for _, name := range compared {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		ctx := backend.NewContext().WithBackend(be)
		b.Run(string(name), func(b *testing.B) {
			b.ResetTimer()
			fn(b, ctx)
		})
	}
}

func BenchmarkMatMul(b *testing.B) {
	for _, sz := range []int{256, 512, 1024} {
		x := bench.RandF32(tensor.Shape{sz, sz}, 1)
		y := bench.RandF32(tensor.Shape{sz, sz}, 2)
		ins := []*tensor.Tensor{x, y}
		b.Run(itoa(sz), func(b *testing.B) {
			eachBackend(b, func(b *testing.B, ctx *backend.Context) {
				for range b.N {
					if _, err := backend.Execute(ctx, backend.OpMatMul, ins, nil); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(2*float64(sz)*float64(sz)*float64(sz)/float64(b.Elapsed().Nanoseconds())*float64(b.N), "GFLOP/s")
			})
		})
	}
}

func BenchmarkMHAForward(b *testing.B) {
	const seq, heads, dk = 512, 8, 64
	dm := heads * dk
	q := bench.RandF32(tensor.Shape{seq, dm}, 1)
	k := bench.RandF32(tensor.Shape{seq, dm}, 2)
	v := bench.RandF32(tensor.Shape{seq, dm}, 3)
	attrs := backend.AttnAttrs{Heads: heads, Causal: true}
	ins := []*tensor.Tensor{q, k, v}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpMHA, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkMHABackward(b *testing.B) {
	const seq, heads, dk = 512, 8, 64
	dm := heads * dk
	q := bench.RandF32(tensor.Shape{seq, dm}, 1)
	k := bench.RandF32(tensor.Shape{seq, dm}, 2)
	v := bench.RandF32(tensor.Shape{seq, dm}, 3)
	g := bench.RandF32(tensor.Shape{seq, dm}, 4)
	attrs := backend.AttnAttrs{Heads: heads, Causal: true}
	ins := []*tensor.Tensor{q, k, v, g}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpMHABackward, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkFlashAttn times the fused FlashAttention-2 forward at the same shape as
// BenchmarkMHAForward, so the online-softmax kernel is directly comparable to the
// two-pass MHA kernel on every backend (§T337 tuned both).
func BenchmarkFlashAttn(b *testing.B) {
	const seq, heads, dk = 512, 8, 64
	dm := heads * dk
	q := bench.RandF32(tensor.Shape{seq, dm}, 1)
	k := bench.RandF32(tensor.Shape{seq, dm}, 2)
	v := bench.RandF32(tensor.Shape{seq, dm}, 3)
	attrs := backend.AttnAttrs{Heads: heads, Causal: true}
	ins := []*tensor.Tensor{q, k, v}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpFlashAttn, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// Retention is a per-head op (Q,K,V are [L,d]); L/d/γ follow the RetNet defaults
// used by the cross-reference tests.
func BenchmarkRetention(b *testing.B) {
	const L, d = 512, 64
	q := bench.RandF32(tensor.Shape{L, d}, 1)
	k := bench.RandF32(tensor.Shape{L, d}, 2)
	v := bench.RandF32(tensor.Shape{L, d}, 3)
	attrs := backend.RetentionAttrs{Gamma: 0.968}
	ins := []*tensor.Tensor{q, k, v}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpRetention, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkRetentionBackward(b *testing.B) {
	const L, d = 512, 64
	q := bench.RandF32(tensor.Shape{L, d}, 1)
	k := bench.RandF32(tensor.Shape{L, d}, 2)
	v := bench.RandF32(tensor.Shape{L, d}, 3)
	g := bench.RandF32(tensor.Shape{L, d}, 4)
	attrs := backend.RetentionAttrs{Gamma: 0.968}
	ins := []*tensor.Tensor{q, k, v, g}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpRetentionBackward, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// Softmax/RMSNorm are the memory-bound per-row ops the GPU buffer pool (§T335/§T336)
// affects most — their rows make pool regressions visible.
func BenchmarkSoftmax(b *testing.B) {
	const rows, dim = 2048, 2048
	x := bench.RandF32(tensor.Shape{rows, dim}, 1)
	ins := []*tensor.Tensor{x}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpSoftmax, ins, nil); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkRMSNorm(b *testing.B) {
	const rows, dim = 2048, 2048
	x := bench.RandF32(tensor.Shape{rows, dim}, 1)
	gamma := bench.RandF32(tensor.Shape{dim}, 2)
	attrs := backend.NormAttrs{Eps: 1e-5}
	ins := []*tensor.Tensor{x, gamma}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpRMSNorm, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkLayerNorm(b *testing.B) {
	const rows, dim = 2048, 2048
	x := bench.RandF32(tensor.Shape{rows, dim}, 1)
	gamma := bench.RandF32(tensor.Shape{dim}, 2)
	beta := bench.RandF32(tensor.Shape{dim}, 3)
	attrs := backend.NormAttrs{Eps: 1e-5}
	ins := []*tensor.Tensor{x, gamma, beta}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpLayerNorm, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// The norm backward kernels (§T347) take (x, gamma, upstream-g) and are hot in every
// training step; both accumulate cross-row parameter gradients via GPU float atomics.
func BenchmarkRMSNormBackward(b *testing.B) {
	const rows, dim = 2048, 2048
	x := bench.RandF32(tensor.Shape{rows, dim}, 1)
	gamma := bench.RandF32(tensor.Shape{dim}, 2)
	g := bench.RandF32(tensor.Shape{rows, dim}, 3)
	attrs := backend.NormAttrs{Eps: 1e-5}
	ins := []*tensor.Tensor{x, gamma, g}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpRMSNormBackward, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkLayerNormBackward(b *testing.B) {
	const rows, dim = 2048, 2048
	x := bench.RandF32(tensor.Shape{rows, dim}, 1)
	gamma := bench.RandF32(tensor.Shape{dim}, 2)
	g := bench.RandF32(tensor.Shape{rows, dim}, 3)
	attrs := backend.NormAttrs{Eps: 1e-5}
	ins := []*tensor.Tensor{x, gamma, g}
	eachBackend(b, func(b *testing.B, ctx *backend.Context) {
		for range b.N {
			if _, err := backend.Execute(ctx, backend.OpLayerNormBackward, ins, attrs); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkConv2D(b *testing.B) {
	// small = the original latency-probing shape; resnet = a ResNet-block-sized conv
	// where the GEMM lowering (§T341) has real arithmetic depth.
	cases := []struct {
		name           string
		n, c, hw, f, k int
	}{
		{"n8c16hw32", 8, 16, 32, 32, 3},
		{"n8c64hw56", 8, 64, 56, 64, 3},
	}
	for _, cs := range cases {
		x := bench.RandF32(tensor.Shape{cs.n, cs.c, cs.hw, cs.hw}, 1)
		w := bench.RandF32(tensor.Shape{cs.f, cs.c, cs.k, cs.k}, 2)
		attrs := backend.ConvAttrs{Stride: 1, Pad: 1}
		ins := []*tensor.Tensor{x, w}
		ho, wo := cs.hw, cs.hw // stride 1, pad 1, k 3 → same spatial size
		b.Run(cs.name, func(b *testing.B) {
			eachBackend(b, func(b *testing.B, ctx *backend.Context) {
				for range b.N {
					if _, err := backend.Execute(ctx, backend.OpConv2D, ins, attrs); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(2*float64(cs.n)*float64(cs.f)*float64(ho)*float64(wo)*float64(cs.c)*float64(cs.k)*float64(cs.k)/float64(b.Elapsed().Nanoseconds())*float64(b.N), "GFLOP/s")
			})
		})
	}
}

// itoa avoids importing strconv just for a benchmark label.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// randGPT builds a realistically-sized GPT with deterministic random f32 weights
// (norms γ=1/β=0, projections small) for the end-to-end throughput benchmarks — the
// real workload (§C3/§B10), across every host backend (§T355).
func randGPT(b *testing.B, cfg nlp.GPTConfig, ffn int) *nlp.GPT {
	rng := rand.New(rand.NewPCG(1, 2))
	small := func(shape ...int) *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape(shape))
		d := t.Storage().F32()
		for i := range d {
			d[i] = float32(rng.NormFloat64()) * 0.02
		}
		return t
	}
	ones := func(n int) *tensor.Tensor {
		t := tensor.New(tensor.F32, tensor.Shape{n})
		d := t.Storage().F32()
		for i := range d {
			d[i] = 1
		}
		return t
	}
	zeros := func(n int) *tensor.Tensor { return tensor.New(tensor.F32, tensor.Shape{n}) }
	ts := map[string]*tensor.Tensor{
		"tok_emb": small(cfg.Vocab, cfg.Dim), "pos_emb": small(cfg.Ctx, cfg.Dim),
		"head": small(cfg.Dim, cfg.Vocab), "lnf.gamma": ones(cfg.Dim), "lnf.beta": zeros(cfg.Dim),
	}
	for l := range cfg.Layers {
		p := func(s string) string { return "blocks." + itoa(l) + "." + s }
		ts[p("ln1.gamma")] = ones(cfg.Dim)
		ts[p("ln1.beta")] = zeros(cfg.Dim)
		ts[p("attn.wq")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wk")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wv")] = small(cfg.Dim, cfg.Dim)
		ts[p("attn.wo")] = small(cfg.Dim, cfg.Dim)
		ts[p("ln2.gamma")] = ones(cfg.Dim)
		ts[p("ln2.beta")] = zeros(cfg.Dim)
		ts[p("ffn.w1")] = small(cfg.Dim, ffn)
		ts[p("ffn.b1")] = zeros(ffn)
		ts[p("ffn.w2")] = small(ffn, cfg.Dim)
		ts[p("ffn.b2")] = zeros(cfg.Dim)
	}
	model, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		b.Fatal(err)
	}
	return model
}

// gptBackends is the set the GPT benchmarks time (the reference is far too slow for a
// realistically-sized model, so it is omitted — cpu is the Pure-Go baseline).
var gptBackends = []backend.Name{backend.CPU, backend.Metal, backend.Vulkan}

// BenchmarkGPTForward / BenchmarkGPTTrainingStep time a full transformer forward (and a
// forward+backward) on cpu/metal/vulkan — the real workload across every backend (§T355).
func BenchmarkGPTForward(b *testing.B) {
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	const seq = 256
	model := randGPT(b, cfg, 4*cfg.Dim)
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = i % cfg.Vocab
	}
	for _, name := range gptBackends {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		ctx := backend.NewContext().WithBackend(be)
		if _, err := model.Forward(ctx, tokens); err != nil {
			b.Fatal(err)
		}
		b.Run(string(name), func(b *testing.B) {
			for range b.N {
				if _, err := model.Forward(ctx, tokens); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}

func BenchmarkGPTTrainingStep(b *testing.B) {
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	const seq = 256
	model := randGPT(b, cfg, 4*cfg.Dim)
	tokens := make([]int, seq)
	targets := tensor.New(tensor.F32, tensor.Shape{seq})
	for i := range tokens {
		tokens[i] = i % cfg.Vocab
		targets.SetF64(float64(i%cfg.Vocab), i)
	}
	step := func(be backend.Backend) {
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		logits, err := model.Forward(ctx, tokens)
		if err != nil {
			b.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			b.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			b.Fatal(err)
		}
	}
	for _, name := range gptBackends {
		be, ok := backend.Get(name)
		if !ok {
			continue
		}
		step(be)
		b.Run(string(name), func(b *testing.B) {
			for range b.N {
				step(be)
			}
			b.ReportMetric(float64(seq)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
		})
	}
}
