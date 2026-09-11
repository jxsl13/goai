package benchcompare

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"math"
	"math/rand/v2"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// cpuAttentionBenchmarkArm is set at link time for attribution binaries:
//
//	-ldflags=-X=github.com/jxsl13/goai/internal/benchcompare.cpuAttentionBenchmarkArm=baseline
//	-ldflags=-X=github.com/jxsl13/goai/internal/benchcompare.cpuAttentionBenchmarkArm=force-off
//
// Both arms use cpuAttentionAttributionBackend, including its kernel wrapper and
// interception counter. The force-off arm deliberately returns zero gradients
// with the right shapes. It is INVALID OUTPUT: the direct MHA row is a clean bound
// on removable backward work, while its GPT row is only a loose bound because the
// zero attention gradients change all downstream backward values. The default
// direct arm uses the registered CPU backend with its normal dispatch cache; it
// is the promotion benchmark for correctness-valid baseline/candidate sources.
var cpuAttentionBenchmarkArm = "direct"

var (
	cpuAttentionBenchmarkTensorSink *tensor.Tensor
	cpuAttentionBenchmarkGradSink   *tensor.Tensor
)

type cpuAttentionAttributionBackend struct {
	inner    backend.Backend
	forceOff bool
	backward atomic.Uint64
}

func (b *cpuAttentionAttributionBackend) Name() backend.Name    { return b.inner.Name() }
func (b *cpuAttentionAttributionBackend) Device() tensor.Device { return b.inner.Device() }
func (b *cpuAttentionAttributionBackend) Synchronize() error    { return b.inner.Synchronize() }

func (b *cpuAttentionAttributionBackend) Kernel(op backend.Op, dtype tensor.Dtype) (backend.Kernel, bool) {
	k, ok := b.inner.Kernel(op, dtype)
	if !ok || op != backend.OpMHABackward {
		return k, ok
	}
	return func(ctx *backend.Context, inputs []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
		b.backward.Add(1)
		if b.forceOff {
			// INVALID OUTPUT attribution control. Keep output arity, dtype and shapes
			// identical to the real dQ/dK/dV contract so downstream VJPs still run.
			return []*tensor.Tensor{
				tensor.ZerosLike(inputs[0]),
				tensor.ZerosLike(inputs[1]),
				tensor.ZerosLike(inputs[2]),
			}, nil
		}
		return k(ctx, inputs, attrs)
	}, true
}

func cpuAttentionBackend(tb testing.TB, forceOff bool) *cpuAttentionAttributionBackend {
	tb.Helper()
	be, ok := backend.Get(backend.CPU)
	if !ok {
		tb.Fatal("cpu backend is not registered")
	}
	return &cpuAttentionAttributionBackend{inner: be, forceOff: forceOff}
}

func cpuAttentionSelectedArm(tb testing.TB) (backend.Backend, *cpuAttentionAttributionBackend, string) {
	tb.Helper()
	switch cpuAttentionBenchmarkArm {
	case "direct":
		be, ok := backend.Get(backend.CPU)
		if !ok {
			tb.Fatal("cpu backend is not registered")
		}
		return be, nil, "direct"
	case "baseline":
		wrapped := cpuAttentionBackend(tb, false)
		return wrapped, wrapped, "baseline_attribution-wrapper"
	case "force-off":
		wrapped := cpuAttentionBackend(tb, true)
		return wrapped, wrapped, "force-off_INVALID_OUTPUT"
	default:
		tb.Fatalf("unknown cpu attention benchmark arm %q (want direct, baseline, or force-off)", cpuAttentionBenchmarkArm)
		return nil, nil, ""
	}
}

type cpuAttentionMHACase struct {
	name                  string
	dtype                 tensor.Dtype
	seq, heads, kv, width int
	causal                bool
}

func cpuAttentionMHAInputs(c cpuAttentionMHACase) []*tensor.Tensor {
	qShape := tensor.Shape{c.seq, c.heads * c.width}
	kvShape := tensor.Shape{c.seq, c.kv * c.width}
	if c.dtype == tensor.F64 {
		return []*tensor.Tensor{
			bench.RandF64(qShape, 1),
			bench.RandF64(kvShape, 2),
			bench.RandF64(kvShape, 3),
			bench.RandF64(qShape, 4),
		}
	}
	return []*tensor.Tensor{
		bench.RandF32(qShape, 1),
		bench.RandF32(kvShape, 2),
		bench.RandF32(kvShape, 3),
		bench.RandF32(qShape, 4),
	}
}

func BenchmarkCPUAttentionBackward(b *testing.B) {
	cases := []cpuAttentionMHACase{
		{"f32_causal_s128_h8_kv8_d64", tensor.F32, 128, 8, 8, 64, true},
		{"f32_causal_s256_h8_kv8_d64", tensor.F32, 256, 8, 8, 64, true},
		{"f32_causal_s512_h8_kv8_d64", tensor.F32, 512, 8, 8, 64, true},
		{"f32_causal_s1024_h8_kv8_d64", tensor.F32, 1024, 8, 8, 64, true},
		{"f32_noncausal_s512_h8_kv8_d64", tensor.F32, 512, 8, 8, 64, false},
		{"f32_causal_gqa_s128_h4_kv2_d16", tensor.F32, 128, 4, 2, 16, true},
		{"f64_causal_s128_h8_kv8_d64", tensor.F64, 128, 8, 8, 64, true},
	}
	for _, c := range cases {
		be, wrapped, arm := cpuAttentionSelectedArm(b)
		b.Run(c.name+"/"+arm, func(b *testing.B) {
			inputs := cpuAttentionMHAInputs(c)
			kvHeads := c.kv
			if kvHeads == c.heads {
				kvHeads = 0 // preserve compare_test.go's default-MHA attribute path
			}
			attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: kvHeads, Causal: c.causal}
			ctx := backend.NewContext().WithBackend(be)
			out, err := backend.Execute(ctx, backend.OpMHABackward, inputs, attrs) // warmup
			if err != nil {
				b.Fatal(err)
			}
			cpuAttentionBenchmarkGradSink = out[0]
			if wrapped != nil {
				wrapped.backward.Store(0)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				out, err = backend.Execute(ctx, backend.OpMHABackward, inputs, attrs)
				if err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			cpuAttentionBenchmarkGradSink = out[0]
			if wrapped != nil {
				if got, want := wrapped.backward.Load(), uint64(b.N); got != want {
					b.Fatalf("MHA backward interceptions = %d, want %d", got, want)
				}
			}
		})
	}
}

func BenchmarkCPUAttentionForward(b *testing.B) {
	const seq, heads, width = 512, 8, 64
	q := bench.RandF32(tensor.Shape{seq, heads * width}, 1)
	k := bench.RandF32(tensor.Shape{seq, heads * width}, 2)
	v := bench.RandF32(tensor.Shape{seq, heads * width}, 3)
	inputs := []*tensor.Tensor{q, k, v}
	attrs := backend.AttnAttrs{Heads: heads, Causal: true}
	be, wrapped, _ := cpuAttentionSelectedArm(b)
	ctx := backend.NewContext().WithBackend(be)
	out, err := backend.Execute(ctx, backend.OpMHA, inputs, attrs) // warmup
	if err != nil {
		b.Fatal(err)
	}
	cpuAttentionBenchmarkTensorSink = out[0]

	b.Run("f32_causal_s512_h8_kv8_d64/control", func(b *testing.B) {
		if wrapped != nil {
			wrapped.backward.Store(0)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			out, err = backend.Execute(ctx, backend.OpMHA, inputs, attrs)
			if err != nil {
				b.Fatal(err)
			}
		}
		b.StopTimer()
		cpuAttentionBenchmarkTensorSink = out[0]
		if wrapped != nil {
			if got := wrapped.backward.Load(); got != 0 {
				b.Fatalf("forward control intercepted %d backward ops", got)
			}
		}
	})
}

type cpuAttentionGPTFill uint8

const (
	cpuAttentionGPTSmall cpuAttentionGPTFill = iota
	cpuAttentionGPTOnes
	cpuAttentionGPTZeros
)

type cpuAttentionGPTEntry struct {
	name  string
	fill  cpuAttentionGPTFill
	shape tensor.Shape
}

// cpuAttentionGPTEntries preserves the exact random evaluation order of
// compare_test.go's randGPT: tok, pos, head, final norm, then each block's
// norms/projections/FFN in source order. The helper is portable and uniquely
// named so it can coexist in builds that also include compare_test.go.
func cpuAttentionGPTEntries(cfg nlp.GPTConfig, ffn int) []cpuAttentionGPTEntry {
	entries := []cpuAttentionGPTEntry{
		{"tok_emb", cpuAttentionGPTSmall, tensor.Shape{cfg.Vocab, cfg.Dim}},
		{"pos_emb", cpuAttentionGPTSmall, tensor.Shape{cfg.Ctx, cfg.Dim}},
		{"head", cpuAttentionGPTSmall, tensor.Shape{cfg.Dim, cfg.Vocab}},
		{"lnf.gamma", cpuAttentionGPTOnes, tensor.Shape{cfg.Dim}},
		{"lnf.beta", cpuAttentionGPTZeros, tensor.Shape{cfg.Dim}},
	}
	for l := range cfg.Layers {
		p := "blocks." + strconv.Itoa(l) + "."
		entries = append(entries,
			cpuAttentionGPTEntry{p + "ln1.gamma", cpuAttentionGPTOnes, tensor.Shape{cfg.Dim}},
			cpuAttentionGPTEntry{p + "ln1.beta", cpuAttentionGPTZeros, tensor.Shape{cfg.Dim}},
			cpuAttentionGPTEntry{p + "attn.wq", cpuAttentionGPTSmall, tensor.Shape{cfg.Dim, cfg.Dim}},
			cpuAttentionGPTEntry{p + "attn.wk", cpuAttentionGPTSmall, tensor.Shape{cfg.Dim, cfg.Dim}},
			cpuAttentionGPTEntry{p + "attn.wv", cpuAttentionGPTSmall, tensor.Shape{cfg.Dim, cfg.Dim}},
			cpuAttentionGPTEntry{p + "attn.wo", cpuAttentionGPTSmall, tensor.Shape{cfg.Dim, cfg.Dim}},
			cpuAttentionGPTEntry{p + "ln2.gamma", cpuAttentionGPTOnes, tensor.Shape{cfg.Dim}},
			cpuAttentionGPTEntry{p + "ln2.beta", cpuAttentionGPTZeros, tensor.Shape{cfg.Dim}},
			cpuAttentionGPTEntry{p + "ffn.w1", cpuAttentionGPTSmall, tensor.Shape{cfg.Dim, ffn}},
			cpuAttentionGPTEntry{p + "ffn.b1", cpuAttentionGPTZeros, tensor.Shape{ffn}},
			cpuAttentionGPTEntry{p + "ffn.w2", cpuAttentionGPTSmall, tensor.Shape{ffn, cfg.Dim}},
			cpuAttentionGPTEntry{p + "ffn.b2", cpuAttentionGPTZeros, tensor.Shape{cfg.Dim}},
		)
	}
	return entries
}

func cpuAttentionGPTTensors(cfg nlp.GPTConfig, ffn int) (map[string]*tensor.Tensor, []cpuAttentionGPTEntry) {
	rng := rand.New(rand.NewPCG(1, 2))
	entries := cpuAttentionGPTEntries(cfg, ffn)
	ts := make(map[string]*tensor.Tensor, len(entries))
	for _, entry := range entries {
		var t *tensor.Tensor
		switch entry.fill {
		case cpuAttentionGPTSmall:
			t = tensor.New(tensor.F32, entry.shape)
			for i := range t.Storage().F32() {
				t.Storage().F32()[i] = float32(rng.NormFloat64()) * 0.02
			}
		case cpuAttentionGPTOnes:
			t = tensor.Ones(tensor.F32, entry.shape)
		case cpuAttentionGPTZeros:
			t = tensor.Zeros(tensor.F32, entry.shape)
		default:
			panic("unknown GPT fixture fill")
		}
		ts[entry.name] = t
	}
	return ts, entries
}

func cpuAttentionRandomGPT(tb testing.TB, cfg nlp.GPTConfig, ffn int) *nlp.GPT {
	tb.Helper()
	ts, _ := cpuAttentionGPTTensors(cfg, ffn)
	model, err := nlp.FromSafetensors(cfg, ts)
	if err != nil {
		tb.Fatal(err)
	}
	return model
}

func cpuAttentionGPTObjective(cfg nlp.GPTConfig) ([]int, *tensor.Tensor) {
	tokens := make([]int, cfg.Ctx)
	targets := tensor.New(tensor.F32, tensor.Shape{cfg.Ctx})
	for i := range tokens {
		tokens[i] = i % cfg.Vocab
		targets.SetF64(float64(i%cfg.Vocab), i)
	}
	return tokens, targets
}

func cpuAttentionGPTStep(tb testing.TB, model *nlp.GPT, be backend.Backend, tokens []int, targets *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	tb.Helper()
	tape := autograd.NewTapeOn(be)
	logits, err := model.Forward(tape.Context(), tokens)
	if err != nil {
		tb.Fatal(err)
	}
	loss, err := nn.CrossEntropy(tape.Context(), logits, targets)
	if err != nil {
		tb.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		tb.Fatal(err)
	}
	return loss, tape.Grad(model.Head)
}

func BenchmarkCPUGPTTrainingStep(b *testing.B) {
	cfg := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	const ffn = 2048
	model := cpuAttentionRandomGPT(b, cfg, ffn)
	tokens, targets := cpuAttentionGPTObjective(cfg)
	be, wrapped, arm := cpuAttentionSelectedArm(b)

	loss, grad := cpuAttentionGPTStep(b, model, be, tokens, targets) // whole-step warmup
	cpuAttentionBenchmarkTensorSink, cpuAttentionBenchmarkGradSink = loss, grad
	if wrapped != nil {
		if got := wrapped.backward.Load(); got != uint64(cfg.Layers) {
			b.Fatalf("warmup MHA backward interceptions = %d, want %d", got, cfg.Layers)
		}
	}

	b.Run(arm, func(b *testing.B) {
		if wrapped != nil {
			wrapped.backward.Store(0)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			loss, grad = cpuAttentionGPTStep(b, model, be, tokens, targets)
		}
		b.StopTimer()
		cpuAttentionBenchmarkTensorSink, cpuAttentionBenchmarkGradSink = loss, grad
		if wrapped != nil {
			if got, want := wrapped.backward.Load(), uint64(b.N*cfg.Layers); got != want {
				b.Fatalf("GPT MHA backward interceptions = %d, want %d (six per whole step)", got, want)
			}
		}
		b.ReportMetric(float64(cfg.Ctx*b.N)/b.Elapsed().Seconds(), "tok/s")
	})
}

func cpuAttentionWriteInt(h hash.Hash, v int) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(v))
	_, _ = h.Write(buf[:])
}

func cpuAttentionTensorDigest(t *tensor.Tensor) string {
	h := sha256.New()
	cpuAttentionWriteInt(h, int(t.Dtype()))
	for _, dim := range t.Shape() {
		cpuAttentionWriteInt(h, dim)
	}
	var buf [8]byte
	switch t.Dtype() {
	case tensor.F32:
		for _, v := range t.Storage().F32() {
			binary.LittleEndian.PutUint32(buf[:4], math.Float32bits(v))
			_, _ = h.Write(buf[:4])
		}
	case tensor.F64:
		for _, v := range t.Storage().F64() {
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(v))
			_, _ = h.Write(buf[:])
		}
	default:
		panic("cpu attention digest only supports floating tensors")
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func cpuAttentionGPTPlanDigest(cfg nlp.GPTConfig, ffn int) string {
	h := sha256.New()
	fmt.Fprintf(h, "pcg=1,2;scale=0.02;vocab=%d;ctx=%d;dim=%d;heads=%d;layers=%d;eps=%g;ffn=%d;tokens=targets=i%%vocab;",
		cfg.Vocab, cfg.Ctx, cfg.Dim, cfg.Heads, cfg.Layers, cfg.Eps, ffn)
	for _, entry := range cpuAttentionGPTEntries(cfg, ffn) {
		fmt.Fprintf(h, "%s:%d:", entry.name, entry.fill)
		for _, dim := range entry.shape {
			fmt.Fprintf(h, "%d,", dim)
		}
		_, _ = h.Write([]byte(";"))
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func cpuAttentionGPTWeightsDigest(ts map[string]*tensor.Tensor, entries []cpuAttentionGPTEntry) string {
	h := sha256.New()
	for _, entry := range entries {
		_, _ = h.Write([]byte(entry.name))
		_, _ = h.Write([]byte(cpuAttentionTensorDigest(ts[entry.name])))
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func cpuAttentionAssertFiniteNontrivial(tb testing.TB, name string, t *tensor.Tensor) {
	tb.Helper()
	if t == nil {
		tb.Fatalf("%s is nil", name)
	}
	nonzero := false
	for i := range t.Numel() {
		idx := tensor.Unravel(i, t.Shape())
		v := t.AtF64(idx...)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			tb.Fatalf("%s[%d] is non-finite: %v", name, i, v)
		}
		if v != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		tb.Fatalf("%s is identically zero", name)
	}
}

func TestCPUAttentionBenchmarkFixtureDigest(t *testing.T) {
	// This cheap manifest locks the exact full timed workload without executing a
	// 90+ MB six-layer model in every ordinary test sweep.
	full := nlp.GPTConfig{Vocab: 4096, Ctx: 256, Dim: 512, Heads: 8, Layers: 6, Eps: 1e-5}
	if got, want := cpuAttentionGPTPlanDigest(full, 2048), "fe17fc805057eb20a8263478c3046123fc7942ec598353b6b9b3f2046c17fae7"; got != want {
		t.Fatalf("full GPT benchmark plan digest = %s, want %s", got, want)
	}

	// The small fixture uses the identical constructor and random evaluation
	// order, making a raw-value digest affordable in the normal correctness lane.
	small := nlp.GPTConfig{Vocab: 32, Ctx: 8, Dim: 16, Heads: 4, Layers: 2, Eps: 1e-5}
	ts, entries := cpuAttentionGPTTensors(small, 64)
	if got, want := cpuAttentionGPTWeightsDigest(ts, entries), "bf37baca395fb1e66588e2692d0bc209da231ccfc6e3cd96c47429d7134e0d17"; got != want {
		t.Fatalf("small GPT fixture digest = %s, want %s", got, want)
	}

	// The MHA fixture remains byte-for-byte tied to compare_test.go's RandF32
	// seeds 1/2/3/4 at its published 512×(8·64) geometry.
	c := cpuAttentionMHACase{"f32_causal_s512_h8_kv8_d64", tensor.F32, 512, 8, 8, 64, true}
	inputs := cpuAttentionMHAInputs(c)
	h := sha256.New()
	for _, in := range inputs {
		_, _ = h.Write([]byte(cpuAttentionTensorDigest(in)))
	}
	if got, want := fmt.Sprintf("%x", h.Sum(nil)), "232facda465b02918ce03cb84f71289ce2529faa9c895c2a8d310b154d7925e0"; got != want {
		t.Fatalf("MHA fixture digest = %s, want %s", got, want)
	}
}

func TestCPUAttentionAttributionArmsMatched(t *testing.T) {
	c := cpuAttentionMHACase{"sanity", tensor.F32, 16, 4, 4, 8, true}
	inputs := cpuAttentionMHAInputs(c)
	before := make([]string, len(inputs))
	for i, in := range inputs {
		before[i] = cpuAttentionTensorDigest(in)
	}
	attrs := backend.AttnAttrs{Heads: c.heads, KVHeads: c.kv, Causal: c.causal}

	for _, tc := range []struct {
		name     string
		forceOff bool
	}{
		{"baseline", false},
		{"force-off_INVALID_OUTPUT", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			be := cpuAttentionBackend(t, tc.forceOff)
			ctx := backend.NewContext().WithBackend(be)
			forward, err := backend.Execute(ctx, backend.OpMHA, inputs[:3], attrs)
			if err != nil {
				t.Fatal(err)
			}
			cpuAttentionAssertFiniteNontrivial(t, "attention output", forward[0])

			grads, err := backend.Execute(ctx, backend.OpMHABackward, inputs, attrs)
			if err != nil {
				t.Fatal(err)
			}
			if len(grads) != 3 {
				t.Fatalf("MHA backward returned %d gradients, want 3", len(grads))
			}
			if got := be.backward.Load(); got != 1 {
				t.Fatalf("MHA backward interceptions = %d, want 1", got)
			}
			for i, grad := range grads {
				if !grad.Shape().Equal(inputs[i].Shape()) || grad.Dtype() != inputs[i].Dtype() {
					t.Fatalf("gradient %d has %v/%v, want %v/%v", i, grad.Shape(), grad.Dtype(), inputs[i].Shape(), inputs[i].Dtype())
				}
				if tc.forceOff {
					for j, v := range grad.Storage().F32() {
						if v != 0 {
							t.Fatalf("INVALID OUTPUT force-off gradient %d[%d] = %v, want zero", i, j, v)
						}
					}
				} else {
					cpuAttentionAssertFiniteNontrivial(t, fmt.Sprintf("gradient %d", i), grad)
				}
			}
			for i, in := range inputs {
				if got := cpuAttentionTensorDigest(in); got != before[i] {
					t.Fatalf("input %d mutated: digest %s, want %s", i, got, before[i])
				}
			}
		})
	}
}

func TestCPUGPTTrainingStepSanity(t *testing.T) {
	// Six layers preserve the exact interception cardinality of the timed GPT,
	// while the reduced other dimensions keep this sanity test inexpensive.
	cfg := nlp.GPTConfig{Vocab: 32, Ctx: 8, Dim: 16, Heads: 4, Layers: 6, Eps: 1e-5}
	for _, tc := range []struct {
		name     string
		forceOff bool
	}{
		{"baseline", false},
		{"force-off_INVALID_OUTPUT_loose-bound", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := cpuAttentionRandomGPT(t, cfg, 64)
			tokens, targets := cpuAttentionGPTObjective(cfg)
			paramBefore := make([]string, len(model.Params()))
			for i, p := range model.Params() {
				paramBefore[i] = cpuAttentionTensorDigest(p)
			}
			targetBefore := cpuAttentionTensorDigest(targets)
			be := cpuAttentionBackend(t, tc.forceOff)
			loss, grad := cpuAttentionGPTStep(t, model, be, tokens, targets)
			cpuAttentionAssertFiniteNontrivial(t, "loss", loss)
			if !tc.forceOff {
				cpuAttentionAssertFiniteNontrivial(t, "head gradient", grad)
			}
			if got := be.backward.Load(); got != 6 {
				t.Fatalf("whole GPT step MHA backward interceptions = %d, want 6", got)
			}
			for i, p := range model.Params() {
				if got := cpuAttentionTensorDigest(p); got != paramBefore[i] {
					t.Fatalf("model parameter %d mutated", i)
				}
			}
			if got := cpuAttentionTensorDigest(targets); got != targetBefore {
				t.Fatal("targets mutated")
			}
			for i, token := range tokens {
				if token != i%cfg.Vocab {
					t.Fatalf("token %d mutated to %d", i, token)
				}
			}
		})
	}
}
