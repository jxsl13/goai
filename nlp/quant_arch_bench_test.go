package nlp_test

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// Prefill and decode benchmarks for every quantized transformer architecture.
//
// These exist because benchmark COVERAGE, not the rewrites, became the binding constraint on
// the nlp allocation sweep. Forward and DecodeStep are separate layer loops with separate
// call sites, so a per-layer change to one is invisible to a benchmark of the other — and an
// A/B across arms that never execute the changed code reports identical counts, which reads
// as "no effect" and means "no coverage". Before this file exactly one of the twelve
// architectures had both paths measurable.
//
// Deliberately small models (the shared GGUF test fixtures), because what is being measured
// is per-layer allocation count, which is geometry-independent. A production-sized model
// would multiply the runtime without changing the number these guard.
//
// Each entry supplies its own closures rather than going through an interface: the cache type
// differs per architecture (CohereCache, FalconCache, …) and the SSM families take a decode
// state instead, so there is no common shape to abstract over. Twelve small closures beat one
// wrong abstraction.

// quantArchBench pairs a prefill and a decode closure for one architecture.
type quantArchBench struct {
	name  string
	setup func(testing.TB) (fwd func() error, dec func() error, closeFn func())
}

func benchQuantArchPrefill(b *testing.B, tc quantArchBench) {
	fwd, _, closeFn := tc.setup(b)
	defer closeFn()
	b.ReportAllocs()
	for b.Loop() {
		if err := fwd(); err != nil {
			b.Fatal(err)
		}
	}
}

func benchQuantArchDecode(b *testing.B, tc quantArchBench) {
	_, dec, closeFn := tc.setup(b)
	defer closeFn()
	b.ReportAllocs()
	for b.Loop() {
		if err := dec(); err != nil {
			b.Fatal(err)
		}
	}
}

// quantBenchPrompt is short on purpose: the layer loop runs once per layer per token either
// way, so a longer prompt scales the runtime without changing the per-layer allocation count.
var quantBenchPrompt = []int{1, 3, 2, 5, 4}

var quantArchBenches = []quantArchBench{
	{"Cohere", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantCohereGGUFBytes(tb, newQuantTestCohere())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantCohereFromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"Falcon", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantFalconGGUFBytes(tb, newQuantTestFalcon())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantFalconFromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"Gemma", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantGemmaGGUFBytes(tb, newQuantTestGemma())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantGemmaFromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"Gemma2", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantGemma2GGUFBytes(tb, newQuantTestGemma2())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantGemma2FromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"GPTNeoX", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantGPTNeoXGGUFBytes(tb, newQuantTestGPTNeoX())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantGPTNeoXFromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"Mixtral", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantMixtralGGUFBytes(tb, newQuantTestMixtral())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantMixtralFromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"Nemotron", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantNemotronGGUFBytes(tb, newQuantTestNemotron())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantNemotronFromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"OLMo2", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantOLMo2GGUFBytes(tb, newQuantTestOLMo2())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantOLMo2FromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"StableLM", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantStableLMGGUFBytes(tb, newQuantTestStableLM())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantStableLMFromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"StarCoder2", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantStarCoder2GGUFBytes(tb, newQuantTestStarCoder2())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantStarCoder2FromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"MPT", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantMPTGGUFBytes(tb, newQuantTestMPT())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantMPTFromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
	{"DeepSeekV2", func(tb testing.TB) (func() error, func() error, func()) {
		raw := quantDeepSeekV2GGUFBytes(tb, newQuantTestDeepSeekV2())
		rf, err := gguf.ReadRaw(bytes.NewReader(raw))
		if err != nil {
			tb.Fatal(err)
		}
		q, err := nlp.QuantDeepSeekV2FromGGUF(rf.Metadata, rf.Tensors)
		if err != nil {
			tb.Fatal(err)
		}
		fwd := func() error {
			_, err := q.Forward(backend.NewContext(), quantBenchPrompt)
			return err
		}
		dec := func() error {
			ctx := backend.NewContext()
			cache := q.NewCache()
			for pos, tok := range quantBenchPrompt {
				if _, err := q.DecodeStep(ctx, cache, tok, pos); err != nil {
					return err
				}
			}
			return nil
		}
		return fwd, dec, func() { q.Close() }
	}},
}

func BenchmarkQuantArchPrefill(b *testing.B) {
	for _, tc := range quantArchBenches {
		b.Run(tc.name, func(b *testing.B) { benchQuantArchPrefill(b, tc) })
	}
}

func BenchmarkQuantArchDecode(b *testing.B) {
	for _, tc := range quantArchBenches {
		b.Run(tc.name, func(b *testing.B) { benchQuantArchDecode(b, tc) })
	}
}
