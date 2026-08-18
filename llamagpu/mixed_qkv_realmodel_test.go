//go:build darwin && cgo

package llamagpu

import (
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nlp"
)

// TestMetalMixedQKVRealModelQualityAndSpeed is the trained-model promotion gate for heterogeneous
// f16-expanded QKV. It is opt-in because it repeatedly constructs and warms a 1.1B decoder. Every
// timed arm is fresh and differs only in groupQWeights; both use the shipping f16-KV cache.
func TestMetalMixedQKVRealModelQualityAndSpeed(t *testing.T) {
	if os.Getenv("GOAI_MIXED_QKV_REAL") != "1" {
		t.Skip("set GOAI_MIXED_QKV_REAL=1 for the trained TinyLlama promotion campaign")
	}
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	path := os.Getenv("GOAI_TINYLLAMA_GGUF")
	if path == "" {
		path = "../models/tinyllama-1.1b-q4km.gguf"
	}
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("model not present: %v", err)
	}
	defer f.Close()
	raw, err := gguf.ReadRaw(f)
	if err != nil {
		t.Fatal(err)
	}
	model, err := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if model.Config.Ctx < 600 {
		t.Fatalf("model context=%d, need at least 600", model.Config.Ctx)
	}
	tok, err := nlp.UnigramFromGGUF(raw.Metadata)
	if err != nil {
		t.Fatal(err)
	}
	prompt := append([]int{1}, tok.Encode("Explain in one sentence why the sky is blue.")...)

	resetCache := func() {
		metal.SetWeightCacheGB(0)
		metal.SetWeightCacheGB(4)
	}
	t.Cleanup(func() {
		metal.SetWeightCacheGB(0)
		metal.SetWeightCacheGB(4)
	})
	type quality struct {
		tokens []int
		logits []float32
		groups int
	}
	qualityArm := func(candidate bool) quality {
		resetCache()
		d, err := newQuantMetalWithMixedQKV(model, true, candidate)
		if err != nil {
			t.Fatal(err)
		}
		groups := 0
		for _, b := range d.blocks {
			if b.wqkvPrefill != nil {
				groups++
			}
		}
		tokens, err := d.Generate(prompt, 64, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		logits, err := d.StepNLast(prompt, 0)
		if err != nil {
			t.Fatal(err)
		}
		result := quality{tokens: slices.Clone(tokens), logits: slices.Clone(logits), groups: groups}
		d.Release()
		metal.SetWeightCacheGB(0)
		return result
	}
	controlQuality := qualityArm(false)
	candidateQuality := qualityArm(true)
	if controlQuality.groups != 0 || candidateQuality.groups != 10 {
		t.Fatalf("mixed selector groups: control=%d candidate=%d, want 0/10", controlQuality.groups, candidateQuality.groups)
	}
	if !slices.Equal(candidateQuality.tokens, controlQuality.tokens) {
		for i := range min(len(candidateQuality.tokens), len(controlQuality.tokens)) {
			if candidateQuality.tokens[i] != controlQuality.tokens[i] {
				t.Fatalf("mixed QKV changed greedy token %d: candidate=%d control=%d", i, candidateQuality.tokens[i], controlQuality.tokens[i])
			}
		}
		t.Fatalf("mixed/control greedy lengths differ: %d/%d", len(candidateQuality.tokens), len(controlQuality.tokens))
	}
	var diff2, ref2, maxAbs float64
	for i, got32 := range candidateQuality.logits {
		got, want := float64(got32), float64(controlQuality.logits[i])
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Fatalf("candidate logit[%d] is non-finite: %g", i, got)
		}
		d := got - want
		diff2 += d * d
		ref2 += want * want
		maxAbs = max(maxAbs, math.Abs(d))
	}
	nrmse := math.Sqrt(diff2 / math.Max(ref2, 1e-30))
	t.Logf("trained mixed-QKV quality: groups 0/10, %d greedy tokens unchanged, logit NRMSE %.6g maxAbs %.6g",
		len(candidateQuality.tokens), nrmse, maxAbs)
	if nrmse > 2e-3 {
		t.Fatalf("trained-model logit NRMSE %.6g exceeds 2e-3", nrmse)
	}

	type metrics struct{ pp64, pp512, tg64 float64 }
	makePrompt := func(n int) []int {
		p := make([]int, n)
		for i := range p {
			p[i] = 1 + (i*131)%min(model.Config.Vocab-1, 30000)
		}
		return p
	}
	measureArm := func(candidate bool) metrics {
		resetCache()
		d, err := newQuantMetalWithMixedQKV(model, true, candidate)
		if err != nil {
			t.Fatal(err)
		}
		measurePrefill := func(n int) float64 {
			p := makePrompt(n)
			if _, err := d.StepNLast(p, 0); err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			if _, err := d.StepNLast(p, 0); err != nil {
				t.Fatal(err)
			}
			return float64(n) / time.Since(start).Seconds()
		}
		pp64 := measurePrefill(64)
		pp512 := measurePrefill(512)
		p := makePrompt(64)
		if _, err := d.StepNLast(p, 0); err != nil {
			t.Fatal(err)
		}
		for i := range 3 {
			if _, err := d.Step(1+(i*17)%1000, 64+i); err != nil {
				t.Fatal(err)
			}
		}
		start := time.Now()
		for i := range 64 {
			if _, err := d.Step(1+(i*37)%1000, 67+i); err != nil {
				t.Fatal(err)
			}
		}
		tg64 := 64 / time.Since(start).Seconds()
		d.Release()
		metal.SetWeightCacheGB(0)
		return metrics{pp64: pp64, pp512: pp512, tg64: tg64}
	}
	const campaigns = 3
	control, candidate := make([]metrics, 0, campaigns), make([]metrics, 0, campaigns)
	for i := range campaigns {
		if i%2 == 0 {
			control = append(control, measureArm(false))
			candidate = append(candidate, measureArm(true))
		} else {
			candidate = append(candidate, measureArm(true))
			control = append(control, measureArm(false))
		}
	}
	median := func(v []float64) float64 {
		sort.Float64s(v)
		return v[len(v)/2]
	}
	gate := func(name string, get func(metrics) float64, minimum float64) {
		baseSamples, candidateSamples := make([]float64, campaigns), make([]float64, campaigns)
		for i := range campaigns {
			baseSamples[i] = get(control[i])
			candidateSamples[i] = get(candidate[i])
		}
		base, got := median(slices.Clone(baseSamples)), median(slices.Clone(candidateSamples))
		ratio := got / base
		controlSpread := slices.Max(baseSamples) / slices.Min(baseSamples)
		fmt.Printf("MIXEDQKV_REAL %s separate=%.3f grouped=%.3f ratio=%.4fx control_spread=%.4fx separate_samples=%v grouped_samples=%v\n",
			name, base, got, ratio, controlSpread, baseSamples, candidateSamples)
		if ratio < minimum {
			t.Errorf("%s grouped/separate %.4fx is below %.2fx", name, ratio, minimum)
		}
		if controlSpread > 1.15 {
			t.Errorf("%s unchanged-control spread %.4fx exceeds 1.15x", name, controlSpread)
		}
	}
	gate("pp64_tok_s", func(m metrics) float64 { return m.pp64 }, 1.03)
	gate("pp512_tok_s", func(m metrics) float64 { return m.pp512 }, 0.99)
	gate("tg64_tok_s", func(m metrics) float64 { return m.tg64 }, 0.99)
}
