//go:build darwin && cgo

package llamagpu_test

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
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestMetalF16KVRealModelQualityAndSpeed is the promotion gate for the opt-in M2 cache path.
// It uses a trained TinyLlama Q4_K_M checkpoint, not a random-weight proxy: the representative
// text prompt must keep every greedy token, logits must stay finite with small normalized error,
// and three interleaved decode-only campaigns must clear the short/long-context leverage gates.
// The checkpoint is external and therefore skipped in hermetic CI; GOAI_TINYLLAMA_GGUF pins it.
func TestMetalF16KVRealModelQualityAndSpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("1.1B model; skipped in -short")
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
	if model.Config.Dim/model.Config.Heads != 64 {
		t.Fatalf("fixture head dimension=%d, want 64", model.Config.Dim/model.Config.Heads)
	}
	tok, err := nlp.UnigramFromGGUF(raw.Metadata)
	if err != nil {
		t.Fatal(err)
	}
	prompt := append([]int{1}, tok.Encode("Explain in one sentence why the sky is blue.")...)
	if len(prompt) < 2 {
		t.Fatalf("representative prompt encoded to %v", prompt)
	}

	f32, err := llamagpu.NewQuant(model)
	if err != nil {
		t.Fatal(err)
	}
	defer f32.Release()
	f16, err := llamagpu.NewQuantF16KV(model)
	if err != nil {
		t.Fatal(err)
	}
	defer f16.Release()

	const generated = 64
	greedy32, err := f32.Generate(prompt, generated, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	greedy16, err := f16.Generate(prompt, generated, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	greedy16Again, err := f16.Generate(prompt, generated, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(greedy16, greedy16Again) {
		t.Fatal("f16-KV real-model greedy output is not deterministic")
	}
	if !slices.Equal(greedy16, greedy32) {
		for i := range min(len(greedy16), len(greedy32)) {
			if greedy16[i] != greedy32[i] {
				t.Fatalf("f16 KV changed representative-prompt greedy token %d: f16=%d f32=%d", i, greedy16[i], greedy32[i])
			}
		}
		t.Fatalf("f16/f32 greedy lengths differ: %d/%d", len(greedy16), len(greedy32))
	}

	_, err = f32.StepNLast(prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = f16.StepNLast(prompt, 0)
	if err != nil {
		t.Fatal(err)
	}
	decodeToken := greedy32[len(prompt)]
	logits32, err := f32.Step(decodeToken, len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	logits16, err := f16.Step(decodeToken, len(prompt))
	if err != nil {
		t.Fatal(err)
	}
	var diff2, ref2, maxAbs float64
	for i := range logits16 {
		got, want := float64(logits16[i]), float64(logits32[i])
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Fatalf("f16-KV logit[%d] is non-finite: %g", i, got)
		}
		d := got - want
		diff2 += d * d
		ref2 += want * want
		maxAbs = max(maxAbs, math.Abs(d))
	}
	nrmse := math.Sqrt(diff2 / math.Max(ref2, 1e-30))
	t.Logf("trained-model quality: %d/%d greedy tokens unchanged; logit NRMSE %.6g maxAbs %.6g", len(greedy16), len(greedy32), nrmse, maxAbs)
	if nrmse > 2e-3 {
		t.Fatalf("trained-model normalized logit RMSE %.6g exceeds 2e-3", nrmse)
	}

	profileAt := func(name string, d *llamagpu.Decoder, context int) {
		p := make([]int, context)
		for i := range p {
			p[i] = 1 + (i*131)%min(model.Config.Vocab-1, 30000)
		}
		if _, e := d.StepNLast(p, 0); e != nil {
			t.Fatal(e)
		}
		_, profile, e := d.ProfileMetalStep(7, context, 1024)
		if e != nil {
			t.Fatal(e)
		}
		byLabel := map[string]time.Duration{}
		for _, event := range profile.Events {
			byLabel[event.Label] += event.Duration
		}
		t.Logf("profile %s ctx=%d command=%s span=%s omittedMPS=%d labels=%v", name, context, profile.CommandDuration, profile.EventSpan, profile.OmittedMPS, byLabel)
	}
	profileAt("f32", f32, 512)
	profileAt("f16kv", f16, 512)

	const (
		steps  = 32
		rounds = 3
	)
	contexts := []int{8, 512, 1024, 1536}
	if model.Config.Ctx < contexts[len(contexts)-1]+steps+4 {
		t.Skipf("model context %d too short for the long-context gate", model.Config.Ctx)
	}
	median := func(v []float64) float64 {
		sort.Float64s(v)
		return v[len(v)/2]
	}
	measure := func(d *llamagpu.Decoder, context int) float64 {
		p := make([]int, context)
		for i := range p {
			p[i] = 1 + (i*131)%min(model.Config.Vocab-1, 30000)
		}
		if _, e := d.StepNLast(p, 0); e != nil {
			t.Fatal(e)
		}
		for i := range 3 {
			if _, e := d.Step(1+(i*17)%1000, context+i); e != nil {
				t.Fatal(e)
			}
		}
		start := time.Now()
		for i := range steps {
			if _, e := d.Step(1+(i*37)%1000, context+3+i); e != nil {
				t.Fatal(e)
			}
		}
		return float64(steps) / time.Since(start).Seconds()
	}

	for _, context := range contexts {
		f32Samples, f16Samples := make([]float64, 0, rounds*2), make([]float64, 0, rounds)
		controlSpread := 1.0
		for range rounds {
			before := measure(f32, context)
			got16 := measure(f16, context)
			after := measure(f32, context)
			f32Samples = append(f32Samples, before, after)
			f16Samples = append(f16Samples, got16)
			spread := max(before, after) / min(before, after)
			controlSpread = max(controlSpread, spread)
		}
		base, got := median(f32Samples), median(f16Samples)
		ratio := got / base
		t.Logf("ctx=%d f32=%.2f tok/s f16kv=%.2f tok/s speedup=%.4fx unchanged-f32 max spread=%.3fx f32=%v f16=%v",
			context, base, got, ratio, controlSpread, f32Samples, f16Samples)
		if context == 8 && ratio < 0.99 {
			t.Errorf("short-context f16-KV ratio %.4fx is below 0.99x", ratio)
		}
		if context >= 512 && ratio < 1.03 {
			t.Errorf("ctx=%d full-token f16-KV ratio %.4fx is below 1.03x", context, ratio)
		}
		if controlSpread > 1.15 {
			t.Errorf("ctx=%d unchanged-f32 control spread %.3fx makes the campaign invalid", context, controlSpread)
		}
		fmt.Printf("M2F16KV ctx=%4d f32=%7.2f f16kv=%7.2f tok/s %.4fx controlSpread %.3fx\n", context, base, got, ratio, controlSpread)
	}
}
