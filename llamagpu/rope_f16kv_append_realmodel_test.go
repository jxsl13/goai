//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"os"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

func TestMetalRoPEF16KVAppendRealModelGate(t *testing.T) {
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
	dec, err := llamagpu.NewQuantF16KV(model)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	previous := metal.SetRoPEF16KVAppend(false)
	defer metal.SetRoPEF16KVAppend(previous)

	prompt := make([]int, 64)
	for i := range prompt {
		prompt[i] = 1 + (i*131)%min(model.Config.Vocab-1, 30000)
	}
	runLogits := func(fused bool) []float32 {
		metal.SetRoPEF16KVAppend(fused)
		if _, err := dec.StepNLast(prompt, 0); err != nil {
			t.Fatal(err)
		}
		logits, err := dec.Step(7, len(prompt))
		if err != nil {
			t.Fatal(err)
		}
		return logits
	}
	controlLogits := runLogits(false)
	candidateLogits := runLogits(true)
	if len(controlLogits) != len(candidateLogits) {
		t.Fatalf("logit lengths differ: control=%d candidate=%d", len(controlLogits), len(candidateLogits))
	}
	for i := range controlLogits {
		if math.Float32bits(controlLogits[i]) != math.Float32bits(candidateLogits[i]) {
			t.Fatalf("trained-model logit %d differs bitwise: control=%g candidate=%g", i, controlLogits[i], candidateLogits[i])
		}
	}
	argmax := func(x []float32) int {
		best := 0
		for i := 1; i < len(x); i++ {
			if x[i] > x[best] {
				best = i
			}
		}
		return best
	}
	if got, want := argmax(candidateLogits), argmax(controlLogits); got != want {
		t.Fatalf("trained-model greedy token differs: candidate=%d control=%d", got, want)
	}

	profileLabels := func(fused bool) []string {
		metal.SetRoPEF16KVAppend(fused)
		if _, err := dec.StepNLast(prompt, 0); err != nil {
			t.Fatal(err)
		}
		_, profile, err := dec.ProfileMetalStep(7, len(prompt), 512)
		if err != nil {
			t.Fatal(err)
		}
		labels := make([]string, len(profile.Events))
		for i := range profile.Events {
			labels[i] = profile.Events[i].Label
		}
		return labels
	}
	controlLabels, candidateLabels := profileLabels(false), profileLabels(true)
	count := func(labels []string, label string) int {
		n := 0
		for _, got := range labels {
			if got == label {
				n++
			}
		}
		return n
	}
	// This mixed Q4_K_M file groups 12 homogeneous QKV layers and leaves 10 mixed layers on the
	// separate projection route. The new boundary applies only to those 10 eligible layers.
	if got := count(controlLabels, "rope"); got != 20 {
		t.Fatalf("control separate-QKV RoPE event count=%d want 20", got)
	}
	if got := count(controlLabels, "kv.f32_to_f16_pair"); got != 22 {
		t.Fatalf("control paired-copy event count=%d want 22", got)
	}
	if got := count(controlLabels, "rope_pair"); got != 12 {
		t.Fatalf("control grouped-QKV RoPE event count=%d want 12", got)
	}
	if got := count(controlLabels, "rope.f16kv.append"); got != 0 {
		t.Fatalf("control fused event count=%d want 0", got)
	}
	if got := count(candidateLabels, "rope.f16kv.append"); got != 10 {
		t.Fatalf("candidate fused event count=%d want 10", got)
	}
	if got := count(candidateLabels, "rope"); got != 0 {
		t.Fatalf("candidate separate-QKV RoPE event count=%d want 0", got)
	}
	if got := count(candidateLabels, "kv.f32_to_f16_pair"); got != 12 {
		t.Fatalf("candidate grouped-layer paired-copy event count=%d want 12", got)
	}
	if got := count(candidateLabels, "rope_pair"); got != 12 {
		t.Fatalf("candidate grouped-QKV RoPE event count=%d want 12", got)
	}
	if got := len(controlLabels) - len(candidateLabels); got != 20 {
		t.Fatalf("profile event reduction=%d want exactly 20", got)
	}
	if slices.Equal(controlLabels, candidateLabels) || !strings.Contains(strings.Join(candidateLabels, ","), "rope.f16kv.append") {
		t.Fatal("profile did not prove distinct candidate routing")
	}

	const (
		context = 512
		steps   = 24
	)
	contextPrompt := make([]int, context)
	for i := range contextPrompt {
		contextPrompt[i] = 1 + (i*131)%min(model.Config.Vocab-1, 30000)
	}
	measure := func(fused bool) float64 {
		metal.SetRoPEF16KVAppend(fused)
		if _, err := dec.StepNLast(contextPrompt, 0); err != nil {
			t.Fatal(err)
		}
		for i := range 3 {
			if _, err := dec.Step(1+(i*17)%1000, context+i); err != nil {
				t.Fatal(err)
			}
		}
		start := time.Now()
		for i := range steps {
			if _, err := dec.Step(1+(i*37)%1000, context+3+i); err != nil {
				t.Fatal(err)
			}
		}
		return float64(steps) / time.Since(start).Seconds()
	}
	for range 2 {
		measure(false)
		measure(true)
	}
	median := func(v []float64) float64 {
		sort.Float64s(v)
		return v[len(v)/2]
	}
	for campaign := range 3 {
		control, candidate := make([]float64, 0, 7), make([]float64, 0, 7)
		for sample := range 7 {
			if (campaign+sample)&1 == 0 {
				control = append(control, measure(false))
				candidate = append(candidate, measure(true))
			} else {
				candidate = append(candidate, measure(true))
				control = append(control, measure(false))
			}
		}
		base, got := median(control), median(candidate)
		ratio := got / base
		t.Logf("campaign=%d control=%.3f tok/s candidate=%.3f tok/s speedup=%.4fx control_samples=%v candidate_samples=%v", campaign+1, base, got, ratio, control, candidate)
		if ratio < 1.01 {
			t.Errorf("campaign %d trained-model decode speedup %.4fx is below 1.01x", campaign+1, ratio)
		}
	}
}
