//go:build darwin && cgo

package llamagpu_test

import (
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

// TestMetalSplitKFusedRealModelGate is one independent end-to-end promotion campaign. Run the
// compiled test binary three times: each process performs AB/BA pairing for both production KV
// dtypes and rejects a campaign whose unchanged control spreads by more than 15 percent.
func TestMetalSplitKFusedRealModelGate(t *testing.T) {
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

	defer metal.SetSplitKFused(true)

	const (
		steps  = 32
		rounds = 3
	)
	contexts := []int{512, 1536}
	if model.Config.Ctx < contexts[len(contexts)-1]+steps+4 {
		t.Skipf("model context %d too short for the long-context gate", model.Config.Ctx)
	}
	median := func(values []float64) float64 {
		values = slices.Clone(values)
		sort.Float64s(values)
		return values[len(values)/2]
	}
	for _, dtype := range []struct {
		name string
		new  func() (*llamagpu.Decoder, error)
	}{{"f16kv", func() (*llamagpu.Decoder, error) { return llamagpu.NewQuantF16KV(model) }}} {
		t.Run(dtype.name, func(t *testing.T) {
			controlDecoder, err := dtype.new()
			if err != nil {
				t.Fatal(err)
			}
			defer controlDecoder.Release()
			candidateDecoder, err := dtype.new()
			if err != nil {
				t.Fatal(err)
			}
			defer candidateDecoder.Release()

			measurePair := func(context int, reverse bool) (float64, float64, []float32, []float32) {
				prompt := make([]int, context)
				for i := range prompt {
					prompt[i] = 1 + (i*131)%min(model.Config.Vocab-1, 30000)
				}
				metal.SetSplitKFused(false)
				if _, e := controlDecoder.StepNLast(prompt, 0); e != nil {
					t.Fatal(e)
				}
				metal.SetSplitKFused(true)
				if _, e := candidateDecoder.StepNLast(prompt, 0); e != nil {
					t.Fatal(e)
				}
				for i := range 3 {
					for _, fused := range [...]bool{false, true} {
						decoder := controlDecoder
						if fused {
							decoder = candidateDecoder
						}
						metal.SetSplitKFused(fused)
						if _, e := decoder.Step(1+(i*17)%1000, context+i); e != nil {
							t.Fatal(e)
						}
					}
				}
				var controlWall, candidateWall time.Duration
				var controlLogits, candidateLogits []float32
				for i := range steps {
					firstFused := (i&1 == 1) != reverse
					for _, fused := range [...]bool{firstFused, !firstFused} {
						decoder := controlDecoder
						if fused {
							decoder = candidateDecoder
						}
						metal.SetSplitKFused(fused)
						start := time.Now()
						logits, e := decoder.Step(1+(i*37)%1000, context+3+i)
						elapsed := time.Since(start)
						if e != nil {
							t.Fatal(e)
						}
						if fused {
							candidateWall += elapsed
							candidateLogits = logits
						} else {
							controlWall += elapsed
							controlLogits = logits
						}
					}
				}
				return float64(steps) / controlWall.Seconds(), float64(steps) / candidateWall.Seconds(), controlLogits, candidateLogits
			}
			profileAt := func(decoder *llamagpu.Decoder, context int, fused bool) metal.RecorderProfile {
				prompt := make([]int, context)
				for i := range prompt {
					prompt[i] = 1 + (i*131)%min(model.Config.Vocab-1, 30000)
				}
				metal.SetSplitKFused(fused)
				if _, e := decoder.StepNLast(prompt, 0); e != nil {
					t.Fatal(e)
				}
				_, profile, e := decoder.ProfileMetalStep(7, context, 1024)
				if e != nil {
					t.Fatal(e)
				}
				return profile
			}

			for _, context := range contexts {
				controlProfile := profileAt(controlDecoder, context, false)
				candidateProfile := profileAt(candidateDecoder, context, true)
				controlLabels := map[string]time.Duration{}
				candidateLabels := map[string]time.Duration{}
				for _, event := range controlProfile.Events {
					controlLabels[event.Label] += event.Duration
				}
				for _, event := range candidateProfile.Events {
					candidateLabels[event.Label] += event.Duration
				}
				t.Logf("%s ctx=%d profile controlCommand=%s candidateCommand=%s controlLabels=%v candidateLabels=%v",
					dtype.name, context, controlProfile.CommandDuration, candidateProfile.CommandDuration, controlLabels, candidateLabels)
				control := make([]float64, 0, rounds)
				candidate := make([]float64, 0, rounds)
				var controlLogits, candidateLogits []float32
				for round := range rounds {
					baseTPS, candidateTPS, baseLogits, fusedLogits := measurePair(context, round&1 == 1)
					control = append(control, baseTPS)
					candidate = append(candidate, candidateTPS)
					if controlLogits == nil {
						controlLogits = slices.Clone(baseLogits)
						candidateLogits = slices.Clone(fusedLogits)
					}
				}
				minControl, maxControl := control[0], control[0]
				pairedRatios := make([]float64, len(control))
				for _, sample := range control[1:] {
					minControl = min(minControl, sample)
					maxControl = max(maxControl, sample)
				}
				for i := range control {
					pairedRatios[i] = candidate[i] / control[i]
				}
				minRatio, maxRatio := pairedRatios[0], pairedRatios[0]
				for _, ratio := range pairedRatios[1:] {
					minRatio = min(minRatio, ratio)
					maxRatio = max(maxRatio, ratio)
				}
				controlSpread := maxControl / minControl
				ratioSpread := maxRatio / minRatio
				base, got := median(control), median(candidate)
				speedup := median(pairedRatios)
				t.Logf("%s ctx=%d control=%.2f candidate=%.2f tok/s pairedSpeedup=%.4fx ratioSpread=%.3fx controlSpread=%.3fx control=%v candidate=%v ratios=%v",
					dtype.name, context, base, got, speedup, ratioSpread, controlSpread, control, candidate, pairedRatios)
				if ratioSpread > 1.15 {
					t.Errorf("%s ctx=%d paired-ratio spread %.3fx makes the campaign invalid", dtype.name, context, ratioSpread)
				}
				if speedup < 1.01 {
					t.Errorf("%s ctx=%d fused speedup %.4fx is below 1.01x", dtype.name, context, speedup)
				}

				var diff2, ref2 float64
				controlArgmax, candidateArgmax := 0, 0
				for i := range controlLogits {
					want, got := float64(controlLogits[i]), float64(candidateLogits[i])
					if math.IsNaN(got) || math.IsInf(got, 0) {
						t.Fatalf("%s ctx=%d candidate logit[%d] is non-finite: %g", dtype.name, context, i, got)
					}
					delta := got - want
					diff2 += delta * delta
					ref2 += want * want
					if controlLogits[i] > controlLogits[controlArgmax] {
						controlArgmax = i
					}
					if candidateLogits[i] > candidateLogits[candidateArgmax] {
						candidateArgmax = i
					}
				}
				nrmse := math.Sqrt(diff2 / math.Max(ref2, 1e-30))
				if nrmse > 2e-5 || candidateArgmax != controlArgmax {
					t.Errorf("%s ctx=%d fused quality NRMSE=%.3e argmax=%d/%d", dtype.name, context, nrmse, controlArgmax, candidateArgmax)
				}
			}
		})
	}
}
