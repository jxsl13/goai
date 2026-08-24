//go:build darwin && cgo

package llamagpu

import (
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
)

var gptBoundaryLogitsSink []float32
var gptBoundaryTokensSink []int

func gptBoundaryPair(b *testing.B) (candidate, control *GPTDecoder, cfg nlp.GPTConfig, prompt []int) {
	b.Helper()
	if !metal.Available() {
		b.Skip("metal: no gpu")
	}
	cfg = nlp.GPTConfig{Vocab: 50257, Ctx: 1024, Dim: 768, Heads: 12, Layers: 12, Eps: 1e-5}
	m := gptStorageModel(b, cfg)
	var err error
	candidate, err = newGPTDecoder(m, metalGPTOps())
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(candidate.Release)
	controlOps := metalGPTOps()
	controlOps.newRecorder = func() (recorder, error) {
		r, err := metal.NewRecorder()
		if err != nil {
			return nil, err
		}
		return mRec{r: r}, nil
	}
	control, err = newGPTDecoder(m, controlOps)
	if err != nil {
		b.Fatal(err)
	}
	control.eagerBoundaryControl = true
	b.Cleanup(control.Release)
	prompt = make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	return candidate, control, cfg, prompt
}

// BenchmarkGPTDecodeBoundaryPairedMetal compares the complete historical public Step boundary with
// caller-owned staging and pooled recorder wrappers inside each order-reversed iteration.
func BenchmarkGPTDecodeBoundaryPairedMetal(b *testing.B) {
	candidate, control, cfg, prompt := gptBoundaryPair(b)
	candidateOut := make([]float32, cfg.Vocab)
	if err := candidate.StepNLastInto(prompt, 0, candidateOut); err != nil {
		b.Fatal(err)
	}
	controlOut, err := control.StepNLast(prompt, 0)
	if err != nil {
		b.Fatal(err)
	}
	gptBoundaryLogitsSink = controlOut
	pos := len(prompt)
	var candidateDuration, controlDuration time.Duration
	runCandidate := func(token, pos int) {
		start := time.Now()
		if err := candidate.StepInto(token, pos, candidateOut); err != nil {
			b.Fatal(err)
		}
		candidateDuration += time.Since(start)
	}
	runControl := func(token, pos int) {
		start := time.Now()
		out, err := control.Step(token, pos)
		if err != nil {
			b.Fatal(err)
		}
		controlDuration += time.Since(start)
		gptBoundaryLogitsSink = out
	}
	b.ResetTimer()
	for i := range b.N {
		if pos >= cfg.Ctx {
			pos = len(prompt) + 1
		}
		token := (pos*97 + 11) % cfg.Vocab
		if i&1 == 0 {
			runControl(token, pos)
			runCandidate(token, pos)
		} else {
			runCandidate(token, pos)
			runControl(token, pos)
		}
		pos++
	}
	b.StopTimer()
	b.ReportMetric(float64(controlDuration)/float64(candidateDuration), "control/caller-buffer")
}

// BenchmarkGPTPrefillBoundaryPairedMetal applies the same paired gate to 16-token StepNLast.
func BenchmarkGPTPrefillBoundaryPairedMetal(b *testing.B) {
	candidate, control, cfg, prompt := gptBoundaryPair(b)
	candidateOut := make([]float32, cfg.Vocab)
	if err := candidate.StepNLastInto(prompt, 0, candidateOut); err != nil {
		b.Fatal(err)
	}
	controlOut, err := control.StepNLast(prompt, 0)
	if err != nil {
		b.Fatal(err)
	}
	gptBoundaryLogitsSink = controlOut
	var candidateDuration, controlDuration time.Duration
	runCandidate := func() {
		start := time.Now()
		if err := candidate.StepNLastInto(prompt, 0, candidateOut); err != nil {
			b.Fatal(err)
		}
		candidateDuration += time.Since(start)
	}
	runControl := func() {
		start := time.Now()
		out, err := control.StepNLast(prompt, 0)
		if err != nil {
			b.Fatal(err)
		}
		controlDuration += time.Since(start)
		gptBoundaryLogitsSink = out
	}
	b.ResetTimer()
	for i := range b.N {
		if i&1 == 0 {
			runControl()
			runCandidate()
		} else {
			runCandidate()
			runControl()
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(controlDuration)/float64(candidateDuration), "control/caller-buffer")
}

func gptGeneratePair(tb testing.TB) (candidate, control *GPTDecoder, prompt []int) {
	tb.Helper()
	if !metal.Available() {
		tb.Skip("metal: no gpu")
	}
	cfg := nlp.GPTConfig{Vocab: 50257, Ctx: 1024, Dim: 768, Heads: 12, Layers: 12, Eps: 1e-5}
	m := gptStorageModel(tb, cfg)
	var err error
	candidate, err = newGPTDecoder(m, metalGPTOps())
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(candidate.Release)
	control, err = newGPTDecoder(m, metalGPTOps())
	if err != nil {
		tb.Fatal(err)
	}
	control.eagerGenerateResultControl = true
	tb.Cleanup(control.Release)
	prompt = make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*97 + 11) % cfg.Vocab
	}
	return candidate, control, prompt
}

// BenchmarkGPTGenerateAllocationsMetal isolates the allocation effect of reusing one Vocab-sized
// logits slice for the prefill and every decode step of an 8-token GPT-2-small generation.
func BenchmarkGPTGenerateAllocationsMetal(b *testing.B) {
	candidate, control, prompt := gptGeneratePair(b)
	const maxNew = 8
	sampler := nlp.Greedy()
	if _, err := candidate.Generate(prompt, maxNew, sampler); err != nil {
		b.Fatal(err)
	}
	if _, err := control.Generate(prompt, maxNew, sampler); err != nil {
		b.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		dec  *GPTDecoder
	}{
		{name: "reuse", dec: candidate},
		{name: "historical", dec: control},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				out, err := tc.dec.Generate(prompt, maxNew, sampler)
				if err != nil {
					b.Fatal(err)
				}
				gptBoundaryTokensSink = out
			}
		})
	}
}

// BenchmarkGPTGeneratePairedMetal measures the complete generation boundary in alternating order.
func BenchmarkGPTGeneratePairedMetal(b *testing.B) {
	candidate, control, prompt := gptGeneratePair(b)
	const maxNew = 8
	sampler := nlp.Greedy()
	if _, err := candidate.Generate(prompt, maxNew, sampler); err != nil {
		b.Fatal(err)
	}
	if _, err := control.Generate(prompt, maxNew, sampler); err != nil {
		b.Fatal(err)
	}
	var candidateDuration, controlDuration time.Duration
	run := func(dec *GPTDecoder, elapsed *time.Duration) {
		start := time.Now()
		out, err := dec.Generate(prompt, maxNew, sampler)
		*elapsed += time.Since(start)
		if err != nil {
			b.Fatal(err)
		}
		gptBoundaryTokensSink = out
	}
	b.ResetTimer()
	for i := range b.N {
		if i&1 == 0 {
			run(control, &controlDuration)
			run(candidate, &candidateDuration)
		} else {
			run(candidate, &candidateDuration)
			run(control, &controlDuration)
		}
	}
	b.StopTimer()
	b.ReportMetric(float64(controlDuration)/float64(candidateDuration), "candidate/control")
}
