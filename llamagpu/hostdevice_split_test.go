//go:build darwin && cgo

package llamagpu_test

import (
	"fmt"
	"sort"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestHostDeviceSplit measures the per-token HOST (CPU sampling) vs DEVICE (GPU
// forward) time split in the metal decode loop, to inform ADR-0021/§T630 (whether
// a CPU-SIMD ∥ GPU overlap pipeline is worth building). Not an assertion test —
// run with -run TestHostDeviceSplit -v to print the numbers.
//
//	CGO_ENABLED=1 go test ./llamagpu/ -run TestHostDeviceSplit -count=1 -v
func TestHostDeviceSplit(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal not available on this machine")
	}
	// A small-but-representative Llama: D=512, GQA 8:2, 6 layers, 32k vocab, ctx 512 —
	// the same shape the package doc benchmarks (24× on metal).
	cfg := nlp.LlamaConfig{
		Vocab: 32000, Ctx: 512, Dim: 512, Heads: 8, KVHeads: 2, Layers: 6,
		Hidden: 1376, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 42)
	if err != nil {
		t.Fatalf("NewLlama: %v", err)
	}
	dec, err := llamagpu.New(m)
	if err != nil {
		t.Fatalf("llamagpu.New: %v", err)
	}
	defer dec.Release()

	// Realistic sampler: temp 0.8 + top-k 40 + top-p 0.9 + repeat penalty 1.1.
	s := nlp.NewSampler(7,
		nlp.WithTemperature(0.8),
		nlp.WithTopK(40),
		nlp.WithTopP(0.9),
		nlp.WithRepeatPenalty(1.1),
	)

	const warm = 8
	const N = 200
	if warm+N >= cfg.Ctx {
		t.Fatalf("warm+N (%d) must be < ctx (%d)", warm+N, cfg.Ctx)
	}

	// history grows as we decode — the repeat penalty scans it every token.
	history := make([]int, 0, warm+N+8)
	buf := make([]float64, cfg.Vocab)
	tok, pos := 3, 0

	toF64 := func(l []float32) {
		for i, x := range l {
			buf[i] = float64(x)
		}
	}

	// Warm-up: build up some history and let the GPU pipeline settle.
	for i := 0; i < warm; i++ {
		l, err := dec.Step(tok, pos)
		if err != nil {
			t.Fatalf("warmup Step: %v", err)
		}
		toF64(l)
		tok = s.SampleWithHistory(buf, history)
		history = append(history, tok)
		pos++
	}

	devNs := make([]float64, 0, N)
	hostNs := make([]float64, 0, N)

	for i := 0; i < N; i++ {
		t0 := time.Now()
		l, err := dec.Step(tok, pos) // DEVICE: GPU forward, blocks until logits ready
		t1 := time.Now()
		if err != nil {
			t.Fatalf("Step: %v", err)
		}
		toF64(l) // conversion excluded from tHost (measured separately below as negligible)
		t2 := time.Now()
		tok = s.SampleWithHistory(buf, history) // HOST: CPU sampling
		t3 := time.Now()

		history = append(history, tok)
		pos++

		devNs = append(devNs, float64(t1.Sub(t0).Nanoseconds()))
		hostNs = append(hostNs, float64(t3.Sub(t2).Nanoseconds()))
	}

	meanDev := mean(devNs)
	meanHost := mean(hostNs)
	frac := 100 * meanHost / (meanHost + meanDev)
	tokPerSec := 1e9 / (meanHost + meanDev)

	t.Logf("model: D=%d heads=%d kvHeads=%d layers=%d vocab=%d ctx=%d hidden=%d  backend=metal",
		cfg.Dim, cfg.Heads, cfg.KVHeads, cfg.Layers, cfg.Vocab, cfg.Ctx, cfg.Hidden)
	t.Logf("N=%d tokens (after %d warm-up)", N, warm)
	t.Logf("tDevice (GPU Step):        mean %s   p50 %s   p95 %s",
		dur(meanDev), dur(median(devNs)), dur(pct(devNs, 95)))
	t.Logf("tHost   (SampleWithHist):  mean %s   p50 %s   p95 %s",
		dur(meanHost), dur(median(hostNs)), dur(pct(hostNs, 95)))
	t.Logf("host fraction tHost/(tHost+tDevice) = %.1f%%", frac)
	t.Logf("serial throughput ≈ %.0f tok/s", tokPerSec)
	t.Logf("if tHost fully hidden behind next Step: ≈ %.0f tok/s (%.1f%% faster)",
		1e9/meanDev, 100*(meanHost/meanDev))
}

func mean(x []float64) float64 {
	var s float64
	for _, v := range x {
		s += v
	}
	return s / float64(len(x))
}

func median(x []float64) float64 { return pct(x, 50) }

func pct(x []float64, p int) float64 {
	c := append([]float64(nil), x...)
	sort.Float64s(c)
	idx := p * (len(c) - 1) / 100
	return c[idx]
}

func dur(ns float64) string { return fmt.Sprintf("%.3fms", ns/1e6) }
