//go:build darwin && cgo

package llamagpu_test

import (
	"encoding/binary"
	"slices"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
)

func syntheticIQ4NLWeight(out, in, seed int) []byte {
	raw := make([]byte, out*(in/32)*18)
	for block := range len(raw) / 18 {
		base := block * 18
		binary.LittleEndian.PutUint16(raw[base:], 0x2800) // f16 d=0.03125
		for i := 2; i < 18; i++ {
			raw[base+i] = byte((block*17 + i*29 + seed*11) & 0xff)
		}
	}
	return raw
}

func llamaIQ4NL(m *nlp.Llama) (*nlp.QuantLlama, error) {
	q, err := nlp.QuantizeLlama(m, gguf.Q4_0)
	if err != nil {
		return nil, err
	}
	seed := 1
	replace := func(l *nn.QuantLinear) {
		l.Weight = syntheticIQ4NLWeight(l.Out, l.In, seed)
		l.QT = gguf.IQ4_NL
		seed++
	}
	for _, block := range q.Blocks {
		for _, l := range []*nn.QuantLinear{block.Wq, block.Wk, block.Wv, block.Wo, block.FFN.Gate, block.FFN.Up, block.FFN.Down} {
			replace(l)
		}
	}
	replace(q.Out)
	return q, nil
}

// TestMetalIQ4NLCooperativeEndToEnd proves that the new type-20 kernel is reachable from a
// complete device-resident decoder and measures its whole-token leverage, not only its leaf gain.
func TestMetalIQ4NLCooperativeEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("TinyLlama-shaped model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal unavailable")
	}
	cfg := nlp.LlamaConfig{
		Vocab: 32000, Ctx: 1024, Dim: 2048, Heads: 16, KVHeads: 4, Layers: 6,
		Hidden: 5632, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	qm, err := llamaIQ4NL(m)
	if err != nil {
		t.Fatal(err)
	}
	defer qm.Close()
	dec, err := llamagpu.NewQuant(qm)
	if err != nil {
		t.Fatal(err)
	}
	defer dec.Release()
	prompt := make([]int, 16)
	for i := range prompt {
		prompt[i] = (i*131 + 5) % cfg.Vocab
	}
	const genN = 32
	type result struct {
		tokens []int
		rate   float64
	}
	sample := func(on bool) result {
		previous := metal.SetIQ4NLCooperative(on)
		defer metal.SetIQ4NLCooperative(previous)
		if _, err := dec.Generate(prompt, genN, nlp.Greedy()); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		tokens, err := dec.Generate(prompt, genN, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		return result{tokens: tokens, rate: float64(genN) / time.Since(start).Seconds()}
	}
	var off, on []float64
	for range 3 {
		control, candidate := sample(false), sample(true)
		if !slices.Equal(control.tokens, candidate.tokens) {
			t.Fatal("cooperative IQ4_NL changed generated tokens")
		}
		off = append(off, control.rate)
		on = append(on, candidate.rate)
	}
	medOff, medOn := coopMedianMetal(off), coopMedianMetal(on)
	ratio := medOn / medOff
	t.Logf("IQ4_NL end-to-end: scalar %.2f tok/s -> cooperative %.2f tok/s = %.3fx", medOff, medOn, ratio)
	if ratio < 1.02 {
		t.Fatalf("IQ4_NL %.3fx: cooperative kernel appears unreachable (off %v on %v)", ratio, off, on)
	}
}
