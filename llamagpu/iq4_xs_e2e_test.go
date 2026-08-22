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

func syntheticIQ4XSWeight(out, in, seed int) []byte {
	raw := make([]byte, out*(in/256)*136)
	for block := range len(raw) / 136 {
		base := block * 136
		//perfscan:ignore PS4001 strided f16 fields in heterogeneous IQ4_XS blocks cannot use a same-layout bulk copy
		binary.LittleEndian.PutUint16(raw[base:], 0x0800) // small f16 d
		binary.LittleEndian.PutUint16(raw[base+2:], 0xaaaa)
		for i := 4; i < 8; i++ {
			raw[base+i] = 0x11 // every signed 6-bit sub-scale is one
		}
		for i := 8; i < 136; i++ {
			raw[base+i] = byte((block*17 + i*29 + seed*11) & 0xff)
		}
	}
	return raw
}

func llamaIQ4XS(m *nlp.Llama) (*nlp.QuantLlama, error) {
	q, err := nlp.QuantizeLlama(m, gguf.Q4_0)
	if err != nil {
		return nil, err
	}
	seed := 1
	replace := func(l *nn.QuantLinear) {
		l.Weight = syntheticIQ4XSWeight(l.Out, l.In, seed)
		l.QT = gguf.IQ4_XS
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

// TestMetalIQ4XSCooperativeEndToEnd proves the type-23 kernel is reachable from a complete
// device-resident decoder and measures whole-token leverage rather than only a leaf kernel.
func TestMetalIQ4XSCooperativeEndToEnd(t *testing.T) {
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
	qm, err := llamaIQ4XS(m)
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
		previous := metal.SetIQ4XSCooperative(on)
		defer metal.SetIQ4XSCooperative(previous)
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
			t.Fatal("cooperative IQ4_XS changed generated tokens")
		}
		off = append(off, control.rate)
		on = append(on, candidate.rate)
	}
	medOff, medOn := coopMedianMetal(off), coopMedianMetal(on)
	ratio := medOn / medOff
	t.Logf("IQ4_XS end-to-end: scalar %.2f tok/s -> cooperative %.2f tok/s = %.3fx", medOff, medOn, ratio)
	if ratio < 1.02 {
		t.Fatalf("IQ4_XS %.3fx: cooperative kernel appears unreachable (off %v on %v)", ratio, off, on)
	}
}
