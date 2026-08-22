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

type llamaTQFormat struct {
	name       string
	qt         gguf.QuantType
	blockBytes int
	toggle     func(bool) bool
}

func syntheticTQWeight(format llamaTQFormat, out, in, seed int) []byte {
	raw := make([]byte, out*(in/256)*format.blockBytes)
	for block := range len(raw) / format.blockBytes {
		base := block * format.blockBytes
		for i := range format.blockBytes - 2 {
			raw[base+i] = byte((block*17 + i*29 + seed*11) & 0xff)
		}
		//perfscan:ignore PS4001 strided trailing f16 field in heterogeneous TQ blocks
		binary.LittleEndian.PutUint16(raw[base+format.blockBytes-2:], 0x0800)
	}
	return raw
}

func llamaTQ(m *nlp.Llama, format llamaTQFormat) (*nlp.QuantLlama, error) {
	q, err := nlp.QuantizeLlama(m, gguf.Q4_0)
	if err != nil {
		return nil, err
	}
	seed := 1
	replace := func(linear *nn.QuantLinear) {
		linear.Weight = syntheticTQWeight(format, linear.Out, linear.In, seed)
		linear.QT = format.qt
		seed++
	}
	for _, block := range q.Blocks {
		for _, linear := range []*nn.QuantLinear{block.Wq, block.Wk, block.Wv, block.Wo, block.FFN.Gate, block.FFN.Up, block.FFN.Down} {
			replace(linear)
		}
	}
	replace(q.Out)
	return q, nil
}

// TestMetalTQCooperativeEndToEnd proves both TQ selectors are reachable from a
// complete resident decoder and measures whole-token leverage with identical output tokens.
func TestMetalTQCooperativeEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("TinyLlama-shaped model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal unavailable")
	}
	formats := []llamaTQFormat{
		{"TQ1_0", gguf.TQ1_0, 54, metal.SetTQ1Cooperative},
		{"TQ2_0", gguf.TQ2_0, 66, metal.SetTQ2Cooperative},
	}
	for _, format := range formats {
		t.Run(format.name, func(t *testing.T) {
			cfg := nlp.LlamaConfig{
				Vocab: 32000, Ctx: 1024, Dim: 2048, Heads: 16, KVHeads: 4, Layers: 6,
				Hidden: 5632, Eps: 1e-5, RopeBase: 10000,
			}
			m, err := nlp.NewLlama(cfg, 7)
			if err != nil {
				t.Fatal(err)
			}
			qm, err := llamaTQ(m, format)
			if err != nil {
				t.Fatal(err)
			}
			defer qm.Close()
			decoder, err := llamagpu.NewQuant(qm)
			if err != nil {
				t.Fatal(err)
			}
			defer decoder.Release()
			prompt := make([]int, 16)
			for i := range prompt {
				prompt[i] = (i*131 + 5) % cfg.Vocab
			}
			const generated = 32
			type result struct {
				tokens []int
				rate   float64
			}
			sample := func(cooperative bool) result {
				previous := format.toggle(cooperative)
				defer format.toggle(previous)
				if _, err := decoder.Generate(prompt, generated, nlp.Greedy()); err != nil {
					t.Fatal(err)
				}
				start := time.Now()
				tokens, err := decoder.Generate(prompt, generated, nlp.Greedy())
				if err != nil {
					t.Fatal(err)
				}
				return result{tokens: tokens, rate: float64(generated) / time.Since(start).Seconds()}
			}
			var scalar, cooperative []float64
			for range 3 {
				control, candidate := sample(false), sample(true)
				if !slices.Equal(control.tokens, candidate.tokens) {
					t.Fatal("cooperative TQ changed generated tokens")
				}
				scalar = append(scalar, control.rate)
				cooperative = append(cooperative, candidate.rate)
			}
			medianScalar, medianCooperative := coopMedianMetal(scalar), coopMedianMetal(cooperative)
			ratio := medianCooperative / medianScalar
			t.Logf("%s end-to-end: scalar %.2f tok/s -> cooperative %.2f tok/s = %.3fx", format.name, medianScalar, medianCooperative, ratio)
			if ratio < 1.02 {
				t.Fatalf("%s %.3fx: cooperative kernel appears unreachable (scalar %v cooperative %v)", format.name, ratio, scalar, cooperative)
			}
		})
	}
}
