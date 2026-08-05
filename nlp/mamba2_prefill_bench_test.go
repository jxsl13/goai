package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// benchMamba2Model builds an unquantized Mamba2 at a size where the causal conv in
// mixer2Prefill is a meaningful share of the prefill. Mirrors the fixture shape used by
// the GGUF tests, scaled up.
func benchMamba2Model(dModel, heads, headDim, groups, n, dConv, layers, vocab int) *nlp.Mamba2 {
	cfg := nlp.Mamba2Config{
		DModel: dModel, NumHeads: heads, HeadDim: headDim, NGroups: groups, N: n,
		DConv: dConv, Intermediate: heads * headDim, Layers: layers, Vocab: vocab,
		Eps: 0x1p-17,
	}
	convDim := cfg.Intermediate + 2*cfg.NGroups*cfg.N
	projSize := cfg.Intermediate + convDim + cfg.NumHeads
	fill := func(shape tensor.Shape, seed, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, shape)
		s := t.Storage().F64()
		for i := range s {
			s[i] = scale * math.Sin(seed+1.9*float64(i))
		}
		return t
	}
	vec := func(width int, seed, scale float64) *tensor.Tensor {
		v := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			v.SetF64(scale*math.Sin(seed+2.3*float64(i))+0.01, i)
		}
		return v
	}
	gain := func(width int, seed float64) *tensor.Tensor {
		g := tensor.New(tensor.F64, tensor.Shape{width})
		for i := range width {
			g.SetF64(1+0.25*math.Sin(seed+2.3*float64(i)), i)
		}
		return g
	}
	m := &nlp.Mamba2{
		Config: cfg,
		Embed:  fill(tensor.Shape{cfg.Vocab, cfg.DModel}, 0.3, 0.5),
		Norm:   &nn.RMSNorm{Gamma: gain(cfg.DModel, 9.9), Eps: cfg.Eps},
	}
	// Tied head: Embed transposed, matching the checkpoint default. Prefill ends in a
	// matmul against it, so a nil Head panics rather than erroring.
	ht := tensor.New(tensor.F64, tensor.Shape{cfg.DModel, cfg.Vocab})
	for i := range cfg.Vocab {
		for j := range cfg.DModel {
			ht.SetF64(m.Embed.AtF64(i, j), j, i)
		}
	}
	m.Head = ht
	for l := range cfg.Layers {
		fl := float64(l)
		m.Layers = append(m.Layers, nlp.Mamba2Layer{
			Norm: &nn.RMSNorm{Gamma: gain(cfg.DModel, fl+0.6), Eps: cfg.Eps},
			Mixer: &nlp.Mamba2Mixer{
				InProj:  fill(tensor.Shape{projSize, cfg.DModel}, fl+1.1, 0.12),
				ConvW:   fill(tensor.Shape{convDim, cfg.DConv}, fl+3.3, 0.2),
				ConvB:   vec(convDim, fl+3.7, 0.05),
				ALog:    fill(tensor.Shape{cfg.NumHeads}, fl+8.8, 1.2),
				D:       vec(cfg.NumHeads, fl+9.1, 0.1),
				DtBias:  vec(cfg.NumHeads, fl+7.9, 0.05),
				NormW:   gain(cfg.Intermediate, fl+6.2),
				OutProj: fill(tensor.Shape{cfg.DModel, cfg.Intermediate}, fl+9.5, 0.12),
				DModel:  cfg.DModel, NumHeads: cfg.NumHeads, HeadDim: cfg.HeadDim,
				NGroups: cfg.NGroups, N: cfg.N, DConv: cfg.DConv,
				Intermediate: cfg.Intermediate, ConvDim: convDim, Eps: cfg.Eps,
			},
		})
	}
	return m
}

func benchMamba2Prefill(b *testing.B, seq int) {
	m := benchMamba2Model(256, 8, 32, 2, 16, 4, 2, 64)
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = (i * 7) % 64
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx := backend.NewContext()
		st := m.NewDecodeState()
		if _, err := m.Prefill(ctx, st, tokens); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMamba2Prefill_256(b *testing.B) { benchMamba2Prefill(b, 256) }
func BenchmarkMamba2Prefill_512(b *testing.B) { benchMamba2Prefill(b, 512) }

// benchMamba2PrefillReal uses realistic Mamba-2 dims (heads=48, headDim=64, N=128, groups=8) where the
// per-head SSD scan is a dominant fraction of prefill — the regime the small default bench understates.
func benchMamba2PrefillReal(b *testing.B, seq int) {
	m := benchMamba2Model(3072, 48, 64, 8, 128, 4, 2, 256)
	tokens := make([]int, seq)
	for i := range tokens {
		tokens[i] = (i * 7) % 256
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx := backend.NewContext()
		st := m.NewDecodeState()
		if _, err := m.Prefill(ctx, st, tokens); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMamba2PrefillReal_512(b *testing.B) { benchMamba2PrefillReal(b, 512) }
