package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The Jamba hybrid decode had no benchmark, which is why its per-token allocation behavior went
// unmeasured while the Mamba2 sibling's was tuned twice. jamba_decode.go calls rows2D six times per
// step, each materializing a [1,width] tensor as [][]float64 to keep row 0, and nothing could be
// claimed about that without an instrument (BENCH-PROVE-THE-CODE-RAN-001).
//
// Sized for the decode path rather than for a parity fixture: the golden checkpoint this package
// tests against is Dim=32/dInner=64, small enough that fixed overhead would dominate and the
// per-layer allocation cost would be invisible. These are still modest but representative
// proportions — dInner = 2*Dim, an SSM state of 16, four layers interleaving both mixer families
// and both FFN families, so one benchmark covers every path a step can take.
func benchJambaModel(dim, dInner, nState, dConv, dtRank, ffn, nExp, layers, vocab, heads, kvHeads int) *nlp.Jamba {
	cfg := nlp.JambaConfig{
		Vocab: vocab, Dim: dim, Heads: heads, KVHeads: kvHeads, Layers: layers, TopK: 2,
		Eps: 0x1p-17,
	}
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
		s := v.Storage().F64()
		for i := range width {
			s[i] = scale*math.Sin(seed+2.3*float64(i)) + 0.01
		}
		return v
	}
	rms := func(width int, seed float64) *nn.RMSNorm {
		g := tensor.New(tensor.F64, tensor.Shape{width})
		s := g.Storage().F64()
		for i := range width {
			s[i] = 1 + 0.25*math.Sin(seed+2.3*float64(i))
		}
		return &nn.RMSNorm{Gamma: g, Eps: cfg.Eps}
	}
	attn := func(seed float64) *nlp.JambaAttention {
		hd := cfg.Dim / cfg.Heads
		return &nlp.JambaAttention{
			Wq: fill(tensor.Shape{cfg.Dim, cfg.Heads * hd}, seed+1.1, 0.12),
			Wk: fill(tensor.Shape{cfg.Dim, cfg.KVHeads * hd}, seed+2.2, 0.12),
			Wv: fill(tensor.Shape{cfg.Dim, cfg.KVHeads * hd}, seed+3.3, 0.12),
			Wo: fill(tensor.Shape{cfg.Heads * hd, cfg.Dim}, seed+4.4, 0.12),
		}
	}
	mixer := func(seed float64) *nlp.JambaMixer {
		return &nlp.JambaMixer{
			Block: &nn.MambaBlock{
				InX:     &nn.Linear{W: fill(tensor.Shape{cfg.Dim, dInner}, seed+1.1, 0.12)},
				InZ:     &nn.Linear{W: fill(tensor.Shape{cfg.Dim, dInner}, seed+2.2, 0.12)},
				ConvW:   fill(tensor.Shape{dInner, dConv}, seed+3.3, 0.2),
				ConvB:   vec(dInner, seed+3.7, 0.05),
				DtLow:   &nn.Linear{W: fill(tensor.Shape{dInner, dtRank}, seed+4.4, 0.12)},
				BProj:   &nn.Linear{W: fill(tensor.Shape{dInner, nState}, seed+5.5, 0.12)},
				CProj:   &nn.Linear{W: fill(tensor.Shape{dInner, nState}, seed+6.6, 0.12)},
				DtProj:  &nn.Linear{W: fill(tensor.Shape{dtRank, dInner}, seed+7.7, 0.12), B: vec(dInner, seed+7.9, 0.05)},
				ALog:    fill(tensor.Shape{dInner, nState}, seed+8.8, 1.2),
				Dskip:   vec(dInner, seed+9.1, 0.1),
				OutProj: &nn.Linear{W: fill(tensor.Shape{dInner, cfg.Dim}, seed+9.5, 0.12)},
				DModel:  cfg.Dim, DInner: dInner, DConv: dConv, N: nState, DtRank: dtRank,
			},
			DtNorm: rms(dtRank, seed+10.1),
			BNorm:  rms(nState, seed+10.4),
			CNorm:  rms(nState, seed+10.7),
		}
	}
	dense := func(seed float64) *nn.SwiGLU {
		return &nn.SwiGLU{
			Wgate: fill(tensor.Shape{cfg.Dim, ffn}, seed+1.3, 0.12),
			Wup:   fill(tensor.Shape{cfg.Dim, ffn}, seed+2.6, 0.12),
			Wdown: fill(tensor.Shape{ffn, cfg.Dim}, seed+3.9, 0.12),
		}
	}
	moe := func(seed float64) *nlp.JambaMoE {
		m := &nlp.JambaMoE{
			Router: fill(tensor.Shape{cfg.Dim, nExp}, seed+0.7, 0.9),
			TopK:   cfg.TopK,
		}
		for e := range nExp {
			fe := float64(e)
			m.Experts = append(m.Experts, &nn.SwiGLU{
				Wgate: fill(tensor.Shape{cfg.Dim, ffn}, seed+4.1+fe, 0.12),
				Wup:   fill(tensor.Shape{cfg.Dim, ffn}, seed+5.2+fe, 0.12),
				Wdown: fill(tensor.Shape{ffn, cfg.Dim}, seed+6.3+fe, 0.12),
			})
		}
		return m
	}
	m := &nlp.Jamba{
		Config: cfg,
		TokEmb: fill(tensor.Shape{cfg.Vocab, cfg.Dim}, 0.3, 0.5),
		Norm:   rms(cfg.Dim, 9.9),
		Out:    fill(tensor.Shape{cfg.Dim, cfg.Vocab}, 0.4, 0.5),
	}
	for l := range cfg.Layers {
		fl := 20 * float64(l)
		layer := &nlp.JambaLayer{
			InputNorm: rms(cfg.Dim, fl+0.2),
			PreFFNorm: rms(cfg.Dim, fl+0.5),
		}
		// Interleave both mixer families and both FFN families so a single benchmark exercises
		// every branch DecodeStep can take.
		if l%2 == 0 {
			layer.Attn = attn(fl)
		} else {
			layer.Mamba = mixer(fl)
		}
		if l%2 == 0 {
			layer.Dense = dense(fl + 11)
		} else {
			layer.MoE = moe(fl + 13)
		}
		m.Layers = append(m.Layers, layer)
	}
	return m
}

// BenchmarkJambaDecode measures one hybrid decode step: two attention layers reading a warmed KV
// cache and two Mamba layers stepping their conv window and SSM state.
func BenchmarkJambaDecode(b *testing.B) {
	m := benchJambaModel(256, 512, 16, 4, 64, 512, 4, 4, 128, 8, 4)
	ctx := backend.NewContext()
	st := m.NewDecodeState()
	// Warm the state: the conv window must be full rather than zero-padded, and the KV cache must
	// hold rows, or the step measured is not the steady-state one.
	for i := range 8 {
		if _, err := m.DecodeStep(ctx, st, i%128); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := m.DecodeStep(ctx, st, (i*7)%128); err != nil {
			b.Fatal(err)
		}
	}
}
