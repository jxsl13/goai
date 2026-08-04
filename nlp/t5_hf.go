package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// T5FromHF builds a [T5] encoder from a Hugging Face T5EncoderModel (or the
// encoder of a T5Model) tensor map. cfg supplies the geometry not derivable from
// the tensors — Heads, HeadDim (d_kv), NumBuckets, MaxDistance, Eps — while
// Vocab, Dim, Layers, FFN and the gated/ReLU FFN variant are inferred. HF stores
// each projection as torch [out, in]; GoAI wants [in, out], so projections are
// transposed. The relative-position bias (an [num_buckets, num_heads] embedding
// on block 0 in HF) loads directly into the shared T5RelativeBias table.
func T5FromHF(ts map[string]*tensor.Tensor, cfg T5Config) (*T5, error) {
	get := func(name string) (*tensor.Tensor, error) {
		if t, ok := ts[name]; ok {
			return t, nil
		}
		return nil, fmt.Errorf("nlp: HF T5 missing %s", name)
	}
	shared, err := get("shared.weight")
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("nlp: T5FromHF needs cfg.Heads and cfg.HeadDim")
	}
	if cfg.NumBuckets == 0 {
		cfg.NumBuckets = 32
	}
	if cfg.MaxDistance == 0 {
		cfg.MaxDistance = 128
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	cfg.Vocab, cfg.Dim = shared.Shape()[0], shared.Shape()[1]

	rms := func(name string) (*nn.RMSNorm, error) {
		g, err := get(name)
		if err != nil {
			return nil, err
		}
		return &nn.RMSNorm{Gamma: cloneF64(g), Eps: cfg.Eps}, nil
	}

	// Relative-position bias table lives on block 0 in HF: [num_buckets, num_heads].
	rbw, err := get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
	if err != nil {
		return nil, err
	}
	relBias, err := nn.NewT5RelativeBias(cfg.NumBuckets, cfg.Heads, cfg.MaxDistance, true, tensor.F64)
	if err != nil {
		return nil, err
	}
	relBias.Table = cloneF64(rbw)

	m := &T5{Config: cfg, Shared: cloneF64(shared), RelBias: relBias}

	// Count layers + detect gated FFN.
	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("encoder.block.%d.layer.0.SelfAttention.q.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF T5 has no encoder.block.*")
	}
	cfg.Layers = layers
	_, gated := ts["encoder.block.0.layer.1.DenseReluDense.wi_0.weight"]
	cfg.Gated = gated
	m.Config = cfg

	//perfscan:ignore PS3060 model-load weight transpose, one-time
	for l := range layers {
		p := fmt.Sprintf("encoder.block.%d.", l)
		w := func(name string) (*tensor.Tensor, error) {
			t, err := get(p + name)
			if err != nil {
				return nil, err
			}
			return transpose2D(t), nil
		}
		wq, err := w("layer.0.SelfAttention.q.weight")
		if err != nil {
			return nil, err
		}
		wk, err := w("layer.0.SelfAttention.k.weight")
		if err != nil {
			return nil, err
		}
		wv, err := w("layer.0.SelfAttention.v.weight")
		if err != nil {
			return nil, err
		}
		wo, err := w("layer.0.SelfAttention.o.weight")
		if err != nil {
			return nil, err
		}
		attnNorm, err := rms(p + "layer.0.layer_norm.weight")
		if err != nil {
			return nil, err
		}
		ffnNorm, err := rms(p + "layer.1.layer_norm.weight")
		if err != nil {
			return nil, err
		}
		blk := &T5Block{AttnNorm: attnNorm, Wq: wq, Wk: wk, Wv: wv, Wo: wo, FFNNorm: ffnNorm}
		if gated {
			if blk.Wi0, err = w("layer.1.DenseReluDense.wi_0.weight"); err != nil {
				return nil, err
			}
			if blk.Wi1, err = w("layer.1.DenseReluDense.wi_1.weight"); err != nil {
				return nil, err
			}
		} else {
			if blk.Wi0, err = w("layer.1.DenseReluDense.wi.weight"); err != nil {
				return nil, err
			}
		}
		if blk.WOut, err = w("layer.1.DenseReluDense.wo.weight"); err != nil {
			return nil, err
		}
		if l == 0 {
			cfg.FFN = blk.WOut.Shape()[0] // [d_ff, dim] after transpose
			m.Config = cfg
		}
		m.Blocks = append(m.Blocks, blk)
	}

	fn, err := rms("encoder.final_layer_norm.weight")
	if err != nil {
		return nil, err
	}
	m.FinalNorm = fn
	return m, nil
}
