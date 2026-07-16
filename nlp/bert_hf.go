package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BertFromHF builds a [Bert] from a Hugging Face BertModel checkpoint's tensor
// map (as loaded by safetensors.Load or pytorch.Load). cfg supplies Heads and
// Eps (from config.json — num_attention_heads and layer_norm_eps, BERT's default
// 1e-12); the dimensions (Vocab, Dim, MaxPos, TypeVocab, Layers) are inferred
// from the tensors. HF stores each dense weight as torch [out, in]; GoAI wants
// [in, out], so every projection is transposed (biases pass through).
func BertFromHF(ts map[string]*tensor.Tensor, cfg BertConfig) (*Bert, error) {
	get := func(name string) (*tensor.Tensor, error) {
		if t, ok := ts[name]; ok {
			return t, nil
		}
		return nil, fmt.Errorf("nlp: HF BERT missing %s", name)
	}
	we, err := get("embeddings.word_embeddings.weight")
	if err != nil {
		return nil, err
	}
	pe, err := get("embeddings.position_embeddings.weight")
	if err != nil {
		return nil, err
	}
	te, err := get("embeddings.token_type_embeddings.weight")
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 {
		return nil, fmt.Errorf("nlp: BertFromHF needs cfg.Heads")
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-12
	}
	cfg.Vocab, cfg.Dim = we.Shape()[0], we.Shape()[1]
	cfg.MaxPos = pe.Shape()[0]
	cfg.TypeVocab = te.Shape()[0]

	// GoAI's projections are F64 (transpose2D widens); keep the whole model F64
	// for a consistent dtype through the forward (matching the Llama converters).
	ln := func(prefix string) (*nn.LayerNorm, error) {
		g, err := get(prefix + ".weight")
		if err != nil {
			return nil, err
		}
		b, err := get(prefix + ".bias")
		if err != nil {
			return nil, err
		}
		return &nn.LayerNorm{Gamma: cloneF64(g), Beta: cloneF64(b), Eps: cfg.Eps}, nil
	}

	embLN, err := ln("embeddings.LayerNorm")
	if err != nil {
		return nil, err
	}
	m := &Bert{Config: cfg, TokEmb: cloneF64(we), PosEmb: cloneF64(pe), SegEmb: cloneF64(te), EmbLN: embLN}

	// Count layers.
	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("encoder.layer.%d.attention.self.query.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF BERT has no encoder.layer.*")
	}
	cfg.Layers = layers
	m.Config = cfg

	for l := range layers {
		p := fmt.Sprintf("encoder.layer.%d.", l)
		proj := func(name string) (w, b *tensor.Tensor, err error) {
			if w, err = get(p + name + ".weight"); err != nil {
				return
			}
			b, err = get(p + name + ".bias")
			return
		}
		qw, qb, err := proj("attention.self.query")
		if err != nil {
			return nil, err
		}
		kw, kb, err := proj("attention.self.key")
		if err != nil {
			return nil, err
		}
		vw, vb, err := proj("attention.self.value")
		if err != nil {
			return nil, err
		}
		ow, ob, err := proj("attention.output.dense")
		if err != nil {
			return nil, err
		}
		attn, err := NewMHA(cfg.Heads, transpose2D(qw), transpose2D(kw), transpose2D(vw), transpose2D(ow))
		if err != nil {
			return nil, err
		}
		attn.Bias = map[string]*tensor.Tensor{"q": cloneF64(qb), "k": cloneF64(kb), "v": cloneF64(vb), "o": cloneF64(ob)}
		attnLN, err := ln(p + "attention.output.LayerNorm")
		if err != nil {
			return nil, err
		}
		iw, ib, err := proj("intermediate.dense")
		if err != nil {
			return nil, err
		}
		dw, db, err := proj("output.dense")
		if err != nil {
			return nil, err
		}
		ffnLN, err := ln(p + "output.LayerNorm")
		if err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, &BertLayer{
			Attn: attn, AttnLN: attnLN,
			W1: transpose2D(iw), B1: cloneF64(ib), W2: transpose2D(dw), B2: cloneF64(db), FFNLN: ffnLN,
		})
	}
	return m, nil
}

// RobertaFromHF builds a [Bert] from a Hugging Face RobertaModel checkpoint. The
// RoBERTa encoder is architecturally identical to BERT (same tensor names), but
// its position ids start at padding_idx+1 (2), so it sets cfg.PosOffset. RoBERTa
// uses a single token-type, so segment ids are all 0 (pass segments=nil to
// [Bert.Forward]). The tokenizer differs (byte-level BPE, see NewBPE), not the
// model.
func RobertaFromHF(ts map[string]*tensor.Tensor, cfg BertConfig) (*Bert, error) {
	if cfg.PosOffset == 0 {
		cfg.PosOffset = 2 // padding_idx (1) + 1
	}
	return BertFromHF(ts, cfg)
}
