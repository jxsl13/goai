package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// RWKVFromHF builds an [RWKV] from a Hugging Face RwkvForCausalLM checkpoint's
// tensor map (as loaded by safetensors.Load / .LoadFile). It is RWKV-4 — the
// only RWKV variant `transformers` ships — and it is a pure LOADER: nn.RWKVBlock
// already reproduces modeling_rwkv's rwkv_linear_attention_cpu and the time/
// channel-mixing math exactly, so no block-level adaptation is needed, only a
// weight-name mapping. Every dimension is inferred from the tensors; only
// cfg.Eps is honoured (default 1e-5).
//
// Weight mapping (verified against transformers.models.rwkv.modeling_rwkv):
//
//   - rwkv.embeddings.weight [vocab, dim] → Embed (no transpose). head.weight
//     [vocab, dim] is UNTIED → transpose to Head [dim, vocab].
//   - rwkv.blocks.0.pre_ln → the model-level PreLN ("ln0"), applied once before
//     block 0. Each block's ln1/ln2 → the block's LN1/LN2. HF LayerNorm has a
//     bias, so Gamma=weight, Beta=bias.
//   - TIME-MIXING (attention.*): time_mix_{key,value,receptance} are [1,1,dim];
//     squeeze to [dim] → MuK/MuV/MuR. nn's mix μ⊙x+(1−μ)⊙shift matches HF's
//     hidden·time_mix + shifted·(1−time_mix) with the SAME μ (no 1−μ flip). The
//     torch [out,in] {key,value,receptance,output}.weight → transpose → Wk/Wv/
//     Wr/Wo [dim,dim] (bias-free).
//   - WKV decay/bonus: nn forms w = exp(WLog) and decays the state by pp−w,
//     while HF forms −exp(time_decay) and updates by max_state+(−exp(time_decay))
//     = max_state−exp(time_decay). These are identical with WLog = time_decay
//     (NO negation at load — the sign lives in nn's subtract). time_first is the
//     current-token bonus added to the key (ww = u+k in both), so U = time_first.
//   - CHANNEL-MIXING (feed_forward.*): time_mix_{key,receptance} [1,1,dim] →
//     CMuK/CMuR. key.weight [hidden,dim]→transpose→CWk [dim,hidden]; value.weight
//     [dim,hidden]→transpose→CWv [hidden,dim]; receptance.weight [dim,dim]→
//     transpose→CWr. The block applies square(relu(CWk··)) then CWv, gated by
//     σ(CWr··) — exactly RwkvFeedForward.
//   - rwkv.ln_out → LNOut (final LayerNorm before the head).
//
// The WKV attention width equals dim (nn.RWKVBlock's projections are square);
// checkpoints whose attention_hidden_size differs from hidden_size are rejected.
func RWKVFromHF(ts map[string]*tensor.Tensor, cfg RWKVConfig) (*RWKV, error) {
	emb, ok := ts["rwkv.embeddings.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF RWKV missing rwkv.embeddings.weight")
	}
	if emb.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: rwkv.embeddings.weight must be 2-D, got %v", emb.Shape())
	}
	cfg.Vocab, cfg.Dim = emb.Shape()[0], emb.Shape()[1]
	if cfg.Eps == 0 {
		cfg.Eps = 1e-5
	}

	head, ok := ts["head.weight"]
	if !ok {
		return nil, fmt.Errorf("nlp: HF RWKV missing head.weight (untied LM head)")
	}

	preLN, ok := lnFromHF(ts, "rwkv.blocks.0.pre_ln", cfg.Eps)
	if !ok {
		return nil, fmt.Errorf("nlp: HF RWKV missing rwkv.blocks.0.pre_ln (ln0)")
	}
	lnOut, ok := lnFromHF(ts, "rwkv.ln_out", cfg.Eps)
	if !ok {
		return nil, fmt.Errorf("nlp: HF RWKV missing rwkv.ln_out")
	}

	// Count layers by probing rwkv.blocks.N.attention.time_decay.
	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("rwkv.blocks.%d.attention.time_decay", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF RWKV has no rwkv.blocks.*")
	}
	cfg.Layers = layers

	m := &RWKV{
		Config: cfg,
		Embed:  cloneF64(emb),     // [vocab,dim], no transpose
		PreLN:  preLN,             // ln0
		LNOut:  lnOut,             // final LayerNorm
		Head:   transpose2D(head), // [dim,vocab], untied
	}

	for l := range layers {
		p := fmt.Sprintf("rwkv.blocks.%d.", l)
		g := func(name string) (*tensor.Tensor, error) {
			t, ok := ts[p+name]
			if !ok {
				return nil, fmt.Errorf("nlp: HF RWKV missing %s%s", p, name)
			}
			return t, nil
		}
		ln1, ok := lnFromHF(ts, p+"ln1", cfg.Eps)
		if !ok {
			return nil, fmt.Errorf("nlp: HF RWKV missing %sln1", p)
		}
		ln2, ok := lnFromHF(ts, p+"ln2", cfg.Eps)
		if !ok {
			return nil, fmt.Errorf("nlp: HF RWKV missing %sln2", p)
		}

		timeDecay, err := g("attention.time_decay")
		if err != nil {
			return nil, err
		}
		timeFirst, err := g("attention.time_first")
		if err != nil {
			return nil, err
		}
		tmK, err := g("attention.time_mix_key")
		if err != nil {
			return nil, err
		}
		tmV, err := g("attention.time_mix_value")
		if err != nil {
			return nil, err
		}
		tmR, err := g("attention.time_mix_receptance")
		if err != nil {
			return nil, err
		}
		attKey, err := g("attention.key.weight")
		if err != nil {
			return nil, err
		}
		attVal, err := g("attention.value.weight")
		if err != nil {
			return nil, err
		}
		attRec, err := g("attention.receptance.weight")
		if err != nil {
			return nil, err
		}
		attOut, err := g("attention.output.weight")
		if err != nil {
			return nil, err
		}
		ffTmK, err := g("feed_forward.time_mix_key")
		if err != nil {
			return nil, err
		}
		ffTmR, err := g("feed_forward.time_mix_receptance")
		if err != nil {
			return nil, err
		}
		ffKey, err := g("feed_forward.key.weight")
		if err != nil {
			return nil, err
		}
		ffVal, err := g("feed_forward.value.weight")
		if err != nil {
			return nil, err
		}
		ffRec, err := g("feed_forward.receptance.weight")
		if err != nil {
			return nil, err
		}

		// attention.key.weight is [attention_hidden_size, hidden_size]; nn's WKV
		// runs at dim, so the two must match.
		if attKey.Shape()[0] != cfg.Dim {
			return nil, fmt.Errorf("nlp: HF RWKV attention_hidden_size %d != hidden_size %d (unsupported)", attKey.Shape()[0], cfg.Dim)
		}
		cfg.Hidden = ffKey.Shape()[0] // feed_forward.key.weight is [intermediate, hidden]

		b := &nn.RWKVBlock{
			Dim: cfg.Dim,
			LN1: ln1, LN2: ln2,
			MuR: squeezeMix(tmR), MuK: squeezeMix(tmK), MuV: squeezeMix(tmV),
			Wr:   &nn.Linear{W: transpose2D(attRec)}, // [dim,dim], no bias
			Wk:   &nn.Linear{W: transpose2D(attKey)},
			Wv:   &nn.Linear{W: transpose2D(attVal)},
			Wo:   &nn.Linear{W: transpose2D(attOut)},
			WLog: cloneF64(timeDecay), // w = exp(WLog); nn subtracts → HF's −exp(time_decay)
			U:    cloneF64(timeFirst), // current-token bonus
			CMuR: squeezeMix(ffTmR), CMuK: squeezeMix(ffTmK),
			CWr: &nn.Linear{W: transpose2D(ffRec)}, // [dim,dim]
			CWk: &nn.Linear{W: transpose2D(ffKey)}, // [dim,hidden]
			CWv: &nn.Linear{W: transpose2D(ffVal)}, // [hidden,dim]
		}
		m.Blocks = append(m.Blocks, b)
	}
	m.Config = cfg
	return m, nil
}

// lnFromHF builds an nn.LayerNorm from a HF "<prefix>.weight"/"<prefix>.bias"
// pair (Gamma=weight, Beta=bias). Returns ok=false if the weight is absent.
func lnFromHF(ts map[string]*tensor.Tensor, prefix string, eps float64) (*nn.LayerNorm, bool) {
	g, ok := ts[prefix+".weight"]
	if !ok {
		return nil, false
	}
	b, ok := ts[prefix+".bias"]
	if !ok {
		return nil, false
	}
	return &nn.LayerNorm{Gamma: cloneF64(g), Beta: cloneF64(b), Eps: eps}, true
}

// squeezeMix turns a HF time-mix tensor [1,1,dim] into the [dim] vector that
// nn.RWKVBlock's Mu* fields expect (storage order is unchanged by dropping the
// size-1 leading axes).
func squeezeMix(t *tensor.Tensor) *tensor.Tensor {
	d := t.Shape()[t.Ndim()-1]
	out := tensor.New(tensor.F64, tensor.Shape{d})
	copy(out.Storage().F64(), cloneF64(t).Storage().F64())
	return out
}
