package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// GPT2FromHF builds a GPT from a HuggingFace GPT-2 checkpoint's tensor map
// (§T506 — possible since §T505 made the biased attention projections
// representable). It reads the HF naming (wte/wpe, h.N.ln_1, h.N.attn.c_attn …)
// and performs the two structural conversions GPT-2 needs:
//
//   - the fused c_attn [d, 3d] (+bias [3d]) is SPLIT column-wise into Wq/Wk/Wv
//     and their biases — HF's Conv1D stores weights [in, out], matching this
//     library's layout, so no transposes are needed for the projections;
//   - the LM head is TIED to the token embedding: head = wteᵀ.
//
// Geometry (vocab, context, dim, layers) is inferred from the tensor shapes;
// heads must be supplied (GPT-2: 12 for 124M). Eps is GPT-2's 1e-5.
func GPT2FromHF(ts map[string]*tensor.Tensor, heads int) (*GPT, error) {
	wte, wpe := ts["wte.weight"], ts["wpe.weight"]
	if wte == nil || wpe == nil {
		return nil, fmt.Errorf("nlp: GPT2FromHF missing wte.weight / wpe.weight")
	}
	if wte.Ndim() != 2 || wpe.Ndim() != 2 || wte.Shape()[1] != wpe.Shape()[1] {
		return nil, fmt.Errorf("nlp: GPT2FromHF wte %v / wpe %v inconsistent", wte.Shape(), wpe.Shape())
	}
	vocab, d := wte.Shape()[0], wte.Shape()[1]
	ctxLen := wpe.Shape()[0]
	layers := 0
	for {
		if ts[fmt.Sprintf("h.%d.ln_1.weight", layers)] == nil {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: GPT2FromHF found no h.N blocks")
	}

	// tied head: head[d, vocab] = wteᵀ (typed transpose, §base-perf).
	head := tensor.New(wte.Dtype(), tensor.Shape{d, vocab})
	wtec := wte.Contiguous()
	switch wtec.Dtype() {
	case tensor.F64:
		src, dst := wtec.Storage().F64(), head.Storage().F64()
		for vv := 0; vv < vocab; vv++ {
			row := vv * d
			for j := 0; j < d; j++ {
				dst[j*vocab+vv] = src[row+j]
			}
		}
	case tensor.F32:
		src, dst := wtec.Storage().F32(), head.Storage().F32()
		for vv := 0; vv < vocab; vv++ {
			row := vv * d
			for j := 0; j < d; j++ {
				dst[j*vocab+vv] = src[row+j]
			}
		}
	default:
		for vv := 0; vv < vocab; vv++ {
			for j := 0; j < d; j++ {
				head.SetF64(wtec.AtF64(vv, j), j, vv)
			}
		}
	}

	out := map[string]*tensor.Tensor{
		"tok_emb": wte, "pos_emb": wpe, "head": head,
	}
	need := func(name string) (*tensor.Tensor, error) {
		t := ts[name]
		if t == nil {
			return nil, fmt.Errorf("nlp: GPT2FromHF missing %s", name)
		}
		return t, nil
	}
	// column slice of a [rows, W] tensor → [rows, hi-lo] (typed row copy, §base-perf).
	cols := func(t *tensor.Tensor, lo, hi int) *tensor.Tensor {
		rows, w := t.Shape()[0], t.Shape()[1]
		cw := hi - lo
		o := tensor.New(t.Dtype(), tensor.Shape{rows, cw})
		tc := t.Contiguous()
		switch tc.Dtype() {
		case tensor.F64:
			src, dst := tc.Storage().F64(), o.Storage().F64()
			for i := 0; i < rows; i++ {
				copy(dst[i*cw:i*cw+cw], src[i*w+lo:i*w+hi])
			}
		case tensor.F32:
			src, dst := tc.Storage().F32(), o.Storage().F32()
			for i := 0; i < rows; i++ {
				copy(dst[i*cw:i*cw+cw], src[i*w+lo:i*w+hi])
			}
		default:
			for i := 0; i < rows; i++ {
				for j := lo; j < hi; j++ {
					o.SetF64(tc.AtF64(i, j), i, j-lo)
				}
			}
		}
		return o
	}
	seg := func(t *tensor.Tensor, lo, hi int) *tensor.Tensor {
		o := tensor.New(t.Dtype(), tensor.Shape{hi - lo})
		tc := t.Contiguous()
		switch tc.Dtype() {
		case tensor.F64:
			copy(o.Storage().F64(), tc.Storage().F64()[lo:hi])
		case tensor.F32:
			copy(o.Storage().F32(), tc.Storage().F32()[lo:hi])
		default:
			for j := lo; j < hi; j++ {
				o.SetF64(tc.AtF64(j), j-lo)
			}
		}
		return o
	}

	type attnBias struct{ q, k, v, o *tensor.Tensor }
	biases := make([]attnBias, layers)
	for l := 0; l < layers; l++ {
		p := fmt.Sprintf("h.%d.", l)
		ln1w, err := need(p + "ln_1.weight")
		if err != nil {
			return nil, err
		}
		ln1b, err := need(p + "ln_1.bias")
		if err != nil {
			return nil, err
		}
		cattnW, err := need(p + "attn.c_attn.weight")
		if err != nil {
			return nil, err
		}
		cattnB, err := need(p + "attn.c_attn.bias")
		if err != nil {
			return nil, err
		}
		if cattnW.Ndim() != 2 || cattnW.Shape()[0] != d || cattnW.Shape()[1] != 3*d ||
			cattnB.Ndim() != 1 || cattnB.Shape()[0] != 3*d {
			return nil, fmt.Errorf("nlp: GPT2FromHF %sattn.c_attn shapes %v/%v, want [%d,%d]/[%d]",
				p, cattnW.Shape(), cattnB.Shape(), d, 3*d, 3*d)
		}
		cprojW, err := need(p + "attn.c_proj.weight")
		if err != nil {
			return nil, err
		}
		cprojB, err := need(p + "attn.c_proj.bias")
		if err != nil {
			return nil, err
		}
		ln2w, err := need(p + "ln_2.weight")
		if err != nil {
			return nil, err
		}
		ln2b, err := need(p + "ln_2.bias")
		if err != nil {
			return nil, err
		}
		fcW, err := need(p + "mlp.c_fc.weight")
		if err != nil {
			return nil, err
		}
		fcB, err := need(p + "mlp.c_fc.bias")
		if err != nil {
			return nil, err
		}
		mprojW, err := need(p + "mlp.c_proj.weight")
		if err != nil {
			return nil, err
		}
		mprojB, err := need(p + "mlp.c_proj.bias")
		if err != nil {
			return nil, err
		}

		q := fmt.Sprintf("blocks.%d.", l)
		out[q+"ln1.gamma"], out[q+"ln1.beta"] = ln1w, ln1b
		out[q+"attn.wq"] = cols(cattnW, 0, d)
		out[q+"attn.wk"] = cols(cattnW, d, 2*d)
		out[q+"attn.wv"] = cols(cattnW, 2*d, 3*d)
		out[q+"attn.wo"] = cprojW
		out[q+"ln2.gamma"], out[q+"ln2.beta"] = ln2w, ln2b
		out[q+"ffn.w1"], out[q+"ffn.b1"] = fcW, fcB
		out[q+"ffn.w2"], out[q+"ffn.b2"] = mprojW, mprojB
		biases[l] = attnBias{
			q: seg(cattnB, 0, d), k: seg(cattnB, d, 2*d), v: seg(cattnB, 2*d, 3*d), o: cprojB,
		}
	}
	lnfW, err := need("ln_f.weight")
	if err != nil {
		return nil, err
	}
	lnfB, err := need("ln_f.bias")
	if err != nil {
		return nil, err
	}
	out["lnf.gamma"], out["lnf.beta"] = lnfW, lnfB

	g, err := FromSafetensors(GPTConfig{
		Vocab: vocab, Ctx: ctxLen, Dim: d, Heads: heads, Layers: layers, Eps: 1e-5,
	}, out)
	if err != nil {
		return nil, err
	}
	for l, b := range biases { // §T505: the biased projections GPT-2 requires
		g.Blocks[l].Attn.Bias = map[string]*tensor.Tensor{"q": b.q, "k": b.k, "v": b.v, "o": b.o}
	}
	return g, nil
}
