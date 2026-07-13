package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MLA is a Multi-head Latent Attention layer (DeepSeek-V2, Liu et al. 2024,
// arXiv:2405.04434, §R74). It slashes the KV-cache by compressing keys and values
// through a shared low-rank latent instead of caching full per-head K/V:
//
//	c^KV = h·W_DKV                    (latent, dim dc — the ONLY thing cached)
//	k^C  = c^KV·W_UK,  v^C = c^KV·W_UV
//	k^R  = RoPE(h·W_KR)               (shared decoupled-RoPE key, dim dR)
//	c^Q  = h·W_DQ,  q^C = c^Q·W_UQ,  q^R = RoPE(c^Q·W_QR)   (per head, dim dR)
//	score = (q^C·k^C + q^R·k^R)/√(dh+dR);  o = softmax(score)·v^C;  u = o·W_O
//
// The decoupled RoPE lives in a separate small dimension because RoPE's
// position-dependence cannot be absorbed into the low-rank up-projection. This is
// the naive (un-absorbed) forward — numerically identical to the inference weight-
// absorption trick, which is a memory optimization left as a follow-up. The
// content/RoPE-score attention (with the per-head RoPE applied internally) is the
// fused OpMLA; the projections are plain matmuls, so the whole layer is
// differentiable via existing VJPs. Cached per token: dc + dR floats vs the
// heads·2·dh of standard MHA.
type MLA struct {
	WDKV, WUK, WUV *tensor.Tensor // KV down/up projections
	WKR            *tensor.Tensor // shared decoupled-RoPE key projection [d, dR]
	WDQ, WUQ, WQR  *tensor.Tensor // query down / content-up / RoPE-up projections
	WO             *tensor.Tensor // output projection [heads*dh, d]
	Heads          int            // attention heads
	Dh             int            // per-head (content) dimension
	DR             int            // decoupled-RoPE key dimension (even)
	Causal         bool           // apply autoregressive (causal) mask
}

// NewMLA builds an MLA layer for hidden dim d, `heads` heads of content dim dh,
// decoupled-RoPE dim dR (even), and latent ranks dc (KV) and dcq (query).
func NewMLA(dtype tensor.Dtype, d, heads, dh, dR, dc, dcq int, causal bool, seed uint64) (*MLA, error) {
	if dR%2 != 0 || dR <= 0 {
		return nil, fmt.Errorf("nn: MLA rope dim %d must be even and > 0", dR)
	}
	hdh := heads * dh
	mk := func(r, c int, s uint64) *tensor.Tensor {
		w := tensor.New(dtype, tensor.Shape{r, c})
		XavierUniform(w, r, c, s)
		return w
	}
	return &MLA{
		WDKV: mk(d, dc, seed), WUK: mk(dc, hdh, seed+1), WUV: mk(dc, hdh, seed+2),
		WKR: mk(d, dR, seed+3),
		WDQ: mk(d, dcq, seed+4), WUQ: mk(dcq, hdh, seed+5), WQR: mk(dcq, heads*dR, seed+6),
		WO:    mk(hdh, d, seed+7),
		Heads: heads, Dh: dh, DR: dR, Causal: causal,
	}, nil
}

func (m *MLA) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward computes MLA for h[seq, d], returning u[seq, d].
func (m *MLA) Forward(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	mm := func(x, w *tensor.Tensor) (*tensor.Tensor, error) { return m.exec(ctx, backend.OpMatMul, nil, x, w) }

	cKV, err := mm(h, m.WDKV)
	if err != nil {
		return nil, err
	}
	kC, err := mm(cKV, m.WUK)
	if err != nil {
		return nil, err
	}
	vC, err := mm(cKV, m.WUV)
	if err != nil {
		return nil, err
	}
	kRpre, err := mm(h, m.WKR)
	if err != nil {
		return nil, err
	}
	cQ, err := mm(h, m.WDQ)
	if err != nil {
		return nil, err
	}
	qC, err := mm(cQ, m.WUQ)
	if err != nil {
		return nil, err
	}
	qRpre, err := mm(cQ, m.WQR)
	if err != nil {
		return nil, err
	}

	o, err := m.exec(ctx, backend.OpMLA, backend.MLAAttrs{Heads: m.Heads, Causal: m.Causal},
		qC, kC, vC, qRpre, kRpre)
	if err != nil {
		return nil, err
	}
	return mm(o, m.WO)
}

// Params returns the eight projection matrices.
func (m *MLA) Params() []*tensor.Tensor {
	return []*tensor.Tensor{m.WDKV, m.WUK, m.WUV, m.WKR, m.WDQ, m.WUQ, m.WQR, m.WO}
}

// CacheBytesPerToken reports the per-token KV-cache size (in float elements) of
// MLA versus standard MHA: MLA caches only the latent c^KV (dc) plus the shared
// decoupled key k^R (dR), whereas MHA caches full per-head keys and values
// (heads·2·dh). Illustrates MLA's ~90%+ cache reduction.
func (m *MLA) CacheBytesPerToken(dc int) (mla, mha int) {
	return dc + m.DR, m.Heads * 2 * m.Dh
}
