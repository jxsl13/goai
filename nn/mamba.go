package nn

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// MambaBlock is the Mamba selective-state-space block (Gu & Dao 2023,
// arXiv:2312.00752 §3.4, §R77) — a linear-time replacement for a transformer's
// attention + FFN. A token sequence is projected up, mixed by a short causal
// depthwise convolution, passed through the input-dependent selective scan (the
// "attention-free" sequence mixer), gated, and projected back:
//
//	x, z = split(in_proj(u))                     // up-projection + gate branch
//	x    = SiLU(causal_conv1d(x))                // local mixing
//	Δ    = softplus(dt_proj(dt_low_proj(x)))     // input-dependent step
//	B, C = b_proj(x), c_proj(x)                  // input-dependent SSM params
//	y    = selective_scan(x, Δ, −exp(A_log), B, C, D)
//	out  = out_proj( y ⊙ SiLU(z) )               // gate + down-projection
//
// It is built entirely from differentiable ops (OpConv1D, OpSoftplus, OpSSM,
// matmul, SiLU, mul), so the whole block trains end to end. Structure confirmed
// against the official mamba_simple.py; this from-scratch version uses biased
// projections rather than mamba's bias-free ones (a harmless extra parameter).
type MambaBlock struct {
	InX, InZ            *Linear        // in_proj split into the x and gate branches
	ConvW               *tensor.Tensor // [d_inner, d_conv] depthwise filters
	ConvB               *tensor.Tensor // [d_inner]
	DtLow, BProj, CProj *Linear        // x_proj split: low-rank Δ, B, C
	DtProj              *Linear        // dt_rank → d_inner (Δ up-projection)
	ALog, Dskip         *tensor.Tensor // [d_inner, N] state, [d_inner] skip
	OutProj             *Linear        // d_inner → d_model
	DModel              int            // model/embedding width
	DInner              int            // inner expanded width (e·DModel)
	DConv               int            // depthwise causal-conv kernel width
	N                   int            // SSM state size
	DtRank              int            // Δ low-rank projection rank
}

// NewMambaBlock builds a Mamba block for model dim dModel, state size n, conv
// width dConv, and expansion factor e (d_inner = e·dModel). dt_rank =
// ceil(dModel/16). Deterministic via seed.
func NewMambaBlock(dtype tensor.Dtype, dModel, n, dConv, e int, seed uint64) (*MambaBlock, error) {
	if dModel <= 0 || n <= 0 || dConv <= 0 || e <= 0 {
		return nil, fmt.Errorf("nn: MambaBlock dims must be positive")
	}
	dInner := e * dModel
	dtRank := (dModel + 15) / 16 // ceil(dModel/16), ≥1 since dModel≥1
	cw := tensor.New(dtype, tensor.Shape{dInner, dConv})
	XavierUniform(cw, dConv, 1, seed+10)
	cb := tensor.New(dtype, tensor.Shape{dInner})
	Zeros(cb)
	alog := tensor.New(dtype, tensor.Shape{dInner, n})
	XavierUniform(alog, dInner, n, seed+11) // A = −exp(A_log) < 0
	dskip := tensor.New(dtype, tensor.Shape{dInner})
	Zeros(dskip)
	return &MambaBlock{
		InX: NewLinear(dtype, dModel, dInner, seed), InZ: NewLinear(dtype, dModel, dInner, seed+1),
		ConvW: cw, ConvB: cb,
		DtLow: NewLinear(dtype, dInner, dtRank, seed+2),
		BProj: NewLinear(dtype, dInner, n, seed+3), CProj: NewLinear(dtype, dInner, n, seed+4),
		DtProj:  NewLinear(dtype, dtRank, dInner, seed+5),
		ALog:    alog,
		Dskip:   dskip,
		OutProj: NewLinear(dtype, dInner, dModel, seed+6),
		DModel:  dModel, DInner: dInner, DConv: dConv, N: n, DtRank: dtRank,
	}, nil
}

func (m *MambaBlock) exec(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Forward runs the block on u[L, dModel], returning out[L, dModel].
func (m *MambaBlock) Forward(ctx *backend.Context, u *tensor.Tensor) (*tensor.Tensor, error) {
	xin, err := m.InX.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	z, err := m.InZ.Forward(ctx, u)
	if err != nil {
		return nil, err
	}
	// causal depthwise conv + SiLU
	//perfscan:ignore PS3024 OpConv1D graph dispatch, full backend kernel, autograd-recorded
	xc, err := m.exec(ctx, backend.OpConv1D, nil, xin, m.ConvW, m.ConvB)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpSiLU graph dispatch, full backend kernel
	if xc, err = m.exec(ctx, backend.OpSiLU, nil, xc); err != nil {
		return nil, err
	}
	// input-dependent Δ, B, C
	dtLow, err := m.DtLow.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	dtPre, err := m.DtProj.Forward(ctx, dtLow)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpSoftplus graph dispatch, full backend kernel
	delta, err := m.exec(ctx, backend.OpSoftplus, nil, dtPre)
	if err != nil {
		return nil, err
	}
	bMat, err := m.BProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	cMat, err := m.CProj.Forward(ctx, xc)
	if err != nil {
		return nil, err
	}
	// A = −exp(A_log)
	//perfscan:ignore PS3024 OpExp graph dispatch, full backend kernel
	expA, err := m.exec(ctx, backend.OpExp, nil, m.ALog)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpNeg graph dispatch, full backend kernel
	A, err := m.exec(ctx, backend.OpNeg, nil, expA)
	if err != nil {
		return nil, err
	}
	// selective scan
	//perfscan:ignore PS3024 OpSSM graph dispatch, full backend kernel
	y, err := m.exec(ctx, backend.OpSSM, nil, xc, delta, A, bMat, cMat, m.Dskip)
	if err != nil {
		return nil, err
	}
	// gate y ⊙ SiLU(z), then down-project
	//perfscan:ignore PS3024 OpSiLU graph dispatch, full backend kernel
	gate, err := m.exec(ctx, backend.OpSiLU, nil, z)
	if err != nil {
		return nil, err
	}
	//perfscan:ignore PS3024 OpMul graph dispatch, full backend kernel
	if y, err = m.exec(ctx, backend.OpMul, nil, y, gate); err != nil {
		return nil, err
	}
	return m.OutProj.Forward(ctx, y)
}

// Params returns every trainable tensor of the block.
func (m *MambaBlock) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.ConvW, m.ConvB, m.ALog, m.Dskip}
	for _, l := range []*Linear{m.InX, m.InZ, m.DtLow, m.BProj, m.CProj, m.DtProj, m.OutProj} {
		ps = append(ps, l.Params()...)
	}
	return ps
}
