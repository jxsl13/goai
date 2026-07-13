package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// FSQ implements Finite Scalar Quantization (Mentzer, Minnen, Agustsson &
// Tschannen 2023, "Finite Scalar Quantization: VQ-VAE Made Simple",
// arXiv:2309.15505, ICLR'24). It is a drop-in replacement for the VQ-VAE codebook
// (nn.VectorQuantizer) that removes the learned codebook entirely — and with it the
// codebook-collapse problem and the auxiliary commitment/codebook losses. Each of
// the d latent channels is squashed by a bounding function and ROUNDED to one of L_i
// integer levels; the implicit codebook is the Cartesian product of the per-channel
// level sets, of size ∏_i L_i, stored as NOTHING.
//
// Per channel i with L_i levels (half-width h = (L_i−1)/2, offset o = 0 for odd L_i
// else ½, shift s = atanh(o/h)):
//
//	bounded = h·tanh(z + s) − o          (lands in [−h−o, h−o] so round hits L_i levels)
//	ẑ       = bounded + sg(round(bounded) − bounded)   (round with a straight-through grad)
//	code    = ẑ / ⌊L_i/2⌋                 (normalized to ≈[−1,1], per the reference)
//
// The round uses the StopGradient op (§T248) so gradients pass straight through the
// tanh to the encoder. There are NO learned parameters and NO auxiliary loss.
type FSQ struct {
	Levels []int     // L_i per channel; total implicit codebook size = ∏ Levels
	half   []float64 // (L_i−1)/2
	offset []float64 // 0 (odd L_i) or ½ (even L_i)
	shift  []float64 // atanh(offset/half)
}

// NewFSQ builds an FSQ quantizer for the given per-channel level counts (each ≥ 2).
func NewFSQ(levels []int) (*FSQ, error) {
	if len(levels) == 0 {
		return nil, fmt.Errorf("nn: FSQ needs ≥1 channel level")
	}
	f := &FSQ{
		Levels: append([]int(nil), levels...),
		half:   make([]float64, len(levels)),
		offset: make([]float64, len(levels)),
		shift:  make([]float64, len(levels)),
	}
	for i, l := range levels {
		if l < 2 {
			return nil, fmt.Errorf("nn: FSQ channel %d has %d levels, want ≥2", i, l)
		}
		f.half[i] = float64(l-1) / 2
		if l%2 == 0 {
			f.offset[i] = 0.5
			f.shift[i] = math.Atanh(f.offset[i] / f.half[i])
		}
	}
	return f, nil
}

// CodebookSize returns the size of the implicit codebook, ∏_i L_i.
func (f *FSQ) CodebookSize() int {
	n := 1
	for _, l := range f.Levels {
		n *= l
	}
	return n
}

// Params returns no parameters — FSQ has no learned codebook.
func (f *FSQ) Params() []*tensor.Tensor { return nil }

// Quantize bounds and rounds z [batch, d] (d == len(Levels)) to the finite scalar
// grid and returns the normalized codes [batch, d] (differentiable w.r.t. z via the
// straight-through round) and the flat codebook indices [batch] ∈ [0, ∏L_i).
func (f *FSQ) Quantize(ctx *backend.Context, z *tensor.Tensor) (codes *tensor.Tensor, indices []int, err error) {
	if z.Ndim() != 2 {
		return nil, nil, fmt.Errorf("nn: FSQ.Quantize wants rank-2 [batch,d], got %v", z.Shape())
	}
	batch, d := z.Shape()[0], z.Shape()[1]
	if d != len(f.Levels) {
		return nil, nil, fmt.Errorf("nn: FSQ channels %d != len(Levels) %d", d, len(f.Levels))
	}
	ex := func(op backend.Op, at backend.Attrs, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, e := backend.Execute(ctx, op, in, at)
		if e != nil {
			return nil, e
		}
		return o[0], nil
	}
	row := func(vals []float64) *tensor.Tensor { // [1,d] per-channel constant to broadcast
		t := tensor.New(z.Dtype(), tensor.Shape{1, d})
		for i, v := range vals {
			t.SetF64(v, 0, i)
		}
		return t
	}
	// normalization divides by the half-width ⌊L/2⌋ (reference: quantized/(levels//2)),
	// which differs from the bounding half (L−1)/2 for even L.
	invHalf := make([]float64, d)
	for i := range f.Levels {
		invHalf[i] = 1 / float64(f.Levels[i]/2)
	}
	// bounded = half·tanh(z + shift) − offset  (differentiable).
	shifted, err := ex(backend.OpAdd, nil, z, row(f.shift))
	if err != nil {
		return nil, nil, err
	}
	th, err := ex(backend.OpTanh, nil, shifted)
	if err != nil {
		return nil, nil, err
	}
	scaled, err := ex(backend.OpMul, nil, th, row(f.half))
	if err != nil {
		return nil, nil, err
	}
	bounded, err := ex(backend.OpSub, nil, scaled, row(f.offset))
	if err != nil {
		return nil, nil, err
	}
	// round with a straight-through gradient, and read the integer level per channel.
	rounded := tensor.New(z.Dtype(), z.Shape())
	indices = make([]int, batch)
	for b := range batch {
		flat, stride := 0, 1
		for i := range d {
			r := math.Round(bounded.AtF64(b, i))
			rounded.SetF64(r, b, i)
			level := int(r) + f.Levels[i]/2 // shift symmetric level into [0, L_i)
			flat += level * stride
			stride *= f.Levels[i]
		}
		indices[b] = flat
	}
	delta, err := ex(backend.OpStopGradient, nil, mustSub(ctx, rounded, bounded))
	if err != nil {
		return nil, nil, err
	}
	zhat, err := ex(backend.OpAdd, nil, bounded, delta) // forward = rounded; grad flows via bounded
	if err != nil {
		return nil, nil, err
	}
	if codes, err = ex(backend.OpMul, nil, zhat, row(invHalf)); err != nil { // normalize to ≈[−1,1]
		return nil, nil, err
	}
	return codes, indices, nil
}
