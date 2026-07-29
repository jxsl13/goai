package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// SinusoidalPositionalEncoding builds the fixed (non-learned) sinusoidal positional encoding of
// the original Transformer (Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser &
// Polosukhin 2017, "Attention is All You Need", NeurIPS, arXiv:1706.03762, §3.5). It returns a
// [seqLen, dModel] matrix to be ADDED to the token embeddings, encoding each absolute position by
// a bank of sinusoids of geometrically increasing wavelength:
//
//	PE(pos, 2i)   = sin( pos / base^(2i/dModel) )
//	PE(pos, 2i+1) = cos( pos / base^(2i/dModel) )
//
// so each adjacent pair of features (2i, 2i+1) shares one frequency 1/base^(2i/dModel), with sin
// at the even index and cos at the odd — the sin/cos are INTERLEAVED across the feature dimension,
// exactly as the paper's equation is written (some implementations, e.g. tensor2tensor, instead
// concatenate all sines then all cosines — an equivalent-capacity but different layout). base is
// the wavelength constant (paper: 10000; base ≤ 0 → 10000). Because for any fixed offset k the
// vector PE(pos+k) is a fixed linear (rotation) function of PE(pos), the encoding exposes relative
// position. It is the classic ABSOLUTE alternative to this library's relative schemes (RoPE, ALiBi,
// xPos). dModel must be even. A pure-f64 deterministic table (not learned, not differentiable).
func SinusoidalPositionalEncoding(seqLen, dModel int, base float64, dtype tensor.Dtype) (*tensor.Tensor, error) {
	if seqLen <= 0 || dModel <= 0 {
		return nil, fmt.Errorf("nn: SinusoidalPositionalEncoding needs positive seqLen/dModel, got %d/%d", seqLen, dModel)
	}
	if dModel%2 != 0 {
		return nil, fmt.Errorf("nn: SinusoidalPositionalEncoding needs an even dModel, got %d", dModel)
	}
	if base <= 0 {
		base = 10000
	}
	pe := tensor.New(dtype, tensor.Shape{seqLen, dModel})
	// The per-frequency wavelength 1/base^(2i/dModel) depends only on i, not pos, so precompute
	// it once instead of re-evaluating math.Pow inside the position loop (seqLen× redundant).
	half := dModel / 2
	freqs := make([]float64, half)
	for i := range half {
		freqs[i] = 1.0 / math.Pow(base, float64(2*i)/float64(dModel))
	}
	for pos := range seqLen {
		fp := float64(pos)
		for i := range half {
			angle := fp * freqs[i]
			sin, cos := math.Sincos(angle) // one argument reduction feeds both (bit-identical to Sin+Cos)
			pe.SetF64(sin, pos, 2*i)
			pe.SetF64(cos, pos, 2*i+1)
		}
	}
	return pe, nil
}

// SinusoidalPositionalEncodingConcat is the CONCATENATED layout of the same sinusoidal encoding
// (Vaswani et al. 2017 §3.5), used by tensor2tensor, Fairseq and several HuggingFace models: the
// first half of the feature dimension holds all the sines and the second half all the cosines,
//
//	PE(pos, i)          = sin( pos / base^(2i/dModel) )   for i in [0, dModel/2)
//	PE(pos, dModel/2+i) = cos( pos / base^(2i/dModel) )
//
// rather than interleaving sin/cos per pair (SinusoidalPositionalEncoding). It uses the identical
// frequencies, so it is exactly a CHANNEL PERMUTATION of the interleaved table
// (concat[:,i]=interleaved[:,2i], concat[:,dModel/2+i]=interleaved[:,2i+1]) — the same
// representational capacity, only a different feature order (as §R230 established). base ≤ 0 →
// 10000; dModel must be even. A pure-f64 deterministic table.
func SinusoidalPositionalEncodingConcat(seqLen, dModel int, base float64, dtype tensor.Dtype) (*tensor.Tensor, error) {
	if seqLen <= 0 || dModel <= 0 {
		return nil, fmt.Errorf("nn: SinusoidalPositionalEncodingConcat needs positive seqLen/dModel, got %d/%d", seqLen, dModel)
	}
	if dModel%2 != 0 {
		return nil, fmt.Errorf("nn: SinusoidalPositionalEncodingConcat needs an even dModel, got %d", dModel)
	}
	if base <= 0 {
		base = 10000
	}
	half := dModel / 2
	pe := tensor.New(dtype, tensor.Shape{seqLen, dModel})
	// Loop-invariant wavelength precompute (see SinusoidalPositionalEncoding above).
	freqs := make([]float64, half)
	for i := range half {
		freqs[i] = 1.0 / math.Pow(base, float64(2*i)/float64(dModel))
	}
	for pos := range seqLen {
		fp := float64(pos)
		for i := range half {
			angle := fp * freqs[i]
			sin, cos := math.Sincos(angle) // one argument reduction feeds both (bit-identical)
			pe.SetF64(sin, pos, i)         // first half: sines
			pe.SetF64(cos, pos, half+i)    // second half: cosines
		}
	}
	return pe, nil
}
