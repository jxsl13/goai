package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// LLMInt8MatMul computes an approximate matmul Y = X·W with the LLM.int8() scheme
// (Dettmers, Lewis, Belkada & Zettlemoyer 2022, "LLM.int8(): 8-bit Matrix
// Multiplication for Transformers at Scale", NeurIPS, arXiv:2208.07339; the method in
// bitsandbytes). Large transformers develop a few systematic OUTLIER feature
// dimensions whose huge magnitudes would wreck a naive int8 quantization (the outlier
// dominates the absmax and crushes every other value to a handful of levels). LLM.int8()
// solves this with a MIXED-PRECISION decomposition of the contraction dimension:
//
//	Y = X[:,O]·W[O,:]  (high precision)  +  dequant( X̂[:,R]·Ŵ[R,:] )  (vector-wise int8)
//
// where O are the outlier dimensions (any |X[:,h]| ≥ threshold, the paper's α=6.0) and
// R the rest. The int8 path uses VECTOR-WISE quantization — each row of X and each
// column of W gets its own absmax int8 scale — and dequantizes by the outer product of
// scales, s_x[i]·s_w[j]/127². The high-precision (here f64) outlier path keeps the few
// dimensions that matter exact.
//
// X is [tokens, C_in], W is [C_in, C_out]. Returns the approximate [tokens, C_out]
// result and the outlier dimension indices. A pure-f64 inference utility (like
// AWQuantize/GPTQ/SmoothQuant), not differentiable.
func LLMInt8MatMul(x, w *tensor.Tensor, threshold float64) (*tensor.Tensor, []int, error) {
	if x.Ndim() != 2 || w.Ndim() != 2 {
		return nil, nil, fmt.Errorf("nn: LLMInt8MatMul wants rank-2 X[t,C], W[C,out], got %v, %v", x.Shape(), w.Shape())
	}
	tokens, cin := x.Shape()[0], x.Shape()[1]
	if w.Shape()[0] != cin {
		return nil, nil, fmt.Errorf("nn: LLMInt8MatMul contraction mismatch: X has %d, W has %d", cin, w.Shape()[0])
	}
	cout := w.Shape()[1]

	// (1) find outlier feature dimensions: any column of X with |value| ≥ threshold.
	outlier := make([]bool, cin)
	var outIdx []int
	for c := range cin {
		var mx float64
		for t := range tokens {
			if a := math.Abs(x.AtF64(t, c)); a > mx {
				mx = a
			}
		}
		if mx >= threshold {
			outlier[c] = true
			outIdx = append(outIdx, c)
		}
	}

	y := tensor.New(x.Dtype(), tensor.Shape{tokens, cout})

	// (2a) high-precision path over the outlier dimensions: Y += X[:,O]·W[O,:].
	if len(outIdx) > 0 {
		for i := range tokens {
			for j := range cout {
				var acc float64
				for _, c := range outIdx {
					acc += x.AtF64(i, c) * w.AtF64(c, j)
				}
				y.SetF64(y.AtF64(i, j)+acc, i, j)
			}
		}
	}

	// (2b) vector-wise int8 path over the regular dimensions.
	// per-row X absmax (over regular cols) and per-column W absmax (over regular rows).
	sx := make([]float64, tokens)
	for i := range tokens {
		for c := range cin {
			if !outlier[c] {
				if a := math.Abs(x.AtF64(i, c)); a > sx[i] {
					sx[i] = a
				}
			}
		}
	}
	sw := make([]float64, cout)
	for j := range cout {
		for c := range cin {
			if !outlier[c] {
				if a := math.Abs(w.AtF64(c, j)); a > sw[j] {
					sw[j] = a
				}
			}
		}
	}
	q := func(val, scale float64) float64 { // symmetric int8 round
		if scale == 0 {
			return 0
		}
		return math.Round(127 * val / scale)
	}
	// Pre-quantize ONCE: q(x[i,c],sx[i]) and q(w[c,j],sw[j]) depend on only two of the three
	// loop indices, yet the old triple loop recomputed q(x) for every j and q(w) for every i —
	// O(tokens·cout·cin) rounds+divides for O(tokens·cin + cin·cout) distinct values. Fill qx
	// [tokens,cin] and qw TRANSPOSED to [cout,cin] (so the inner dot reads both operands
	// contiguously), leaving outlier columns zero so acc += 0 reproduces the old `continue`
	// bit-for-bit. The inner GEMM is then a plain int8·int8 dot in ascending-c order (Go keeps
	// the scalar float reduction order, so it is bit-identical).
	qx := make([]float64, tokens*cin)
	for i := range tokens {
		base := i * cin
		for c := range cin {
			if !outlier[c] {
				qx[base+c] = q(x.AtF64(i, c), sx[i])
			}
		}
	}
	qwt := make([]float64, cout*cin)
	for c := range cin {
		if outlier[c] {
			continue // leave column c of qwt zero across all j → contributes 0 to every acc
		}
		for j := range cout {
			qwt[j*cin+c] = q(w.AtF64(c, j), sw[j])
		}
	}
	for i := range tokens {
		qxi := qx[i*cin : i*cin+cin : i*cin+cin]
		for j := range cout {
			qwtj := qwt[j*cin : j*cin+cin : j*cin+cin]
			var acc float64 // int32 accumulator (integer-valued)
			for c := range cin {
				acc += qxi[c] * qwtj[c]
			}
			deq := acc * sx[i] * sw[j] / (127 * 127) // outer-product dequant
			y.SetF64(y.AtF64(i, j)+deq, i, j)
		}
	}
	return y, outIdx, nil
}
