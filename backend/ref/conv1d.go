package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// softplusKernel is the elementwise softplus log(1+eˣ), computed stably as
// max(x,0)+log(1+e^(−|x|)) so large x never overflows (§V12). Used to keep
// Mamba's Δ strictly positive (§R76). Reuses the softplus helper from dpo.go.
func softplusKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: softplus wants 1 input, got %d", len(in))
	}
	x := in[0]
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())
	n := x.Numel()

	// Devirtualised fast paths (§T646): a hot cpu→ref fallback on the CPU training
	// path. The generic AtF64/SetF64 loop pays a dtype dispatch + Unravel/flat-offset
	// per element; here we grab the raw typed slices once and index flat (softplus is
	// elementwise, so flat order over Numel matches the generic ravel order exactly).
	// The scalar formula is identical; the F32 path reads as float64, computes in
	// float64 and rounds the STORED result once. Contiguous() runs once (returns self
	// when already contiguous).
	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()
		os := out.Storage().F64()
		for i := range n {
			os[i] = softplus(xs[i])
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()
		os := out.Storage().F32()
		for i := range n {
			os[i] = float32(softplus(float64(xs[i])))
		}
		return []*tensor.Tensor{out}, nil
	}

	// Generic fallback for exotic dtypes (verbatim original loop).
	for i := range n {
		idx := tensor.Unravel(i, x.Shape())
		out.SetF64(softplus(x.AtF64(idx...)), idx...)
	}
	return []*tensor.Tensor{out}, nil
}

// conv1dKernel is a causal depthwise 1-D convolution (Mamba's mixing conv, §R77;
// also WaveNet/TCN). Each channel c has its own length-K filter; the sequence is
// left-padded with K−1 zeros so output t depends only on inputs ≤ t (causal):
//
//	out[t,c] = Σ_{k=0}^{K-1} w[c,k]·x[t−(K−1)+k, c] + b[c]     (x[j]=0 for j<0)
//
// Inputs: x[L,D], w[D,K], and an OPTIONAL bias b[D]. Output [L,D]. The last tap
// (k=K−1) is the current position x[t]; earlier taps look into the past.
func conv1dKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) < 2 || len(in) > 3 {
		return nil, fmt.Errorf("ref: conv1d wants (x, w[, bias]), got %d inputs", len(in))
	}
	x, w := in[0], in[1]
	var bias *tensor.Tensor
	if len(in) == 3 {
		bias = in[2]
	}
	if x.Ndim() != 2 || w.Ndim() != 2 {
		return nil, fmt.Errorf("ref: conv1d needs x[L,D] and w[D,K]")
	}
	L, D := x.Shape()[0], x.Shape()[1]
	K := w.Shape()[1]
	if w.Shape()[0] != D {
		return nil, fmt.Errorf("ref: conv1d w channels %d != D %d", w.Shape()[0], D)
	}
	if bias != nil && (bias.Ndim() != 1 || bias.Shape()[0] != D) {
		return nil, fmt.Errorf("ref: conv1d bias must be [%d], got %v", D, bias.Shape())
	}

	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())

	// Devirtualised fast paths (§T646): a hot cpu→ref fallback on the CPU training
	// path. The generic AtF64/SetF64 nest pays a dtype dispatch + flat-offset per
	// tap; here we grab the raw typed slices once (row-major: x[j,c] at j*D+c,
	// w[c,k] at c*K+k, out[t,c] at t*D+c, bias[c] at c) and index directly. The loop
	// nesting, index/stride/pad arithmetic, boundary guard and accumulation order are
	// copied verbatim; the accumulator stays float64 on ALL paths, and the F32 path
	// only rounds the single STORED sum — so results are byte-for-byte identical to
	// the generic path. Contiguous() runs once per tensor (returns self when already
	// contiguous).
	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()
		ws := w.Contiguous().Storage().F64()
		os := out.Storage().F64()
		var bs []float64
		if bias != nil {
			bs = bias.Contiguous().Storage().F64()
		}
		for t := range L {
			// j = t-(K-1)+k >= 0  <=>  k >= (K-1)-t; j is always < L (j <= t). Hoist the
			// per-t lower tap bound so the innermost dot drops the per-tap branch.
			kStart := 0
			if lo := (K - 1) - t; lo > 0 {
				kStart = lo
			}
			for c := range D {
				var acc float64
				for k := kStart; k < K; k++ {
					acc += ws[c*K+k] * xs[(t-(K-1)+k)*D+c]
				}
				if bias != nil {
					acc += bs[c]
				}
				os[t*D+c] = acc
			}
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()
		ws := w.Contiguous().Storage().F32()
		os := out.Storage().F32()
		var bs []float32
		if bias != nil {
			bs = bias.Contiguous().Storage().F32()
		}
		for t := range L {
			kStart := 0
			if lo := (K - 1) - t; lo > 0 {
				kStart = lo
			}
			for c := range D {
				var acc float64 // accumulate in float64; only the store rounds
				for k := kStart; k < K; k++ {
					acc += float64(ws[c*K+k]) * float64(xs[(t-(K-1)+k)*D+c])
				}
				if bias != nil {
					acc += float64(bs[c])
				}
				os[t*D+c] = float32(acc)
			}
		}
		return []*tensor.Tensor{out}, nil
	}

	// Generic fallback for exotic dtypes (verbatim original loop).
	for t := range L {
		kStart := 0
		if lo := (K - 1) - t; lo > 0 {
			kStart = lo
		}
		for c := range D {
			var acc float64
			for k := kStart; k < K; k++ {
				acc += w.AtF64(c, k) * x.AtF64(t-(K-1)+k, c)
			}
			if bias != nil {
				acc += bias.AtF64(c)
			}
			out.SetF64(acc, t, c)
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpSoftplus, tensor.F32, softplusKernel)
	std.add(backend.OpSoftplus, tensor.F64, softplusKernel)
	std.add(backend.OpConv1D, tensor.F32, conv1dKernel)
	std.add(backend.OpConv1D, tensor.F64, conv1dKernel)
}
