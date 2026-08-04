package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// conv1dKernelCPU is the parallel-over-time depthwise causal conv1d (Mamba/Jamba
// mixer). Each output row t writes a disjoint out[t*D:] and accumulates its K taps
// in ascending-k order — identical to the serial ref kernel (byte-for-byte: f64
// accumulator, single f32 store-rounding) — so parallelizing the independent t-rows
// over the worker pool is bit-exact. F32/F64 only; exotic dtypes have no cpu kernel
// and fall back to the ref kernel.
func conv1dKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) < 2 || len(in) > 3 {
		return nil, fmt.Errorf("cpu: conv1d wants (x, w[, bias]), got %d inputs", len(in))
	}
	x, w := in[0], in[1]
	var bias *tensor.Tensor
	if len(in) == 3 {
		bias = in[2]
	}
	if x.Ndim() != 2 || w.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: conv1d needs x[L,D] and w[D,K]")
	}
	L, D := x.Shape()[0], x.Shape()[1]
	K := w.Shape()[1]
	if w.Shape()[0] != D {
		return nil, fmt.Errorf("cpu: conv1d w channels %d != D %d", w.Shape()[0], D)
	}
	if bias != nil && (bias.Ndim() != 1 || bias.Shape()[0] != D) {
		return nil, fmt.Errorf("cpu: conv1d bias must be [%d], got %v", D, bias.Shape())
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())

	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()
		ws := w.Contiguous().Storage().F64()
		os := out.Storage().F64()
		var bs []float64
		if bias != nil {
			bs = bias.Contiguous().Storage().F64()
		}
		parallelWork(L, D*K, func(loT, hiT int) {
			for t := loT; t < hiT; t++ {
				kStart := 0
				if lo := (K - 1) - t; lo > 0 {
					kStart = lo
				}
				for c := 0; c < D; c++ {
					var acc float64
					//perfscan:ignore PS3010,PS4008 short K-tap inner loop, already parallel typed fastpath | conv width K~4 tiny, strided x access, not matmul do
					for k := kStart; k < K; k++ {
						acc += ws[c*K+k] * xs[(t-(K-1)+k)*D+c]
					}
					if bias != nil {
						acc += bs[c]
					}
					os[t*D+c] = acc
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()
		ws := w.Contiguous().Storage().F32()
		os := out.Storage().F32()
		var bs []float32
		if bias != nil {
			bs = bias.Contiguous().Storage().F32()
		}
		parallelWork(L, D*K, func(loT, hiT int) {
			for t := loT; t < hiT; t++ {
				kStart := 0
				if lo := (K - 1) - t; lo > 0 {
					kStart = lo
				}
				for c := 0; c < D; c++ {
					var acc float64
					//perfscan:ignore PS3010 F32 short-K loop, already optimized parallel fastpath
					for k := kStart; k < K; k++ {
						acc += float64(ws[c*K+k]) * float64(xs[(t-(K-1)+k)*D+c])
					}
					if bias != nil {
						acc += float64(bs[c])
					}
					os[t*D+c] = float32(acc)
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	}
	return nil, fmt.Errorf("cpu: conv1d unsupported dtype %v", x.Dtype())
}

func init() {
	std.add(backend.OpConv1D, tensor.F32, conv1dKernelCPU)
	std.add(backend.OpConv1D, tensor.F64, conv1dKernelCPU)
}
