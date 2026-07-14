package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// Optimized elementwise kernels (§T11). Binary ops use the internal/simd
// primitives on contiguous typed slices; unary ops use tight typed loops. Both
// parallelize above parThreshold. Non-contiguous inputs are materialized to
// contiguous first (still far cheaper than the reference's per-element Unravel).
// Results are bit-identical to backend/ref (§V3, §V11 tol 0) — same f64 math.

// broadcastContig materializes t broadcast to outShape as a fresh contiguous
// tensor (numpy rules); returns t unchanged if it already has outShape. Used only
// on the broadcasting path — the same-shape SIMD path never calls it.
func broadcastContig(t *tensor.Tensor, outShape tensor.Shape) *tensor.Tensor {
	if t.Shape().Equal(outShape) {
		return t
	}
	tc := t.Contiguous() // dense input (fast §base-perf Contiguous)
	offset := len(outShape) - tc.Ndim()
	out := tensor.New(tc.Dtype(), outShape)
	n := out.Numel()
	if n == 0 {
		return out
	}
	ndo := len(outShape)
	tsh := tc.Shape()
	// Input's strides expressed in the OUTPUT coordinate space (§base-perf): 0 where
	// the input is absent (leading dims) or size-1 (broadcast), else its row-major
	// stride. Walking output coords with these keeps the source flat offset
	// incrementally — no per-element Unravel alloc / BroadcastCoords / AtF64-SetF64.
	tst := tensor.RowMajorStrides(tsh)
	bstride := make([]int, ndo)
	for d := 0; d < ndo; d++ {
		if id := d - offset; id >= 0 && tsh[id] != 1 {
			bstride[d] = tst[id]
		}
	}
	idx := make([]int, ndo)
	switch tc.Dtype() {
	case tensor.F32:
		src, dst := tc.Storage().F32(), out.Storage().F32()
		off := 0
		for pos := 0; pos < n; pos++ {
			dst[pos] = src[off]
			for d := ndo - 1; d >= 0; d-- {
				idx[d]++
				off += bstride[d]
				if idx[d] < outShape[d] {
					break
				}
				idx[d] = 0
				off -= bstride[d] * outShape[d]
			}
		}
	case tensor.F64:
		src, dst := tc.Storage().F64(), out.Storage().F64()
		off := 0
		for pos := 0; pos < n; pos++ {
			dst[pos] = src[off]
			for d := ndo - 1; d >= 0; d-- {
				idx[d]++
				off += bstride[d]
				if idx[d] < outShape[d] {
					break
				}
				idx[d] = 0
				off -= bstride[d] * outShape[d]
			}
		}
	default: // int and other dtypes: the generic coord path
		ic := make([]int, tc.Ndim())
		for pos := range n {
			oc := tensor.Unravel(pos, outShape)
			backend.BroadcastCoords(ic, oc, tsh, offset)
			out.SetF64(tc.AtF64(ic...), oc...)
		}
	}
	return out
}

// binOp builds a binary kernel from the per-dtype simd primitives.
func binOp(f64 func(dst, a, b []float64), f32 func(dst, a, b []float32)) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) != 2 {
			return nil, fmt.Errorf("cpu: binary op wants 2 inputs, got %d", len(in))
		}
		a, b := in[0], in[1]
		if a.Dtype() != b.Dtype() {
			return nil, fmt.Errorf("cpu: binary dtype mismatch %v vs %v", a.Dtype(), b.Dtype())
		}
		if !a.Shape().Equal(b.Shape()) {
			// broadcasting: materialize both operands to the common shape, then run
			// the same SIMD path (the fast same-shape path below is unchanged).
			outShape, err := backend.BroadcastShape(a.Shape(), b.Shape())
			if err != nil {
				return nil, err
			}
			a, b = broadcastContig(a, outShape), broadcastContig(b, outShape)
		}
		ac, bc := a.Contiguous(), b.Contiguous()
		out := tensor.NewOn(ctx.Device(), a.Dtype(), a.Shape())
		switch a.Dtype() {
		case tensor.F64:
			da, db, do := ac.Storage().F64(), bc.Storage().F64(), out.Storage().F64()
			parallel(len(do), func(lo, hi int) { f64(do[lo:hi], da[lo:hi], db[lo:hi]) })
		case tensor.F32:
			da, db, do := ac.Storage().F32(), bc.Storage().F32(), out.Storage().F32()
			parallel(len(do), func(lo, hi int) { f32(do[lo:hi], da[lo:hi], db[lo:hi]) })
		default:
			return nil, fmt.Errorf("cpu: unsupported dtype %v", a.Dtype())
		}
		return []*tensor.Tensor{out}, nil
	}
}

// unOp builds a unary kernel applying scalar f (computed in f64) over contiguous
// data, narrowing for F32.
func unOp(f func(float64) float64) backend.Kernel {
	return func(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
		if len(in) != 1 {
			return nil, fmt.Errorf("cpu: unary op wants 1 input, got %d", len(in))
		}
		xc := in[0].Contiguous()
		out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
		switch in[0].Dtype() {
		case tensor.F64:
			d, o := xc.Storage().F64(), out.Storage().F64()
			parallel(len(o), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					o[i] = f(d[i])
				}
			})
		case tensor.F32:
			d, o := xc.Storage().F32(), out.Storage().F32()
			parallel(len(o), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					o[i] = float32(f(float64(d[i])))
				}
			})
		default:
			return nil, fmt.Errorf("cpu: unsupported dtype %v", in[0].Dtype())
		}
		return []*tensor.Tensor{out}, nil
	}
}

func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}
func gelu(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	z := math.Exp(x)
	return z / (1 + z)
}

// reluKernel / geluKernel / siluKernel: devirtualized unary hot paths (§base-perf) —
// concrete []T loops so the F32 path avoids the unOp closure's per-element indirect call
// AND the float32↔float64 round-trip. relu/neg are fully native; gelu/silu keep f64 math
// (erf/exp are f64-only) but inline it (no func-value indirection). Parity unchanged.
func geluKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	const s = math.Sqrt2
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				x := d[i]
				o[i] = 0.5 * x * (1 + math.Erf(x/s))
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				x := float64(d[i])
				o[i] = float32(0.5 * x * (1 + math.Erf(x/s)))
			}
		})
	default:
		return nil, fmt.Errorf("cpu: gelu unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}
func negKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				o[i] = -d[i]
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				o[i] = -d[i]
			}
		})
	default:
		return nil, fmt.Errorf("cpu: neg unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// stopGradKernelCPU is the identity forward (detach); the VJP zeros the grad. A
// direct copy beats the old per-element unOp closure (memmove vs indirect calls).
func stopGradKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		copy(out.Storage().F64(), xc.Storage().F64())
	case tensor.F32:
		copy(out.Storage().F32(), xc.Storage().F32())
	default:
		return nil, fmt.Errorf("cpu: stopgradient unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func reluKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				if d[i] > 0 {
					o[i] = d[i]
				}
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				if d[i] > 0 {
					o[i] = d[i]
				}
			}
		})
	default:
		return nil, fmt.Errorf("cpu: relu unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}
func siluKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				x := d[i]
				o[i] = x / (1 + math.Exp(-x))
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				x := float64(d[i])
				o[i] = float32(x / (1 + math.Exp(-x)))
			}
		})
	default:
		return nil, fmt.Errorf("cpu: silu unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpAdd, binOp(simd.AddF64, simd.AddF32))
	reg(backend.OpSub, binOp(simd.SubF64, simd.SubF32))
	reg(backend.OpMul, binOp(simd.MulF64, simd.MulF32))
	reg(backend.OpDiv, binOp(simd.DivF64, simd.DivF32))

	reg(backend.OpStopGradient, stopGradKernelCPU)
	reg(backend.OpNeg, negKernelCPU)
	reg(backend.OpExp, unOp(math.Exp))
	reg(backend.OpLog, unOp(math.Log))
	reg(backend.OpTanh, unOp(math.Tanh))
	reg(backend.OpReLU, reluKernelCPU)
	reg(backend.OpGELU, geluKernelCPU)
	reg(backend.OpSigmoid, unOp(sigmoid))
	reg(backend.OpSiLU, siluKernelCPU)
}
