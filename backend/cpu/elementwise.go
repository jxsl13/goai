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
//
//perfscan:ignore PS6004 test-evidence finding not perf; fix is a test not code
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
		broadcastRuns(out.Storage().F32(), tc.Storage().F32(), bstride, outShape, idx, n)
	case tensor.F64:
		broadcastRuns(out.Storage().F64(), tc.Storage().F64(), bstride, outShape, idx, n)
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

// suffixPeriod reports whether t (right-aligned against outShape, numpy rules)
// broadcasts as a pure suffix tile: flat output index i reads t at i%period.
// True when t's dims form a contiguous trailing run matching outShape, with
// every dim left of that run equal to 1. Covers the two dominant broadcast
// shapes in model code — bias vectors [..,N]⊕[N] and scalars [..]⊕[1] — which
// then skip the materialize-both-operands slow path entirely.
func suffixPeriod(tsh, outShape tensor.Shape) (int, bool) {
	off := len(outShape) - len(tsh)
	period := 1
	i := len(outShape) - 1
	for ; i >= off; i-- {
		if tsh[i-off] != outShape[i] {
			break // run ends; everything left must be size 1
		}
		period *= tsh[i-off]
	}
	for ; i >= off; i-- {
		if tsh[i-off] != 1 {
			return 0, false
		}
	}
	return period, true
}

// prefixBlock reports whether t broadcasts as constant blocks: flat output
// index i reads t at i/block. True when t has outShape's rank, its leading dims
// match outShape, and every dim after the first mismatch is 1. Covers column
// broadcasts like [M,N]⊘[M,1] (row normalization). Mutually exclusive with the
// shapes-equal case (block would be 1), and checked after suffixPeriod, which
// owns the scalar/all-ones overlap.
func prefixBlock(tsh, outShape tensor.Shape) (int, bool) {
	if len(tsh) != len(outShape) {
		return 0, false
	}
	i := 0
	for ; i < len(tsh); i++ {
		if tsh[i] != outShape[i] {
			break
		}
	}
	block := 1
	for ; i < len(tsh); i++ {
		if tsh[i] != 1 {
			return 0, false
		}
		block *= outShape[i]
	}
	return block, true
}

// bcastTile: minimum slice length fed to a simd primitive on the broadcast fast
// path. Tiny periods (scalar, narrow vectors) are pre-tiled to this length so
// the per-call dispatch and vector-tail costs amortize over ≥bcastTile lanes.
//
//perfscan:ignore PS6023 missing-test-evidence, not a defect; fix is a test
const bcastTile = 4096

// bcastSuffixApply computes out = f(full, small-tiled) (or f(small-tiled, full)
// when smallIsA) where small repeats every period elements. Identical
// element-pair operations to the materialized path — bit-identical results
// (§V3) — but without writing an n-sized broadcast copy first.
func bcastSuffixApply[T normFloat](f func(dst, a, b []T), out, full, small []T, period int, smallIsA bool) {
	n := len(out)
	if n == 0 {
		return
	}
	s := small
	if period < bcastTile && n > period {
		reps := min((bcastTile+period-1)/period, n/period)
		tile := make([]T, reps*period)
		for i := 0; i < len(tile); i += period {
			copy(tile[i:], small)
		}
		s = tile
	}
	sp := len(s) // chunk stride; always a multiple of period, so s realigns
	rows := (n + sp - 1) / sp
	parallelWork(rows, sp, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			start := r * sp
			end := min(start+sp, n)
			sv := s[:end-start]
			if smallIsA {
				f(out[start:end], sv, full[start:end])
			} else {
				f(out[start:end], full[start:end], sv)
			}
		}
	})
}

// bcastBlockApply computes out = f(full, small-blocks) (or the swap when
// smallIsA) where small[k] is constant over output block k of length block.
// Each worker splats the block's scalar into a small cached tile and feeds the
// simd primitive — identical element pairs to the materialized path
// (bit-identical results, §V3) without allocating or streaming an n-sized
// broadcast copy.
func bcastBlockApply[T normFloat](f func(dst, a, b []T), out, full, small []T, block int, smallIsA bool) {
	n := len(out)
	if n == 0 {
		return
	}
	nb := n / block
	parallelWork(nb, block, func(lo, hi int) {
		//perfscan:ignore PS6008 per-worker tile bounded by GOMAXPROCS, one dispatch per op; sanctioned
		tile := make([]T, min(block, bcastTile))
		for k := lo; k < hi; k++ {
			// Refill unconditionally: a value-equality skip would treat -0.0 and
			// +0.0 as the same tile and flip Mul/Div result signs.
			v := small[k]
			for i := range tile {
				tile[i] = v
			}
			start := k * block
			for off := 0; off < block; off += len(tile) {
				end := min(off+len(tile), block)
				sv := tile[:end-off]
				if smallIsA {
					f(out[start+off:start+end], sv, full[start+off:start+end])
				} else {
					f(out[start+off:start+end], full[start+off:start+end], sv)
				}
			}
		}
	})
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
			outShape, err := backend.BroadcastShape(a.Shape(), b.Shape())
			if err != nil {
				return nil, err
			}
			// Fast path: one operand already has the output shape and the other
			// tiles it as a contiguous suffix (bias vector / scalar) — apply f
			// per tile, skipping the O(n) broadcast materialization below.
			var full, small *tensor.Tensor
			smallIsA := false
			if a.Shape().Equal(outShape) {
				full, small = a, b
			} else if b.Shape().Equal(outShape) {
				full, small, smallIsA = b, a, true
			}
			if full != nil {
				if period, ok := suffixPeriod(small.Shape(), outShape); ok {
					fc, sc := full.Contiguous(), small.Contiguous()
					out := tensor.NewOn(ctx.Device(), a.Dtype(), outShape)
					switch a.Dtype() {
					case tensor.F64:
						bcastSuffixApply(f64, out.Storage().F64(), fc.Storage().F64(), sc.Storage().F64(), period, smallIsA)
					case tensor.F32:
						bcastSuffixApply(f32, out.Storage().F32(), fc.Storage().F32(), sc.Storage().F32(), period, smallIsA)
					default:
						return nil, fmt.Errorf("cpu: unsupported dtype %v", a.Dtype())
					}
					return []*tensor.Tensor{out}, nil
				}
				if block, ok := prefixBlock(small.Shape(), outShape); ok {
					fc, sc := full.Contiguous(), small.Contiguous()
					out := tensor.NewOn(ctx.Device(), a.Dtype(), outShape)
					switch a.Dtype() {
					case tensor.F64:
						bcastBlockApply(f64, out.Storage().F64(), fc.Storage().F64(), sc.Storage().F64(), block, smallIsA)
					case tensor.F32:
						bcastBlockApply(f32, out.Storage().F32(), fc.Storage().F32(), sc.Storage().F32(), block, smallIsA)
					default:
						return nil, fmt.Errorf("cpu: unsupported dtype %v", a.Dtype())
					}
					return []*tensor.Tensor{out}, nil
				}
			}
			// General broadcasting: materialize both operands to the common
			// shape, then run the same SIMD path.
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

// reluKernel / geluKernel / siluKernel / expKernel / logKernel / tanhKernel /
// sigmoidKernel: devirtualized unary hot paths (§base-perf) — concrete []T loops
// so the F32 path avoids the unOp closure's per-element indirect call AND the
// float32↔float64 round-trip. relu/neg are fully native; the transcendentals
// keep f64 math (erf/exp/log/tanh are f64-only) but inline it (no func-value
// indirection). Parity unchanged — same f64 ops per element as ref (§V3 tol 0).
func geluKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	const s = math.Sqrt2
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		if vexpF64Fast {
			// SIMD perf build: f64-native vectorized GELU — the Cephes erf (erfF64x4,
			// built on expF64x4; vexp_amd64.go) replacing scalar math.Erf, the last
			// transcendental activation still scalar in F64 (SiLU/Sigmoid/Tanh/Softplus
			// F64 already vectorized). ~1 ulp, rides the model f64 tolerance. The default
			// build keeps the scalar math.Erf path below, bit-for-bit vs ref.
			parallel(len(o), func(lo, hi int) { vgeluF64(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F64 gelu erf ref-exact-locked (CPU==Ref tol0 on the default build)
			for i := lo; i < hi; i++ {
				x := d[i]
				o[i] = 0.5 * x * (1 + math.Erf(x/s))
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		if vexpF32Fast {
			// SIMD perf build: f32-native vectorized GELU — AS-7.1.26 erf assembled
			// on the vexp exp primitive (4-wide NEON on arm64, 8-wide AVX2 on amd64;
			// vexp.go / vexp_amd64.go). Scalar math.Erf was 13.6% of the f32 GPT
			// forward. Compile-time const: the plain (no-simd) build keeps the f64
			// path below bit-for-bit; this path rides the ADR-0021 f32 tolerance.
			parallel(len(o), func(lo, hi int) { vgeluF32(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F32 no-simd fallback; vgeluF32 SIMD sibling already registered
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

// geluBackwardKernelCPU is the arm64 perf-build F32 GELU VJP (§T664): dx =
// g·(Φ(x) + x·φ(x)) evaluated f32-native through the NEON vgeluGrad pipeline
// (vexp.go / vexp_arm64.s) — the same AS-7.1.26 erf + shared e^(−x²/2) the
// forward vgelu path uses, so ONE exp feeds both terms. gelu_backward was
// 18.9% of the f32 GPT training step as the scalar-f64 ref fallback (§V22
// op-profile). Formula identical to ref's geluGrad (§T353); only the
// evaluation is vectorized, riding the ADR-0021 f32 tolerant-parity budget
// (TestGeluGradF32Accuracy). Registered ONLY when vexpNeon and ONLY for F32 —
// the default build and amd64 keep the untouched ref fallback bit-for-bit,
// and so does F64 on this build. A mixed-dtype or mismatched-shape input pair
// is handed to the reference kernel unchanged.
func geluBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 && in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 &&
		in[1].Shape().Equal(in[0].Shape()) {
		xc, gc := in[0].Contiguous(), in[1].Contiguous()
		dx := tensor.NewOn(ctx.Device(), tensor.F32, in[0].Shape())
		xs, gs, ds := xc.Storage().F32(), gc.Storage().F32(), dx.Storage().F32()
		parallel(len(ds), func(lo, hi int) { vgeluGradF32(ds[lo:hi], xs[lo:hi], gs[lo:hi]) })
		return []*tensor.Tensor{dx}, nil
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpGELUBackward, in, attrs)
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
func expKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F64 exp ref-exact-locked (tol0 invariant)
			for i := lo; i < hi; i++ {
				o[i] = math.Exp(d[i])
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		if vexpF32Fast {
			// SIMD perf build (§T666): f32-native vectorized full-domain exp — the
			// vexp reduction with split 2^n scaling and exact overflow/underflow
			// masks (4-wide NEON on arm64, 8-wide AVX2 on amd64; vexp.go /
			// vexp_amd64.go). Compile-time const: the plain (no-simd) build keeps
			// the f64 path below bit-for-bit; rides the ADR-0021 f32 tolerance.
			parallel(len(o), func(lo, hi int) { vexpFullF32(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F32 no-simd fallback; vexpFullF32 sibling exists
			for i := lo; i < hi; i++ {
				o[i] = float32(math.Exp(float64(d[i])))
			}
		})
	default:
		return nil, fmt.Errorf("cpu: exp unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}
func logKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F64 log ref-exact-locked (tol0 invariant)
			for i := lo; i < hi; i++ {
				o[i] = math.Log(d[i])
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		if vexpF32Fast {
			// SIMD perf build (§T666): f32-native vectorized log — the Cephes logf
			// reduction (4-wide NEON on arm64, 8-wide AVX2 on amd64; vexp.go /
			// vexp_amd64.go). Compile-time const: the plain (no-simd) build keeps
			// the f64 path bit-for-bit; rides the ADR-0021 f32 tolerance.
			parallel(len(o), func(lo, hi int) { vlogF32(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F32 no-simd fallback; vlogF32 sibling exists
			for i := lo; i < hi; i++ {
				o[i] = float32(math.Log(float64(d[i])))
			}
		})
	default:
		return nil, fmt.Errorf("cpu: log unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}
func tanhKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		if vexpF64Fast {
			// amd64 SIMD build: 4-wide f64-native tanh on the AVX2 expF64x4 primitive
			// — the stable sign-split (1−e^(−2|x|))/(1+e^(−2|x|)) (the vsoftcapF64
			// lane at cap=1). Not under the CPU==Ref exact invariant (that test skips
			// OpTanh/F64); rides the model f64 tolerance. The non-SIMD build below
			// keeps math.Tanh bit-for-bit.
			parallel(len(o), func(lo, hi int) { vtanhF64(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F64 tanh no-simd fallback; vtanhF64 sibling exists
			for i := lo; i < hi; i++ {
				o[i] = math.Tanh(d[i])
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		if vexpF32Fast {
			// SIMD perf build (§T666): f32-native vectorized tanh — the stable
			// sign-split (1−e^(−2|x|))/(1+e^(−2|x|)) on the vexp exp primitive
			// (4-wide NEON on arm64, 8-wide AVX2 on amd64; vexp.go / vexp_amd64.go).
			// Compile-time const: the plain (no-simd) build keeps the f64 path
			// bit-for-bit; rides the ADR-0021 f32 tolerance.
			parallel(len(o), func(lo, hi int) { vtanhF32(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F32 no-simd fallback; vtanhF32 sibling exists
			for i := lo; i < hi; i++ {
				o[i] = float32(math.Tanh(float64(d[i])))
			}
		})
	default:
		return nil, fmt.Errorf("cpu: tanh unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// sigmoidKernelCPU keeps ref's numerically-stable split form: exp(-x) for x≥0,
// exp(x)/(1+exp(x)) for x<0 — never exponentiates a positive magnitude.
func sigmoidKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		if vexpF64Fast {
			// amd64 SIMD build: 4-wide f64-native sigmoid on the AVX2 expF64x4
			// primitive — the same stable split, vectorized (the vsiluF64 lane
			// without the x multiply). Not under the CPU==Ref exact invariant (that
			// test skips OpSigmoid/F64); rides the model f64 tolerance. The non-SIMD
			// build below keeps the scalar path bit-for-bit.
			parallel(len(o), func(lo, hi int) { vsigmoidF64(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F64 sigmoid no-simd fallback; vsigmoidF64 sibling exists
			for i := lo; i < hi; i++ {
				x := d[i]
				if x >= 0 {
					o[i] = 1 / (1 + math.Exp(-x))
				} else {
					z := math.Exp(x)
					o[i] = z / (1 + z)
				}
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		if vexpF32Fast {
			// SIMD perf build: f32-native vectorized sigmoid — the same stable
			// split on the vexp exp primitive, 4-wide NEON on arm64 / 8-wide AVX2
			// on amd64 (vexp.go / vexp_amd64.go, §T665). Compile-time const: the
			// plain (no-simd) build keeps the f64 path below bit-for-bit; this
			// path rides the ADR-0021 f32 tolerance.
			parallel(len(o), func(lo, hi int) { vsigmoidF32(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F32 no-simd fallback; vsigmoidF32 sibling exists
			for i := lo; i < hi; i++ {
				x := float64(d[i])
				if x >= 0 {
					o[i] = float32(1 / (1 + math.Exp(-x)))
				} else {
					z := math.Exp(x)
					o[i] = float32(z / (1 + z))
				}
			}
		})
	default:
		return nil, fmt.Errorf("cpu: sigmoid unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}
func siluKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		if vexpF64Fast {
			// amd64 SIMD build: 4-wide f64-native SiLU on the AVX2 expF64x4 primitive
			// (§T667) — the SwiGLU FFN gate was scalar f64 math.Exp here (~40% of
			// Llama f64 prefill). Not under the CPU==Ref exact invariant (that test
			// skips OpSiLU/F64); rides the model f64 tolerance. Non-SIMD build below
			// keeps the scalar path bit-for-bit.
			parallel(len(o), func(lo, hi int) { vsiluF64(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F64 silu no-simd fallback; vsiluF64 sibling exists
			for i := lo; i < hi; i++ {
				x := d[i]
				o[i] = x / (1 + math.Exp(-x))
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		if vexpF32Fast {
			// SIMD perf build: f32-native vectorized SiLU on the vexp exp primitive
			// (4-wide NEON on arm64 / 8-wide AVX2 on amd64; vexp.go / vexp_amd64.go,
			// §T665) — the SwiGLU FFN activation on every Llama/Qwen/Mistral layer
			// was scalar f64 math.Exp here. Compile-time const: the plain (no-simd)
			// build keeps the f64 path below bit-for-bit; rides the ADR-0021 tolerance.
			parallel(len(o), func(lo, hi int) { vsiluF32(o[lo:hi], d[lo:hi]) })
			break
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F32 no-simd fallback; vsiluF32 sibling exists
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

// softplusKernelCPU is the F64 CPU softplus (Mamba/Jamba Δ). softplus was scalar
// math.Exp+math.Log1p on the ref backend and dominated Mamba f64 prefill (~32% of
// samples, split ref.softplus + the mixer). The amd64 SIMD build routes F64 through
// the 4-wide vsoftplusF64 (expF64x4 + Cephes double log); F32 stays on the ref
// fallback (unregistered). Not under the CPU==Ref exact invariant (that test omits
// OpSoftplus); rides the model f64 tolerance (Mamba goldens gate at 1e-9). The
// non-SIMD build keeps the scalar formula bit-for-bit.
func softplusKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	switch xc.Dtype() {
	case tensor.F64:
		out := tensor.NewOn(ctx.Device(), tensor.F64, in[0].Shape())
		d, o := xc.Storage().F64(), out.Storage().F64()
		if vexpF64Fast {
			parallel(len(o), func(lo, hi int) { vsoftplusF64(o[lo:hi], d[lo:hi]) })
			return []*tensor.Tensor{out}, nil
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F64 softplus no-simd fallback; vsoftplusF64 sibling exists
			for i := lo; i < hi; i++ {
				x := d[i]
				if x > 0 {
					o[i] = x + math.Log1p(math.Exp(-x))
				} else {
					o[i] = math.Log1p(math.Exp(x))
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		out := tensor.NewOn(ctx.Device(), tensor.F32, in[0].Shape())
		d, o := xc.Storage().F32(), out.Storage().F32()
		if vexpF32Fast {
			// SIMD perf build: f32-native vectorized softplus via vsoftplusF32 (8-wide
			// AVX2 on amd64) instead of the per-element scalar f64 math.Log1p/math.Exp
			// below — the Mamba Δ / griffin/hymba gate / focal & reward loss hot path.
			// Rides the ADR-0021 f32 tolerance (same expF32/logF32 primitive as
			// vtanhF32/vgeluF32); the default build below stays bit-for-bit vs ref.
			parallel(len(o), func(lo, hi int) { vsoftplusF32(o[lo:hi], d[lo:hi]) })
			return []*tensor.Tensor{out}, nil
		}
		// Default build: the stable softplus (same branch as the F64 path and ref's
		// softplus()) evaluated in f64 with a single round on store — bit-identical to
		// ref's F32 path; disjoint outputs.
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F32 no-simd fallback; vsoftplusF32 sibling exists
			for i := lo; i < hi; i++ {
				x := float64(d[i])
				if x > 0 {
					o[i] = float32(x + math.Log1p(math.Exp(-x)))
				} else {
					o[i] = float32(math.Log1p(math.Exp(x)))
				}
			}
		})
		return []*tensor.Tensor{out}, nil
	}
	return nil, fmt.Errorf("cpu: softplus unsupported dtype %v", xc.Dtype())
}

// softCapKernelCPU is the F64 CPU soft-cap (Gemma-2 attn_logit_softcapping /
// final_logit_softcapping): out = cap·tanh(x/cap), applied to every attention
// score on every layer — a hot scalar-tanh op that had no CPU kernel and fell to
// the ref backend. The amd64 SIMD build routes F64 through the 4-wide vsoftcapF64
// (tanh off expF64x4); F32 stays on the ref fallback. Not under the CPU==Ref exact
// invariant; rides the model f64 tolerance. Non-SIMD builds keep the scalar formula
// (bit-for-bit ref) below.
func softCapKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	pa, _ := attrs.(backend.SoftCapAttrs)
	if pa.Cap <= 0 {
		return nil, fmt.Errorf("cpu: softcap cap must be > 0, got %g", pa.Cap)
	}
	xc := in[0].Contiguous()
	switch xc.Dtype() {
	case tensor.F64:
		out := tensor.NewOn(ctx.Device(), tensor.F64, in[0].Shape())
		d, o := xc.Storage().F64(), out.Storage().F64()
		if vexpF64Fast {
			parallel(len(o), func(lo, hi int) { vsoftcapF64(o[lo:hi], d[lo:hi], pa.Cap) })
			return []*tensor.Tensor{out}, nil
		}
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F64 softcap no-simd fallback; vsoftcapF64 sibling exists
			for i := lo; i < hi; i++ {
				o[i] = pa.Cap * math.Tanh(d[i]/pa.Cap)
			}
		})
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		out := tensor.NewOn(ctx.Device(), tensor.F32, in[0].Shape())
		d, o := xc.Storage().F32(), out.Storage().F32()
		if vexpF32Fast {
			// SIMD perf build: f32-native vectorized cap·tanh(x/cap) via vsoftcapF32
			// (8-wide AVX2 on amd64) instead of the per-element scalar f64 math.Tanh
			// below — the Gemma-2 attention-logit and [T,vocab] final-logit soft-cap
			// hot path. Rides the ADR-0021 f32 tolerance (same expF32 primitive as
			// vtanhF32/vgeluF32); the default build below stays bit-for-bit vs ref.
			capf := float32(pa.Cap)
			parallel(len(o), func(lo, hi int) { vsoftcapF32(o[lo:hi], d[lo:hi], capf) })
			return []*tensor.Tensor{out}, nil
		}
		// Default build: cap·tanh(x/cap) in f64 with a single round on store —
		// bit-identical to ref's F32 path (backend/ref/softcap.go); disjoint output
		// elements → race-free.
		parallel(len(o), func(lo, hi int) {
			//perfscan:ignore PS4002 F32 no-simd fallback; vsoftcapF32 sibling exists
			for i := lo; i < hi; i++ {
				o[i] = float32(pa.Cap * math.Tanh(float64(d[i])/pa.Cap))
			}
		})
		return []*tensor.Tensor{out}, nil
	}
	return nil, fmt.Errorf("cpu: softcap unsupported dtype %v", xc.Dtype())
}

// siluBackwardKernelCPU is the arm64 perf-build F32 SiLU VJP (§T665): dx =
// g·silu'(x) evaluated f32-native through the NEON vsiluGrad pipeline
// (vexp.go / vexp_arm64.s) — one exp per element instead of the ref
// fallback's serial scalar f64 sigmoid. Formula identical to ref's
// siluBackwardKernel (§T362); only the evaluation is vectorized, riding the
// ADR-0021 f32 tolerant-parity budget (TestSiluGradF32Accuracy). Registered
// ONLY when vexpNeon and ONLY for F32 — the default build and amd64 keep the
// untouched ref fallback bit-for-bit, and so does F64 on this build. A
// mixed-dtype or mismatched-shape input pair is handed to the reference
// kernel unchanged.
func siluBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 && in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 &&
		in[1].Shape().Equal(in[0].Shape()) {
		xc, gc := in[0].Contiguous(), in[1].Contiguous()
		dx := tensor.NewOn(ctx.Device(), tensor.F32, in[0].Shape())
		xs, gs, ds := xc.Storage().F32(), gc.Storage().F32(), dx.Storage().F32()
		parallel(len(ds), func(lo, hi int) { vsiluGradF32(ds[lo:hi], xs[lo:hi], gs[lo:hi]) })
		return []*tensor.Tensor{dx}, nil
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpSiLUBackward, in, attrs)
}

// softplusBackwardKernelCPU computes dx = g·softplus'(x) = g·σ(x). The F64 case (Mamba/Jamba
// Δ backward) runs the vectorized vsoftplusGradF64 (4-wide AVX2 on amd64, scalar elsewhere)
// over parallel chunks; other dtypes fall back to the reference kernel.
func softplusBackwardKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 2 && in[0].Dtype() == tensor.F64 && in[1].Dtype() == tensor.F64 &&
		in[1].Shape().Equal(in[0].Shape()) {
		xc, gc := in[0].Contiguous(), in[1].Contiguous()
		dx := tensor.NewOn(ctx.Device(), tensor.F64, in[0].Shape())
		xs, gs, ds := xc.Storage().F64(), gc.Storage().F64(), dx.Storage().F64()
		parallel(len(ds), func(lo, hi int) { vsoftplusGradF64(ds[lo:hi], xs[lo:hi], gs[lo:hi]) })
		return []*tensor.Tensor{dx}, nil
	}
	if len(in) == 2 && in[0].Dtype() == tensor.F32 && in[1].Dtype() == tensor.F32 &&
		in[1].Shape().Equal(in[0].Shape()) {
		xc, gc := in[0].Contiguous(), in[1].Contiguous()
		dx := tensor.NewOn(ctx.Device(), tensor.F32, in[0].Shape())
		xs, gs, ds := xc.Storage().F32(), gc.Storage().F32(), dx.Storage().F32()
		parallel(len(ds), func(lo, hi int) { vsoftplusGradF32(ds[lo:hi], xs[lo:hi], gs[lo:hi]) })
		return []*tensor.Tensor{dx}, nil
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpSoftplusBackward, in, attrs)
}

// maxSliceF64/minSliceF64 (and the F32 pair) are the elementwise Max/Min slice bodies for binOp.
// They call math.Max/math.Min directly (a DIRECT call, not the reference binaryKernel's per-element
// func-VALUE indirection) so the compiler emits the same IEEE-correct max/min the ref uses — NaN
// absorbing, Max(+0,−0)=+0, Min(+0,−0)=−0 — bit-for-bit. The F32 bodies compute in float64 and round
// only the store, exactly as the ref F32 path does. binOp fans them across cores; the ref fell through
// single-threaded because the CPU backend registered no Maximum/Minimum kernel.
func maxSliceF64(dst, a, b []float64) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = math.Max(a[i], b[i])
	}
}
func minSliceF64(dst, a, b []float64) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = math.Min(a[i], b[i])
	}
}
func maxSliceF32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = float32(math.Max(float64(a[i]), float64(b[i])))
	}
}
func minSliceF32(dst, a, b []float32) {
	a, b = a[:len(dst)], b[:len(dst)]
	for i := range dst {
		dst[i] = float32(math.Min(float64(a[i]), float64(b[i])))
	}
}

// absKernelCPU / sqrtKernelCPU are the devirtualized+parallel CPU unary kernels for OpAbs/OpSqrt,
// which had no CPU registration and fell to the single-threaded reference unaryKernel — where each
// element pays a non-inlinable func-VALUE call whose payload (a bit-mask abs / a hardware sqrt) is far
// cheaper than the call itself. Here the math is a direct inlined call, fanned across cores. Bit-exact
// with the ref: F32 computes in float64 and rounds only the store (matching float32(f(float64(x)))).
func absKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				o[i] = math.Abs(d[i])
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				o[i] = float32(math.Abs(float64(d[i])))
			}
		})
	default:
		return nil, fmt.Errorf("cpu: abs unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

func sqrtKernelCPU(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	xc := in[0].Contiguous()
	out := tensor.NewOn(ctx.Device(), in[0].Dtype(), in[0].Shape())
	switch in[0].Dtype() {
	case tensor.F64:
		d, o := xc.Storage().F64(), out.Storage().F64()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				o[i] = math.Sqrt(d[i])
			}
		})
	case tensor.F32:
		d, o := xc.Storage().F32(), out.Storage().F32()
		parallel(len(o), func(lo, hi int) {
			for i := lo; i < hi; i++ {
				o[i] = float32(math.Sqrt(float64(d[i])))
			}
		})
	default:
		return nil, fmt.Errorf("cpu: sqrt unsupported dtype %v", in[0].Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// whereKernelCPU is the devirtualized+parallel select for OpWhere (cond?a:b). The reference kernel
// is already devirtualized to typed slices but runs single-threaded; here the same-shape/same-dtype
// F64/F32 fast paths fan the flat select across cores. Bit-exact: the select is a verbatim copy (no
// narrowing), same order. Shape mismatches, Lo>Hi-style errors, and exotic/mixed dtypes delegate to
// the reference kernel so its validation and any-dtype contract are preserved unchanged.
func whereKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) == 3 {
		cond, a, b := in[0], in[1], in[2]
		if a.Dtype() == b.Dtype() && cond.Shape().Equal(a.Shape()) && b.Shape().Equal(a.Shape()) {
			switch a.Dtype() {
			case tensor.F64:
				if cond.Dtype() == tensor.F64 {
					cs, as, bs := cond.Contiguous().Storage().F64(), a.Contiguous().Storage().F64(), b.Contiguous().Storage().F64()
					out := tensor.NewOn(ctx.Device(), tensor.F64, a.Shape())
					os := out.Storage().F64()
					parallel(len(os), func(lo, hi int) {
						for i := lo; i < hi; i++ {
							if cs[i] != 0 {
								os[i] = as[i]
							} else {
								os[i] = bs[i]
							}
						}
					})
					return []*tensor.Tensor{out}, nil
				}
			case tensor.F32:
				if cond.Dtype() == tensor.F32 {
					cs, as, bs := cond.Contiguous().Storage().F32(), a.Contiguous().Storage().F32(), b.Contiguous().Storage().F32()
					out := tensor.NewOn(ctx.Device(), tensor.F32, a.Shape())
					os := out.Storage().F32()
					parallel(len(os), func(lo, hi int) {
						for i := lo; i < hi; i++ {
							if cs[i] != 0 {
								os[i] = as[i]
							} else {
								os[i] = bs[i]
							}
						}
					})
					return []*tensor.Tensor{out}, nil
				}
			}
		}
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpWhere, in, attrs)
}

// clipKernelCPU is the devirtualized+parallel clamp for OpClip (Max(Lo,Min(x,Hi))). The reference
// kernel is devirtualized but single-threaded; here the F64/F32 fast paths fan the clamp across cores.
// Bit-exact: same math.Max/math.Min clamp, F32 clamps in float64 and rounds only the store — exactly
// the ref. The Lo>Hi error and exotic dtypes delegate to the reference kernel.
func clipKernelCPU(ctx *backend.Context, in []*tensor.Tensor, attrs backend.Attrs) ([]*tensor.Tensor, error) {
	pa, _ := attrs.(backend.ClipAttrs)
	if len(in) == 1 && pa.Lo <= pa.Hi {
		x := in[0]
		switch x.Dtype() {
		case tensor.F64:
			xs := x.Contiguous().Storage().F64()
			out := tensor.NewOn(ctx.Device(), tensor.F64, x.Shape())
			os := out.Storage().F64()
			parallel(len(os), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					os[i] = math.Max(pa.Lo, math.Min(xs[i], pa.Hi))
				}
			})
			return []*tensor.Tensor{out}, nil
		case tensor.F32:
			xs := x.Contiguous().Storage().F32()
			out := tensor.NewOn(ctx.Device(), tensor.F32, x.Shape())
			os := out.Storage().F32()
			parallel(len(os), func(lo, hi int) {
				for i := lo; i < hi; i++ {
					os[i] = float32(math.Max(pa.Lo, math.Min(float64(xs[i]), pa.Hi)))
				}
			})
			return []*tensor.Tensor{out}, nil
		}
	}
	return backend.Execute(ctx.WithBackend(backend.Reference()).WithRecorder(nil), backend.OpClip, in, attrs)
}

func init() {
	reg := func(op backend.Op, k backend.Kernel) {
		//perfscan:ignore PS3052 false-positive: std.add kernel-registration in init, op/k not staging buffer
		std.add(op, tensor.F32, k)
		std.add(op, tensor.F64, k)
	}
	reg(backend.OpAdd, binOp(simd.AddF64, simd.AddF32))
	reg(backend.OpSub, binOp(simd.SubF64, simd.SubF32))
	reg(backend.OpMul, binOp(simd.MulF64, simd.MulF32))
	reg(backend.OpDiv, binOp(simd.DivF64, simd.DivF32))
	// Elementwise Max/Min of two tensors and unary Abs/Sqrt: previously CPU-unregistered, so they
	// fell to the single-threaded reference kernels (a per-element func-VALUE indirect call). These
	// devirtualize the math and fan it across cores via binOp/parallel. Bit-exact with the ref.
	reg(backend.OpMaximum, binOp(maxSliceF64, maxSliceF32))
	reg(backend.OpMinimum, binOp(minSliceF64, minSliceF32))
	reg(backend.OpAbs, absKernelCPU)
	reg(backend.OpSqrt, sqrtKernelCPU)
	// Select/clamp: devirtualized in ref but single-threaded and CPU-unregistered → parallelize.
	reg(backend.OpWhere, whereKernelCPU)
	reg(backend.OpClip, clipKernelCPU)

	reg(backend.OpStopGradient, stopGradKernelCPU)
	reg(backend.OpNeg, negKernelCPU)
	reg(backend.OpExp, expKernelCPU)
	reg(backend.OpLog, logKernelCPU)
	reg(backend.OpTanh, tanhKernelCPU)
	reg(backend.OpReLU, reluKernelCPU)
	reg(backend.OpGELU, geluKernelCPU)
	reg(backend.OpSigmoid, sigmoidKernelCPU)
	reg(backend.OpSiLU, siluKernelCPU)

	// F64-only: softplus (Mamba/Jamba Δ) rides the amd64 vsoftplusF64 SIMD kernel;
	// F32 stays on the ref fallback. The non-SIMD build runs the scalar formula.
	std.add(backend.OpSoftplus, tensor.F64, softplusKernelCPU)
	std.add(backend.OpSoftplus, tensor.F32, softplusKernelCPU)
	std.add(backend.OpSoftplusBackward, tensor.F64, softplusBackwardKernelCPU)
	// F64-only: soft-cap (Gemma-2 attention/final logits) rides vsoftcapF64.
	std.add(backend.OpSoftCap, tensor.F64, softCapKernelCPU)
	std.add(backend.OpSoftCap, tensor.F32, softCapKernelCPU)

	if vexpF32Fast {
		// SIMD perf build, F32 only (§T664/§T665): the vectorized GELU and SiLU VJPs
		// — 4-wide NEON on arm64, 8-wide AVX2 on amd64. gelu_backward was 18.9% of
		// the f32 GPT training step; silu_backward is the SwiGLU-FFN VJP on every
		// Llama/Qwen/Mistral layer. The plain build keeps the ref fallbacks.
		std.add(backend.OpGELUBackward, tensor.F32, geluBackwardKernelCPU)
		std.add(backend.OpSiLUBackward, tensor.F32, siluBackwardKernelCPU)
		// Softplus VJP (dx = g·σ(x)) — the Mamba/Jamba Δ, Griffin/Hymba gate, and focal-loss
		// backward; forward vsoftplusF32 + the silu/gelu backward F32 kernels above all shipped,
		// this was the one sibling left on the serial ref fallback.
		std.add(backend.OpSoftplusBackward, tensor.F32, softplusBackwardKernelCPU)
	}
}

// broadcastRuns materializes a broadcast by hoisting the INNERMOST axis out of the
// odometer. Its effective stride is constant across a run, so the run is either one
// repeated source value (stride 0 — a broadcast axis, a fill) or a straight
// contiguous read (stride 1 — the axis survives), leaving the odometer to tick once
// per run instead of once per element. The strided arm keeps the walk total.
//
// Bit-identical by construction: this is a pure memmove reordering with no
// arithmetic on the values, and every dst position still receives the same src
// offset the per-element odometer computed. Over a full innermost run the odometer's
// net contribution to off is inner*sInner - sInner*inner = 0, so the outer tick is
// unchanged.
func broadcastRuns[T float32 | float64](dst, src []T, bstride []int, outShape tensor.Shape, idx []int, n int) {
	ndo := len(outShape)
	if ndo == 0 { // scalar output: the odometer never ticked
		dst[0] = src[0]
		return
	}
	inner, sInner := outShape[ndo-1], bstride[ndo-1]
	off := 0
	for pos := 0; pos < n; pos += inner {
		run := dst[pos : pos+inner]
		switch sInner {
		case 0:
			v := src[off]
			for i := range run {
				run[i] = v
			}
		case 1:
			copy(run, src[off:off+inner])
		default:
			o := off
			for i := range run {
				run[i] = src[o]
				o += sInner
			}
		}
		for d := ndo - 2; d >= 0; d-- {
			idx[d]++
			off += bstride[d]
			if idx[d] < outShape[d] {
				break
			}
			idx[d] = 0
			off -= bstride[d] * outShape[d]
		}
	}
}
