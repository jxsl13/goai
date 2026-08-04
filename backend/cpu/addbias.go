package cpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/simd"
	"github.com/jxsl13/goai/tensor"
)

// Optimized row-broadcast bias add: out[i,j] = x[i,j] + b[j]. Each row is one
// simd.Add over contiguous slices (AVX 8/4-lane under GOEXPERIMENT=simd, plain
// loop otherwise), parallel over rows. Bit-identical to ref (§V3): a SIMD lane
// add is the same IEEE-754 add, and every out[i,j] depends on exactly one
// (x[i,j], b[j]) pair, so lane order cannot change results.
func addBiasKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("cpu: addbias wants 2 inputs, got %d", len(in))
	}
	x, b := in[0], in[1]
	if x.Ndim() != 2 || b.Ndim() != 1 {
		return nil, fmt.Errorf("cpu: addbias needs x rank-2 and b rank-1, got %dD, %dD", x.Ndim(), b.Ndim())
	}
	if x.Dtype() != b.Dtype() {
		return nil, fmt.Errorf("cpu: addbias dtype mismatch %v vs %v", x.Dtype(), b.Dtype())
	}
	m, n := x.Shape()[0], x.Shape()[1]
	if b.Shape()[0] != n {
		return nil, fmt.Errorf("cpu: addbias bias len %d != cols %d", b.Shape()[0], n)
	}
	xc, bc := x.Contiguous(), b.Contiguous()
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())

	switch x.Dtype() {
	case tensor.F64:
		X, B, O := xc.Storage().F64(), bc.Storage().F64(), out.Storage().F64()
		parallelWork(m, n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				simd.AddF64(O[i*n:(i+1)*n], X[i*n:(i+1)*n], B)
			}
		})
	case tensor.F32:
		X, B, O := xc.Storage().F32(), bc.Storage().F32(), out.Storage().F32()
		parallelWork(m, n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				simd.AddF32(O[i*n:(i+1)*n], X[i*n:(i+1)*n], B)
			}
		})
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", x.Dtype())
	}
	return []*tensor.Tensor{out}, nil
}

// addBiasBackwardKernel computes dbias[j] = Σ_i g[i,j] — the column sum of
// g[m,n] — parallel over COLUMN ranges. Each parallelWork chunk owns a
// disjoint slice of columns [lo,hi), so no goroutine ever touches another's
// accumulator (no partial-sum reduction, no race). Within a chunk the loops
// run rows-outer / columns-inner, streaming g row-major for cache locality;
// per column j the adds still happen in row order i = 0..m−1 into a single
// float64 accumulator, exactly the sequence ref's serial column loop performs,
// so the result is bit-identical to ref (§V3) on both dtypes. The F32 path
// accumulates in float64 and rounds only the store, mirroring ref (§V10).
func addBiasBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("cpu: addbias-backward wants (g), got %d", len(in))
	}
	g := in[0]
	if g.Ndim() != 2 {
		return nil, fmt.Errorf("cpu: addbias-backward needs g rank-2, got %dD", g.Ndim())
	}
	m, n := g.Shape()[0], g.Shape()[1]
	gc := g.Contiguous()
	db := tensor.NewOn(ctx.Device(), g.Dtype(), tensor.Shape{n})

	switch g.Dtype() {
	case tensor.F64:
		gs, dbs := gc.Storage().F64(), db.Storage().F64()
		parallelWork(n, m, func(lo, hi int) {
			acc := dbs[lo:hi] // zero-initialized output doubles as the f64 accumulator
			for i := range m {
				row := gs[i*n+lo : i*n+hi]
				for k, v := range row {
					acc[k] += v
				}
			}
		})
	case tensor.F32:
		gs, dbs := gc.Storage().F32(), db.Storage().F32()
		parallelWork(n, m, func(lo, hi int) {
			//perfscan:ignore PS6008 f64 accum scratch per parallel chunk (not per-row); required §V10 numerics
			acc := make([]float64, hi-lo) // f64 accumulation (§V10); round only the store
			for i := range m {
				row := gs[i*n+lo : i*n+hi]
				for k, v := range row {
					acc[k] += float64(v)
				}
			}
			for k, s := range acc {
				dbs[lo+k] = float32(s)
			}
		})
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", g.Dtype())
	}
	return []*tensor.Tensor{db}, nil
}

func init() {
	std.add(backend.OpAddBias, tensor.F32, addBiasKernel)
	std.add(backend.OpAddBias, tensor.F64, addBiasKernel)
	std.add(backend.OpAddBiasBackward, tensor.F32, addBiasBackwardKernel)
	std.add(backend.OpAddBiasBackward, tensor.F64, addBiasBackwardKernel)
}
