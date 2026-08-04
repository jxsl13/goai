package ref

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// addBiasKernel is the reference row-broadcast bias add (§T15): out[i,j] =
// x[i,j] + b[j] for x[m,n], b[n]. The dominant NN case; general broadcasting is
// deferred (§B18).
func addBiasKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 2 {
		return nil, fmt.Errorf("ref: addbias wants 2 inputs, got %d", len(in))
	}
	x, b := in[0], in[1]
	if x.Ndim() != 2 || b.Ndim() != 1 {
		return nil, fmt.Errorf("ref: addbias needs x rank-2 and b rank-1, got %dD, %dD", x.Ndim(), b.Ndim())
	}
	if x.Dtype() != b.Dtype() {
		return nil, fmt.Errorf("ref: addbias dtype mismatch %v vs %v", x.Dtype(), b.Dtype())
	}
	m, n := x.Shape()[0], x.Shape()[1]
	if b.Shape()[0] != n {
		return nil, fmt.Errorf("ref: addbias bias len %d != cols %d", b.Shape()[0], n)
	}
	out := tensor.NewOn(ctx.Device(), x.Dtype(), x.Shape())

	// Devirtualised fast paths (§T646): the generic AtF64/SetF64 loop below pays a
	// dtype dispatch + flat-offset computation per element. x and b share a dtype
	// (guarded above); when it is a plain float type we grab the raw typed slices
	// once (row-major: (i,j) of [m,n] at i*n+j, b[j] at j) and index directly.
	// Iteration order and arithmetic are byte-for-byte identical to the generic
	// path; the F32 path reads inputs as float64 and rounds the STORED sum once.
	// Contiguous() is called once per tensor (returns self when already contiguous).
	switch x.Dtype() {
	case tensor.F64:
		xs := x.Contiguous().Storage().F64()
		bs := b.Contiguous().Storage().F64()
		os := out.Storage().F64()
		for i := range m {
			base := i * n
			for j := range n {
				os[base+j] = xs[base+j] + bs[j]
			}
		}
		return []*tensor.Tensor{out}, nil
	case tensor.F32:
		xs := x.Contiguous().Storage().F32()
		bs := b.Contiguous().Storage().F32()
		os := out.Storage().F32()
		for i := range m {
			base := i * n
			for j := range n {
				os[base+j] = float32(float64(xs[base+j]) + float64(bs[j]))
			}
		}
		return []*tensor.Tensor{out}, nil
	}

	// Generic fallback for exotic dtypes (verbatim original loop).
	for i := range m {
		for j := range n {
			out.SetF64(x.AtF64(i, j)+b.AtF64(j), i, j)
		}
	}
	return []*tensor.Tensor{out}, nil
}

// addBiasBackwardKernel computes dbias[j] = Σ_i g[i,j] (the column sum), f64 accumulation
// (§V10). The bias-add VJP dispatches this so the reduction runs on the active backend
// instead of a CPU scalar loop (§T354); the input gradient is g itself (identity).
func addBiasBackwardKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: addbias-backward wants (g), got %d", len(in))
	}
	g := in[0]
	if g.Ndim() != 2 {
		return nil, fmt.Errorf("ref: addbias-backward needs g rank-2, got %dD", g.Ndim())
	}
	m, n := g.Shape()[0], g.Shape()[1]
	db := tensor.NewOn(ctx.Device(), g.Dtype(), tensor.Shape{n})

	// Devirtualised fast paths (§T646): the hottest cpu→ref fallback on the CPU
	// training path. The generic AtF64/SetF64 loop pays a dtype dispatch + flat
	// offset per element; here we grab the raw typed slices once (g[i,j] of [m,n]
	// at i*n+j, db[j] at j) and index directly. The column-sum accumulator stays
	// float64 on ALL paths and the F32 path only rounds the STORED result — so the
	// reduction is byte-for-byte identical to the generic path. Contiguous() runs
	// once (returns self when already contiguous).
	switch g.Dtype() {
	case tensor.F64:
		gs := g.Contiguous().Storage().F64()
		dbs := db.Storage().F64()
		//perfscan:ignore PS1006 reference oracle: intentionally simple, correctness baseline not an optimization target
		for j := range n {
			var s float64
			//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i := range m {
				//perfscan:ignore PS6011 reference oracle: intentionally simple, correctness baseline not an optimization target
				s += gs[i*n+j]
			}
			dbs[j] = s
		}
		return []*tensor.Tensor{db}, nil
	case tensor.F32:
		gs := g.Contiguous().Storage().F32()
		dbs := db.Storage().F32()
		//perfscan:ignore PS1006 reference oracle: intentionally simple, correctness baseline not an optimization target
		for j := range n {
			var s float64 // column-sum accumulates in float64; only the store rounds
			//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
			for i := range m {
				//perfscan:ignore PS6011 reference oracle: intentionally simple, correctness baseline not an optimization target
				s += float64(gs[i*n+j])
			}
			dbs[j] = float32(s)
		}
		return []*tensor.Tensor{db}, nil
	}

	// Generic fallback for exotic dtypes (verbatim original loop).
	for j := range n {
		var s float64
		for i := range m {
			s += g.AtF64(i, j)
		}
		db.SetF64(s, j)
	}
	return []*tensor.Tensor{db}, nil
}

func init() {
	std.add(backend.OpAddBias, tensor.F32, addBiasKernel)
	std.add(backend.OpAddBias, tensor.F64, addBiasKernel)
	std.add(backend.OpAddBiasBackward, tensor.F32, addBiasBackwardKernel)
	std.add(backend.OpAddBiasBackward, tensor.F64, addBiasBackwardKernel)
}
