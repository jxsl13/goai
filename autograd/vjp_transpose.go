package autograd

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// blockT is the cache tile edge for the transpose copy; keeps a B×B source column
// band resident while the destination row band is written contiguously.
const blockT = 32

// Transpose VJP: for y = xᵀ, the gradient w.r.t. x is gᵀ (transpose is its own adjoint).
func init() {
	RegisterVJP(backend.OpTranspose, func(_ *backend.Context, in, _ []*tensor.Tensor, _ backend.Attrs, g *tensor.Tensor) ([]*tensor.Tensor, error) {
		x := in[0]
		m, n := x.Shape()[0], x.Shape()[1]
		gin := tensor.New(x.Dtype(), x.Shape())
		// Fast paths: the per-element AtF64/SetF64 walk pays an interface dispatch for
		// every one of m·n cells. When g and gin share the element dtype, copy through
		// the contiguous storage in cache tiles instead. gin is fresh (contiguous);
		// g is made contiguous. Pure permutation — bit-identical to the scalar loop:
		// (gᵀ)[i,j] = g[j,i], i.e. gin[i*n+j] = g[j*m+i].
		switch x.Dtype() {
		case tensor.F64:
			if g.Dtype() == tensor.F64 {
				gs := g.Contiguous().Storage().F64()
				ds := gin.Storage().F64()
				for ii := 0; ii < m; ii += blockT {
					iMax := min(ii+blockT, m)
					for jj := 0; jj < n; jj += blockT {
						jMax := min(jj+blockT, n)
						for i := ii; i < iMax; i++ {
							dbase := i * n
							for j := jj; j < jMax; j++ {
								ds[dbase+j] = gs[j*m+i]
							}
						}
					}
				}
				return []*tensor.Tensor{gin}, nil
			}
		case tensor.F32:
			if g.Dtype() == tensor.F32 {
				gs := g.Contiguous().Storage().F32()
				ds := gin.Storage().F32()
				for ii := 0; ii < m; ii += blockT {
					iMax := min(ii+blockT, m)
					for jj := 0; jj < n; jj += blockT {
						jMax := min(jj+blockT, n)
						for i := ii; i < iMax; i++ {
							dbase := i * n
							for j := jj; j < jMax; j++ {
								ds[dbase+j] = gs[j*m+i]
							}
						}
					}
				}
				return []*tensor.Tensor{gin}, nil
			}
		}
		for i := range m {
			for j := range n {
				gin.SetF64(g.AtF64(j, i), i, j) // (gᵀ)[i,j] = g[j,i]
			}
		}
		return []*tensor.Tensor{gin}, nil
	})
}
