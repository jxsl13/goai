package ref

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// qrKernel computes the reduced (economy) QR factorization of an m×n matrix A with
// m ≥ n via Householder reflectors (numpy.linalg.qr mode='reduced'; Golub & Van Loan
// §5.1-5.2, LAPACK dgeqrf): A = Q·R with Q ∈ Rᵐˣⁿ orthonormal columns (QᵀQ = Iₙ) and
// R ∈ Rⁿˣⁿ upper-triangular. Householder QR is backward-stable. Two outputs [Q, R];
// all arithmetic is f64 (§V10), F32 input factors in f64 and narrows on store.
func qrKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("ref: qr wants 1 input, got %d", len(in))
	}
	a := in[0]
	if a.Ndim() != 2 {
		return nil, fmt.Errorf("ref: qr needs a rank-2 matrix, got shape %v", a.Shape())
	}
	m, n := a.Shape()[0], a.Shape()[1]
	if m < n {
		return nil, fmt.Errorf("ref: qr needs m ≥ n (tall/square), got %dx%d", m, n)
	}

	// working copy (becomes R in its upper triangle); reflectors stored in vs/betas.
	// Working matrix as ONE flat [m*n] row-major buffer, not m allocated rows. The
	// Householder loops below walk COLUMNS (rm[i][k] with i varying), so with a
	// row-of-slices layout every step dereferences a different heap row. Flat makes
	// that a constant stride through one allocation. Index arithmetic only, so the
	// factorization is bit-identical.
	rm := make([]float64, m*n)
	if as, ok := f64Data(a); ok {
		copy(rm, as)
	} else {
		for i := range m {
			for j := range n {
				rm[i*n+j] = a.AtF64(i, j)
			}
		}
	}
	vs := make([][]float64, n)
	betas := make([]float64, n)
	for k := range n {
		var nrm float64 // ‖x‖ over rm[k:m, k]
		for i := k; i < m; i++ {
			nrm += rm[i*n+k] * rm[i*n+k]
		}
		nrm = math.Sqrt(nrm)
		v := make([]float64, m)
		vs[k] = v
		if nrm == 0 {
			continue // zero column → no reflection
		}
		alpha := -nrm // sign opposite x₀ avoids cancellation
		if rm[k*n+k] < 0 {
			alpha = nrm
		}
		for i := k; i < m; i++ {
			v[i] = rm[i*n+k]
		}
		v[k] -= alpha
		var vtv float64
		for i := k; i < m; i++ {
			vtv += v[i] * v[i]
		}
		if vtv == 0 {
			continue
		}
		beta := 2 / vtv
		betas[k] = beta
		for j := k; j < n; j++ { // apply H_k to trailing columns
			var s float64
			for i := k; i < m; i++ {
				s += v[i] * rm[i*n+j]
			}
			bs := beta * s
			for i := k; i < m; i++ {
				rm[i*n+j] -= bs * v[i]
			}
		}
		rm[k*n+k] = alpha
		for i := k + 1; i < m; i++ {
			rm[i*n+k] = 0
		}
	}

	// R = upper n×n triangle.
	rt := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n, n})
	for i := range n {
		for j := i; j < n; j++ {
			rt.SetF64(rm[i*n+j], i, j)
		}
	}
	// reduced Q = H_0···H_{n-1} applied to the first n columns of Iₘ (reflectors in reverse).
	// Same flattening for the Q accumulator: the reflector application below walks
	// q[i][j] with i varying, another column walk.
	q := make([]float64, m*n)
	for i := range m {
		if i < n {
			q[i*n+i] = 1
		}
	}
	for k := n - 1; k >= 0; k-- {
		if betas[k] == 0 {
			continue
		}
		v := vs[k]
		for j := range n {
			var s float64
			for i := k; i < m; i++ {
				s += v[i] * q[i*n+j]
			}
			bs := betas[k] * s
			for i := k; i < m; i++ {
				q[i*n+j] -= bs * v[i]
			}
		}
	}
	qt := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			qt.SetF64(q[i*n+j], i, j)
		}
	}
	return []*tensor.Tensor{qt, rt}, nil
}

func init() {
	std.add(backend.OpQR, tensor.F32, qrKernel)
	std.add(backend.OpQR, tensor.F64, qrKernel)
}
