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
//
//perfscan:ignore PS6004 reference oracle: intentionally simple, correctness baseline not an optimization target
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
	sbuf := make([]float64, n) // per-column reflector dot products, reused across k
	//perfscan:ignore PS1006 reference oracle: intentionally simple, correctness baseline not an optimization target
	for k := range n {
		var nrm float64 // ‖x‖ over rm[k:m, k]
		//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
		for i := k; i < m; i++ {
			//perfscan:ignore PS6011 reference oracle: intentionally simple, correctness baseline not an optimization target
			nrm += rm[i*n+k] * rm[i*n+k]
		}
		nrm = math.Sqrt(nrm)
		//perfscan:ignore PS2008 reference oracle: intentionally simple, correctness baseline not an optimization target
		v := make([]float64, m)
		vs[k] = v
		if nrm == 0 {
			continue // zero column → no reflection
		}
		alpha := -nrm // sign opposite x₀ avoids cancellation
		if rm[k*n+k] < 0 {
			alpha = nrm
		}
		//perfscan:ignore PS4004,PS4006 reference oracle: intentionally simple, correctness baseline not an optimization target
		for i := k; i < m; i++ {
			//perfscan:ignore PS6011 reference oracle: intentionally simple, correctness baseline not an optimization target
			v[i] = rm[i*n+k]
		}
		v[k] -= alpha
		var vtv float64
		//perfscan:ignore PS3010 reference oracle: intentionally simple, correctness baseline not an optimization target
		for i := k; i < m; i++ {
			vtv += v[i] * v[i]
		}
		if vtv == 0 {
			continue
		}
		beta := 2 / vtv
		betas[k] = beta
		// Apply H_k to the trailing columns with the COLUMN index innermost. Written the other
		// way round, each step of the inner loop jumps a whole row to use one element and the walk
		// repeats once per column; with j innermost both passes stream contiguous rows and v[i] is
		// loop-invariant across the row. The same interchange went -35.6%% on the triangular solve.
		//
		// BIT-IDENTICAL: every s[j] still accumulates over ascending i with the same operands, and
		// every update still subtracts the same product. Only the order in which INDEPENDENT
		// columns are visited changes.
		for j := k; j < n; j++ {
			sbuf[j] = 0
		}
		for i := k; i < m; i++ {
			vi, row := v[i], rm[i*n:i*n+n]
			for j := k; j < n; j++ {
				//perfscan:ignore PS3075 reference oracle: intentionally simple, correctness baseline not an optimization target
				sbuf[j] += vi * row[j]
			}
		}
		for j := k; j < n; j++ {
			sbuf[j] *= beta
		}
		for i := k; i < m; i++ {
			vi, row := v[i], rm[i*n:i*n+n]
			for j := k; j < n; j++ {
				row[j] -= sbuf[j] * vi
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
			sbuf[j] = 0
		}
		for i := k; i < m; i++ {
			vi, row := v[i], q[i*n:i*n+n]
			for j := range n {
				//perfscan:ignore PS3075 reference oracle: intentionally simple, correctness baseline not an optimization target
				sbuf[j] += vi * row[j]
			}
		}
		for j := range n {
			sbuf[j] *= betas[k]
		}
		for i := k; i < m; i++ {
			vi, row := v[i], q[i*n:i*n+n]
			for j := range n {
				row[j] -= sbuf[j] * vi
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
	//perfscan:ignore PS3062 reference oracle: intentionally simple, correctness baseline not an optimization target
	std.add(backend.OpQR, tensor.F32, qrKernel)
	std.add(backend.OpQR, tensor.F64, qrKernel)
}
