package cpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Optimized Cholesky factorization: A = L·Lᵀ with L lower-triangular. OpCholesky had NO cpu
// kernel at all until now — every caller on the default backend fell through to ref, which is
// the correctness reference and is entirely serial.
//
// FANNING THE ROW UPDATE OUT WAS TRIED FIRST AND IS SLOWER. The column loop carries a real
// dependence and the row loop under it does not, which is the classical-factorization shape
// PS3040 describes — but there are n columns, so the fork is paid n times for a row update
// that shrinks as j grows. At n=512 that measured 24.1 ms against ref's 22.0, with allocations
// going from 6 to 878. What pays instead is arithmetic: four rows per pass, sharing the pivot
// row's loads and running four independent accumulator chains.
//
// Bit-identical to ref: each row's inner product runs over the same ascending k with the same
// operands into its own accumulator. The arithmetic is ref's, line for line, on purpose — a
// kernel that merely agreed to a tolerance would make every cross-backend golden a tolerance
// test.
//
// F32 is computed in float64 exactly as ref does and rounded once on the way out, so the two
// backends agree bit-for-bit in both dtypes rather than only in F64.
func choleskyKernel(ctx *backend.Context, in []*tensor.Tensor, _ backend.Attrs) ([]*tensor.Tensor, error) {
	if len(in) != 1 {
		return nil, fmt.Errorf("cpu: cholesky wants 1 input, got %d", len(in))
	}
	a := in[0]
	if a.Ndim() != 2 || a.Shape()[0] != a.Shape()[1] {
		return nil, fmt.Errorf("cpu: cholesky needs a square matrix, got shape %v", a.Shape())
	}
	n := a.Shape()[0]
	ac := a.Contiguous()
	var af64 []float64
	var af32 []float32
	switch ac.Dtype() {
	case tensor.F64:
		af64 = ac.Storage().F64()
	case tensor.F32:
		af32 = ac.Storage().F32()
	default:
		return nil, fmt.Errorf("cpu: unsupported dtype %v", a.Dtype())
	}
	at := func(i, j int) float64 {
		if af64 != nil {
			return af64[i*n+j]
		}
		return float64(af32[i*n+j]) // exact widening, as ref's AtF64 does
	}

	l := make([]float64, n*n)
	//perfscan:ignore PS3034 fan-out tried and measured slower (documented); column dependence
	for j := range n {
		lj := l[j*n : j*n+n]
		d := at(j, j)
		for k := range j {
			//perfscan:ignore PS3025 FMA-portability correctness class, not a throughput win
			d -= lj[k] * lj[k]
		}
		if d <= 0 {
			return nil, fmt.Errorf("cpu: cholesky: matrix is not positive-definite "+
				"(non-positive pivot %.6g at %d)", d, j)
		}
		ljj := math.Sqrt(d)
		lj[j] = ljj
		// FOUR ROWS PER PASS. Every row's inner product streams the SAME lj[k]; taking four
		// at once loads it once and runs four independent accumulator chains, which is where
		// the time was — a single-accumulator dot runs at FMA latency rather than throughput.
		// BIT-IDENTICAL, because the jammed dimension is the FREE one: row i still accumulates
		// over the same ascending k with the same operands into its own accumulator.
		i := j + 1
		//perfscan:ignore PS3066,PS5001 already-optimized factorization, loops are dependent stages not mergeable | reciprocal breaks bit-identity-wit
		for ; i+3 < n; i += 4 {
			l0 := l[(i+0)*n : (i+0)*n+j]
			l1 := l[(i+1)*n : (i+1)*n+j]
			l2 := l[(i+2)*n : (i+2)*n+j]
			l3 := l[(i+3)*n : (i+3)*n+j]
			s0, s1 := at(i+0, j), at(i+1, j)
			s2, s3 := at(i+2, j), at(i+3, j)
			for k, v := range lj[:j] {
				//perfscan:ignore PS3025 FMA-portability correctness class, not a throughput win
				s0 -= l0[k] * v
				//perfscan:ignore PS3025 FMA-portability correctness class, not a throughput win
				s1 -= l1[k] * v
				//perfscan:ignore PS3025 FMA-portability correctness class, not a throughput win
				s2 -= l2[k] * v
				//perfscan:ignore PS3025 FMA-portability correctness class, not a throughput win
				s3 -= l3[k] * v
			}
			l[(i+0)*n+j] = s0 / ljj
			l[(i+1)*n+j] = s1 / ljj
			l[(i+2)*n+j] = s2 / ljj
			l[(i+3)*n+j] = s3 / ljj
		}
		//perfscan:ignore PS5001 reciprocal breaks bit-identity claim; tail loop low trip count
		for ; i < n; i++ {
			li := l[i*n : i*n+n]
			s := at(i, j)
			for k := range j {
				//perfscan:ignore PS3025 FMA-portability correctness class, not a throughput win
				s -= li[k] * lj[k]
			}
			li[j] = s / ljj
		}
	}

	out := tensor.NewOn(ctx.Device(), a.Dtype(), tensor.Shape{n, n})
	switch a.Dtype() {
	case tensor.F64:
		os := out.Storage().F64()
		for i := range n {
			copy(os[i*n:i*n+i+1], l[i*n:i*n+i+1])
		}
	case tensor.F32:
		os := out.Storage().F32()
		for i := range n {
			for j := 0; j <= i; j++ {
				os[i*n+j] = float32(l[i*n+j])
			}
		}
	}
	return []*tensor.Tensor{out}, nil
}

func init() {
	std.add(backend.OpCholesky, tensor.F32, choleskyKernel)
	std.add(backend.OpCholesky, tensor.F64, choleskyKernel)
}
