package linalg_test

import (
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// The exported factorizations had no benchmark. BenchmarkCholSolveMat and BenchmarkLstsqMat do not
// reach them: CholSolve calls cholFactor directly and Lstsq calls toFlat/householder directly, so
// both skip the tensor-assembly loops at the top of Cholesky and QR. Their real callers are
// nn/gptq.go and nn/sparsegpt.go for Cholesky and nn/init.go's Orthogonal for QR.
//
// Measuring the assembly loop without an instrument that reaches it is how a base arm ends up
// looking clean while nothing exercised the changed line at all.

// spdMatrix returns a diagonally dominant symmetric matrix, which is SPD and so factors without
// hitting the not-positive-definite error path.
func spdMatrix(n int) *tensor.Tensor {
	d := make([]float64, n*n)
	for i := range n {
		for j := range n {
			v := float64((i*7+j*13)%11) * 0.05
			d[i*n+j] += v
			d[j*n+i] += v
		}
		d[i*n+i] += float64(n) + 4
	}
	return tensor.FromFloat64(tensor.Shape{n, n}, d)
}

func BenchmarkCholeskyFactor(b *testing.B) {
	for _, n := range []int{64, 256, 512} {
		a := spdMatrix(n)
		b.Run(benchN(n), func(b *testing.B) {
			for b.Loop() {
				if _, err := linalg.Cholesky(a); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkQRFactor(b *testing.B) {
	for _, n := range []int{64, 256, 512} {
		d := make([]float64, n*n)
		for i := range d {
			d[i] = float64((i*31)%97) * 0.25
		}
		a := tensor.FromFloat64(tensor.Shape{n, n}, d)
		b.Run(benchN(n), func(b *testing.B) {
			for b.Loop() {
				if _, _, err := linalg.QR(a); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func benchN(n int) string {
	if n == 0 {
		return "0"
	}
	var d [8]byte
	i := len(d)
	for n > 0 {
		i--
		d[i] = byte('0' + n%10)
		n /= 10
	}
	return "n" + string(d[i:])
}
