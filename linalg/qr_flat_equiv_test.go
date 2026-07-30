package linalg

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// reference: the previous [][]float64 Householder QR, verbatim, as the golden.
func refQR(a *tensor.Tensor) (qMat, rMat []float64, m, n int) {
	m, n, _ = shapeMN(a)
	rm := make([][]float64, m)
	for i := 0; i < m; i++ {
		rm[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			rm[i][j] = a.AtF64(i, j)
		}
	}
	vs := make([][]float64, n)
	betas := make([]float64, n)
	for k := 0; k < n; k++ {
		var norm float64
		for i := k; i < m; i++ {
			norm += rm[i][k] * rm[i][k]
		}
		norm = math.Sqrt(norm)
		v := make([]float64, m)
		vs[k] = v
		if norm == 0 {
			continue
		}
		alpha := -norm
		if rm[k][k] < 0 {
			alpha = norm
		}
		for i := k; i < m; i++ {
			v[i] = rm[i][k]
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
		for j := k; j < n; j++ {
			s := 0.0
			for i := k; i < m; i++ {
				s += v[i] * rm[i][j]
			}
			bs := beta * s
			for i := k; i < m; i++ {
				rm[i][j] -= bs * v[i]
			}
		}
		rm[k][k] = alpha
		for i := k + 1; i < m; i++ {
			rm[i][k] = 0
		}
	}
	rMat = make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			rMat[i*n+j] = rm[i][j]
		}
	}
	qc := make([][]float64, m)
	for i := 0; i < m; i++ {
		qc[i] = make([]float64, n)
		if i < n {
			qc[i][i] = 1
		}
	}
	for k := n - 1; k >= 0; k-- {
		beta := betas[k]
		if beta == 0 {
			continue
		}
		v := vs[k]
		for j := 0; j < n; j++ {
			s := 0.0
			for i := k; i < m; i++ {
				s += v[i] * qc[i][j]
			}
			bs := beta * s
			for i := k; i < m; i++ {
				qc[i][j] -= bs * v[i]
			}
		}
	}
	qMat = make([]float64, m*n)
	for i := 0; i < m; i++ {
		copy(qMat[i*n:(i+1)*n], qc[i])
	}
	return
}

func TestQRFlatEquivRef(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 9))
	for trial := 0; trial < 500; trial++ {
		n := 1 + rng.IntN(30)
		m := n + rng.IntN(30)
		d := make([]float64, m*n)
		for i := range d {
			d[i] = rng.NormFloat64()
		}
		a := tensor.FromFloat64(tensor.Shape{m, n}, d)
		q, r, err := QR(a)
		if err != nil {
			t.Fatal(err)
		}
		wq, wr, _, _ := refQR(a)
		gq, gr := q.Storage().F64(), r.Storage().F64()
		for i := range wq {
			if math.Float64bits(gq[i]) != math.Float64bits(wq[i]) {
				t.Fatalf("trial %d m=%d n=%d Q[%d]: got %v want %v", trial, m, n, i, gq[i], wq[i])
			}
		}
		for i := range wr {
			if math.Float64bits(gr[i]) != math.Float64bits(wr[i]) {
				t.Fatalf("trial %d m=%d n=%d R[%d]: got %v want %v", trial, m, n, i, gr[i], wr[i])
			}
		}
	}
}
