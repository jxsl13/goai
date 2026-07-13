package autograd_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func rectTensor(a [][]float64) *tensor.Tensor {
	m, n := len(a), len(a[0])
	t := tensor.New(tensor.F64, tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			t.SetF64(a[i][j], i, j)
		}
	}
	return t
}

func randRect(rng *rand.Rand, m, n int) [][]float64 {
	a := make([][]float64, m)
	for i := range m {
		a[i] = make([]float64, n)
		for j := range n {
			a[i][j] = rng.NormFloat64()
		}
	}
	return a
}

// qrLoss = Σ WQ⊙Q + Σ WR⊙R (WR nil ⇒ Q-only), the scalar whose A-gradient is checked.
func qrLoss(a, wq, wr [][]float64) (float64, error) {
	out, err := backend.Execute(backend.NewContext(), backend.OpQR, []*tensor.Tensor{rectTensor(a)}, nil)
	if err != nil {
		return 0, err
	}
	q, r := out[0], out[1]
	m, n := len(a), len(a[0])
	var s float64
	for i := range m {
		for j := range n {
			s += wq[i][j] * q.AtF64(i, j)
		}
	}
	if wr != nil {
		for i := range n {
			for j := range n {
				s += wr[i][j] * r.AtF64(i, j)
			}
		}
	}
	return s, nil
}

// §V2: the Seeger 2017 QR gradient matches central finite differences, for square
// (m=n) and tall (m>n) matrices, with both Q̄ and R̄ active. Also exercises the
// multi-output tape path (both Q and R feed the loss).
func TestQRGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 7))
	const h = 1e-6
	shapes := [][2]int{{2, 2}, {3, 3}, {4, 2}, {5, 3}}
	for _, s := range shapes {
		m, n := s[0], s[1]
		a := randRect(rng, m, n)
		wq := randRect(rng, m, n)
		wr := randRect(rng, n, n)

		at := rectTensor(a)
		wqt := rectTensor(wq)
		wrt := rectTensor(wr)
		tape := autograd.NewTape()
		out, err := backend.Execute(tape.Context(), backend.OpQR, []*tensor.Tensor{at}, nil)
		if err != nil {
			t.Fatalf("%dx%d forward: %v", m, n, err)
		}
		pq, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[0], wqt}, nil)
		pr, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[1], wrt}, nil)
		sq, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pq[0]}, nil)
		sr, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pr[0]}, nil)
		loss, _ := backend.Execute(tape.Context(), backend.OpAdd, []*tensor.Tensor{sq[0], sr[0]}, nil)
		if err := tape.Backward(loss[0]); err != nil {
			t.Fatalf("%dx%d backward: %v", m, n, err)
		}
		abar := tape.Grad(at)
		if abar == nil {
			t.Fatalf("%dx%d: nil gradient", m, n)
		}
		for i := range m {
			for j := range n {
				ap := deepCopy(a)
				am := deepCopy(a)
				ap[i][j] += h
				am[i][j] -= h
				lp, _ := qrLoss(ap, wq, wr)
				lm, _ := qrLoss(am, wq, wr)
				num := (lp - lm) / (2 * h)
				if ana := abar.AtF64(i, j); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
					t.Errorf("%dx%d Ā[%d,%d]: numeric %.8g vs analytic %.8g", m, n, i, j, num, ana)
				}
			}
		}
	}
}

// §V2 + multi-output zero-cotangent path: a loss that uses ONLY Q (R unused ⇒ its
// cotangent is filled with zeros by the tape) still gradchecks.
func TestQRGradcheckQOnly(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 8))
	const h = 1e-6
	m, n := 4, 3
	a := randRect(rng, m, n)
	wq := randRect(rng, m, n)

	at := rectTensor(a)
	wqt := rectTensor(wq)
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), backend.OpQR, []*tensor.Tensor{at}, nil)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	pq, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out[0], wqt}, nil) // R untouched
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pq[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatalf("backward: %v", err)
	}
	abar := tape.Grad(at)
	for i := range m {
		for j := range n {
			ap := deepCopy(a)
			am := deepCopy(a)
			ap[i][j] += h
			am[i][j] -= h
			lp, _ := qrLoss(ap, wq, nil)
			lm, _ := qrLoss(am, wq, nil)
			num := (lp - lm) / (2 * h)
			if ana := abar.AtF64(i, j); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("Ā[%d,%d]: numeric %.8g vs analytic %.8g", i, j, num, ana)
			}
		}
	}
}
