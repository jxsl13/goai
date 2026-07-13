package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func klRow(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * (math.Log(a[i]) - math.Log(b[i]))
	}
	return s
}

// gkdRef independently computes the generalized JSD loss (mean over rows).
func gkdRef(student, teacher [][]float64, beta float64) float64 {
	var total float64
	for r := range student {
		q := softmaxRow(student[r]) // student = Q
		p := softmaxRow(teacher[r]) // teacher = P
		switch beta {
		case 0:
			total += klRow(p, q) // forward KL(P‖Q)
		case 1:
			total += klRow(q, p) // reverse KL(Q‖P)
		default:
			m := make([]float64, len(p))
			for i := range m {
				m[i] = beta*p[i] + (1-beta)*q[i]
			}
			total += beta*klRow(p, m) + (1-beta)*klRow(q, m)
		}
	}
	return total / float64(len(student))
}

var (
	gkdStudent = [][]float64{{2, 0.5, -1, 1}, {0.3, 1.2, 0.1, -0.5}}
	gkdTeacher = [][]float64{{1.5, 1, -0.5, 0.2}, {-0.2, 0.8, 1.5, 0.4}}
)

func tt(m [][]float64) *tensor.Tensor {
	rows, cols := len(m), len(m[0])
	t := tensor.New(tensor.F64, tensor.Shape{rows, cols})
	for r := range rows {
		for c := range cols {
			t.SetF64(m[r][c], r, c)
		}
	}
	return t
}

// §V16 tier-1: GKDLoss matches an independent generalized-JSD computation at the
// endpoints (β=0 forward KL, β=1 reverse KL) and in the interior (β=0.5 = JSD, β=0.3).
func TestGKDFormula(t *testing.T) {
	s, tea := tt(gkdStudent), tt(gkdTeacher)
	for _, beta := range []float64{0, 0.3, 0.5, 1} {
		got, err := nn.GKDLoss(backend.NewContext(), s, tea, beta)
		if err != nil {
			t.Fatalf("β=%g: %v", beta, err)
		}
		want := gkdRef(gkdStudent, gkdTeacher, beta)
		if math.Abs(got.AtF64()-want) > 1e-9 {
			t.Errorf("β=%g: GKDLoss = %.10g, want %.10g", beta, got.AtF64(), want)
		}
	}
}

// §V16 tier-1: at β=0.5 the JSD is symmetric — swapping student and teacher is a no-op.
func TestGKDSymmetric(t *testing.T) {
	s, tea := tt(gkdStudent), tt(gkdTeacher)
	a, _ := nn.GKDLoss(backend.NewContext(), s, tea, 0.5)
	b, _ := nn.GKDLoss(backend.NewContext(), tea, s, 0.5) // student/teacher swapped
	if math.Abs(a.AtF64()-b.AtF64()) > 1e-12 {
		t.Errorf("JSD(0.5) not symmetric: %.12g vs %.12g", a.AtF64(), b.AtF64())
	}
}

// §V16 tier-1: the generalized JSD is a non-negative divergence.
func TestGKDNonNegative(t *testing.T) {
	s, tea := tt(gkdStudent), tt(gkdTeacher)
	for _, beta := range []float64{0, 0.25, 0.5, 0.75, 1} {
		v, _ := nn.GKDLoss(backend.NewContext(), s, tea, beta)
		if v.AtF64() < -1e-12 {
			t.Errorf("β=%g: JSD = %.6g < 0", beta, v.AtF64())
		}
	}
}

// §V2: the loss is differentiable w.r.t. the student logits (the distillation target).
func TestGKDGradcheck(t *testing.T) {
	const beta = 0.5
	rows, cols := len(gkdStudent), len(gkdStudent[0])
	s := tt(gkdStudent)
	tea := tt(gkdTeacher)

	tape := autograd.NewTape()
	loss, err := nn.GKDLoss(tape.Context(), s, tea, beta)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(s)
	if g == nil {
		t.Fatal("nil student gradient")
	}
	const h = 1e-6
	for r := range rows {
		for c := range cols {
			orig := s.AtF64(r, c)
			s.SetF64(orig+h, r, c)
			lp, _ := nn.GKDLoss(backend.NewContext(), s, tea, beta)
			s.SetF64(orig-h, r, c)
			lm, _ := nn.GKDLoss(backend.NewContext(), s, tea, beta)
			s.SetF64(orig, r, c)
			num := (lp.AtF64() - lm.AtF64()) / (2 * h)
			if ana := g.AtF64(r, c); math.Abs(num-ana) > 1e-5*math.Max(1, math.Abs(num)) {
				t.Errorf("∂/∂student[%d,%d]: numeric %.8g vs analytic %.8g", r, c, num, ana)
			}
		}
	}
}

func TestGKDErrors(t *testing.T) {
	s := tt(gkdStudent)
	if _, err := nn.GKDLoss(backend.NewContext(), s, tt([][]float64{{1, 2}}), 0.5); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.GKDLoss(backend.NewContext(), s, tt(gkdTeacher), 1.5); err == nil {
		t.Error("beta>1: want error")
	}
}

func ExampleGKDLoss() {
	// Identical teacher and student ⇒ zero divergence at any β.
	logits := tt([][]float64{{1, 2, 3}})
	v, _ := nn.GKDLoss(backend.NewContext(), logits, logits, 0.5)
	fmt.Printf("%.4f\n", v.AtF64())
	// Output:
	// 0.0000
}
