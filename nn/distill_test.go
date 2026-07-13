package nn_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func loadDistill(t *testing.T) (temp float64, shape []int, student, teacher []float64, loss float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/distill.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g struct {
		Temperature float64   `json:"temperature"`
		Shape       []int     `json:"shape"`
		Student     []float64 `json:"student"`
		Teacher     []float64 `json:"teacher"`
		Loss        float64   `json:"loss"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g.Temperature, g.Shape, g.Student, g.Teacher, g.Loss
}

// §V1 + §V16: the fused distillation loss matches an INDEPENDENT numpy T²·KL(p‖q)
// computation (Hinton et al. 2015) at f64 rtol 1e-12.
func TestDistillParity(t *testing.T) {
	temp, shape, student, teacher, want := loadDistill(t)
	s := tensor.FromFloat64(tensor.Shape{shape[0], shape[1]}, student)
	te := tensor.FromFloat64(tensor.Shape{shape[0], shape[1]}, teacher)
	got, err := nn.DistillLoss(backend.NewContext(), s, te, temp)
	if err != nil {
		t.Fatal(err)
	}
	if v := got.AtF64(); math.Abs(v-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Errorf("distill loss = %.17g, want %.17g", v, want)
	}
}

// §V2: central finite differences of the loss w.r.t. the STUDENT logits match the
// analytic VJP; the frozen teacher gets no gradient.
func TestDistillGradCheck(t *testing.T) {
	temp, shape, student, teacher, _ := loadDistill(t)
	m, c := shape[0], shape[1]
	teT := tensor.FromFloat64(tensor.Shape{m, c}, teacher)

	lossAt := func(sd []float64) float64 {
		out, err := nn.DistillLoss(backend.NewContext(), tensor.FromFloat64(tensor.Shape{m, c}, sd), teT, temp)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	sT := tensor.FromFloat64(tensor.Shape{m, c}, student)
	tape := autograd.NewTape()
	loss, err := nn.DistillLoss(tape.Context(), sT, teT, temp)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if tape.Grad(teT) != nil {
		t.Error("teacher logits must be frozen (nil gradient)")
	}
	gs := tape.Grad(sT)
	if gs == nil {
		t.Fatal("student logits must receive gradients")
	}
	const h = 1e-6
	work := append([]float64(nil), student...)
	for k := range student {
		o := work[k]
		work[k] = o + h
		lp := lossAt(work)
		work[k] = o - h
		lm := lossAt(work)
		work[k] = o
		num := (lp - lm) / (2 * h)
		ana := gs.AtF64(k/c, k%c)
		if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", k, num, ana)
		}
	}
}

// KL(p‖p) = 0: a student that already matches the teacher has zero distillation
// loss, for any temperature.
func TestDistillPerfectMatchIsZero(t *testing.T) {
	logits := []float64{2.0, -1.0, 0.5, 1.0, 0.0, 3.0}
	s := tensor.FromFloat64(tensor.Shape{2, 3}, logits)
	te := tensor.FromFloat64(tensor.Shape{2, 3}, logits)
	for _, T := range []float64{1.0, 2.0, 4.0} {
		out, err := nn.DistillLoss(backend.NewContext(), s, te, T)
		if err != nil {
			t.Fatal(err)
		}
		if v := out.AtF64(); math.Abs(v) > 1e-12 {
			t.Errorf("T=%g: perfect-match loss = %g, want 0", T, v)
		}
	}
}

// End-to-end: distilling a frozen teacher into a student — training the student
// logits on the soft loss drives the loss toward zero and matches the teacher's
// softened distribution (model compression, teacher frozen).
func TestDistillTrainsStudent(t *testing.T) {
	temp := 3.0
	student := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{0.1, 0.0, -0.1, 0.2, -0.2, 0.0})
	teacher := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{3.0, 0.5, -0.5, -1.0, 2.5, 0.3})
	opt := nn.NewAdam([]*tensor.Tensor{student}, 0.1)

	var first, last float64
	for step := range 300 {
		tape := autograd.NewTape()
		loss, err := nn.DistillLoss(tape.Context(), student, teacher, temp)
		if err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = loss.AtF64()
		}
		last = loss.AtF64()
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	if last >= 0.02*first {
		t.Fatalf("distillation did not converge: loss %.4f → %.4f", first, last)
	}
	// teacher must be byte-for-byte unchanged (frozen)
	if teacher.AtF64(0, 0) != 3.0 {
		t.Error("teacher logits changed — must be frozen")
	}
	t.Logf("distilled: soft loss %.4f → %.6f", first, last)
}
