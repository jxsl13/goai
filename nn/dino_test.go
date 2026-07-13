package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// softmaxT returns softmax((logits − sub)/temp) for one row.
func softmaxT(logits, sub []float64, temp float64) []float64 {
	k := len(logits)
	z := make([]float64, k)
	mx := math.Inf(-1)
	for i := range logits {
		z[i] = (logits[i] - sub[i]) / temp
		if z[i] > mx {
			mx = z[i]
		}
	}
	var sum float64
	for i := range z {
		z[i] = math.Exp(z[i] - mx)
		sum += z[i]
	}
	for i := range z {
		z[i] /= sum
	}
	return z
}

// dinoRef independently computes mean_b(−Σ_k P_t·log P_s), P_t=softmax((t−C)/τ_t),
// P_s=softmax(s/τ_s). §V16 tier-1 parity oracle.
func dinoRef(s, teach [][]float64, center []float64, tauS, tauT float64) float64 {
	b, k := len(s), len(s[0])
	zero := make([]float64, k)
	var total float64
	for i := 0; i < b; i++ {
		pt := softmaxT(teach[i], center, tauT)
		ps := softmaxT(s[i], zero, tauS)
		for j := 0; j < k; j++ {
			total -= pt[j] * math.Log(ps[j])
		}
	}
	return total / float64(b)
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2104.14294): matches the centered+sharpened
// cross-entropy reference.
func TestDINOLossFormula(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2104))
	s := randMat(rng, 4, 5)
	teach := randMat(rng, 4, 5)
	center := toT([][]float64{{0.1, -0.2, 0.05, 0.0, 0.3}})
	got, err := nn.DINOLoss(backend.NewContext(), s, teach, center, 0.1, 0.04)
	if err != nil {
		t.Fatal(err)
	}
	want := dinoRef(mat(s), mat(teach), mat(center)[0], 0.1, 0.04)
	if math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.12g want %.12g", got.AtF64(), want)
	}
}

// §V2 (closed form): when the student matches the teacher (g_s = (τ_s/τ_t)(g_t − C) ⇒
// P_s = P_t) the cross-entropy equals the teacher's entropy H(P_t) — the loss minimum.
func TestDINOLossPerfectMatch(t *testing.T) {
	const tauS, tauT = 0.1, 0.05
	teach := [][]float64{{1.0, 0.2, -0.5, 0.8}}
	center := []float64{0.1, 0.0, -0.1, 0.2}
	s := [][]float64{{}}
	s[0] = make([]float64, len(teach[0]))
	for k := range teach[0] {
		s[0][k] = (tauS / tauT) * (teach[0][k] - center[k]) // ⇒ P_s = P_t
	}
	loss, err := nn.DINOLoss(backend.NewContext(), toT(s), toT(teach), toT([][]float64{center}), tauS, tauT)
	if err != nil {
		t.Fatal(err)
	}
	pt := softmaxT(teach[0], center, tauT)
	var ent float64
	for _, p := range pt {
		ent -= p * math.Log(p)
	}
	if math.Abs(loss.AtF64()-ent) > 1e-9 {
		t.Errorf("perfect-match loss=%g, want teacher entropy H(P_t)=%g", loss.AtF64(), ent)
	}
}

// §V16 tier-1 (§3.2 centering EMA): C ← m·C + (1−m)·mean_batch(g_t).
func TestDINOCenterUpdate(t *testing.T) {
	center := toT([][]float64{{0.5, -0.5, 1.0}})
	teach := toT([][]float64{{1, 2, 3}, {3, 2, 1}}) // batch mean = [2,2,2]
	const m = 0.9
	got, err := nn.DINOCenterUpdate(center, teach, m)
	if err != nil {
		t.Fatal(err)
	}
	for j, mean := range []float64{2, 2, 2} {
		want := m*center.AtF64(0, j) + (1-m)*mean
		if math.Abs(got.AtF64(0, j)-want) > 1e-12 {
			t.Errorf("center[%d]=%g want %g", j, got.AtF64(0, j), want)
		}
	}
}

// §V2: differentiable w.r.t. the student logits (teacher + center are stop-gradient).
func TestDINOLossGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2105))
	s := randMat(rng, 3, 4)
	teach := randMat(rng, 3, 4)
	center := toT([][]float64{{0.1, 0.0, -0.1, 0.2}})
	tape := autograd.NewTape()
	loss, err := nn.DINOLoss(tape.Context(), s, teach, center, 0.2, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(s)
	if g == nil {
		t.Fatal("nil grad")
	}
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.DINOLoss(backend.NewContext(), s, teach, center, 0.2, 0.1)
		return l.AtF64()
	}
	for idx := range s.Numel() {
		coord := tensor.Unravel(idx, s.Shape())
		o := s.AtF64(coord...)
		s.SetF64(o+h, coord...)
		lp := fwd()
		s.SetF64(o-h, coord...)
		lm := fwd()
		s.SetF64(o, coord...)
		if num, ana := (lp-lm)/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad%v: numeric %.8g vs analytic %.8g", coord, num, ana)
		}
	}
}

func TestDINOLossErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.DINOLoss(backend.NewContext(), r(4, 5), r(4, 6), r(1, 5), 0.1, 0.04); err == nil {
		t.Error("student/teacher shape mismatch: want error")
	}
	if _, err := nn.DINOLoss(backend.NewContext(), r(4, 5), r(4, 5), r(1, 4), 0.1, 0.04); err == nil {
		t.Error("wrong center shape: want error")
	}
	if _, err := nn.DINOLoss(backend.NewContext(), r(4, 5), r(4, 5), r(1, 5), 0, 0.04); err == nil {
		t.Error("τ_s=0: want error")
	}
}

func ExampleDINOLoss() {
	// Student already matches the (centered, sharpened) teacher, so the loss is small — the
	// teacher's low entropy under a small τ_t.
	teach := toT([][]float64{{3.0, 0.0, 0.0}})
	center := toT([][]float64{{0, 0, 0}})
	s := toT([][]float64{{6.0, 0.0, 0.0}}) // (τ_s/τ_t)=2 ⇒ s = 2·teacher ⇒ P_s=P_t
	loss, _ := nn.DINOLoss(backend.NewContext(), s, teach, center, 0.1, 0.05)
	fmt.Println(loss.AtF64() < 0.05)
	// Output:
	// true
}
