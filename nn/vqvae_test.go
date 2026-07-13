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

func nearestRef(ze, codebook [][]float64) []int {
	idx := make([]int, len(ze))
	for i := range ze {
		best, bd := 0, math.Inf(1)
		for j := range codebook {
			var d float64
			for c := range ze[i] {
				diff := ze[i][c] - codebook[j][c]
				d += diff * diff
			}
			if d < bd {
				best, bd = j, d
			}
		}
		idx[i] = best
	}
	return idx
}

// §V16 tier-1 (paper CONFIRMED, arXiv:1711.00937 §3.1): nearest-neighbor assignment,
// and the straight-through output's forward value is the selected codebook vector.
func TestVQNearestAndForward(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 40))
	const batch, k, d = 5, 4, 3
	ze := randMat(rng, batch, d)
	vq := nn.NewVectorQuantizer(tensor.F64, k, d, 0.25)
	for i := range vq.Codebook.Numel() {
		c := tensor.Unravel(i, vq.Codebook.Shape())
		vq.Codebook.SetF64(rng.NormFloat64(), c...)
	}
	zqST, _, idx, err := vq.Quantize(backend.NewContext(), ze)
	if err != nil {
		t.Fatal(err)
	}
	wantIdx := nearestRef(toRows(ze), toRows(vq.Codebook))
	for i := range idx {
		if idx[i] != wantIdx[i] {
			t.Errorf("idx[%d]=%d want %d", i, idx[i], wantIdx[i])
		}
		for c := 0; c < d; c++ { // zqST forward == codebook[idx[i]]
			if math.Abs(zqST.AtF64(i, c)-vq.Codebook.AtF64(idx[i], c)) > 1e-12 {
				t.Errorf("zqST[%d,%d]=%g want codebook[%d,%d]=%g", i, c, zqST.AtF64(i, c), idx[i], c, vq.Codebook.AtF64(idx[i], c))
			}
		}
	}
}

// §V2 (straight-through, §3.2): the gradient of a downstream loss on zqST flows to
// ze as the identity, and NOT to the codebook (recon path detaches the codebook).
func TestVQStraightThroughGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 41))
	const batch, k, d = 4, 3, 3
	ze := randMat(rng, batch, d)
	vq := nn.NewVectorQuantizer(tensor.F64, k, d, 0.25)
	for i := range vq.Codebook.Numel() {
		c := tensor.Unravel(i, vq.Codebook.Shape())
		vq.Codebook.SetF64(rng.NormFloat64(), c...)
	}
	coef := randMat(rng, batch, d)
	tape := autograd.NewTape()
	zqST, _, _, err := vq.Quantize(tape.Context(), ze)
	if err != nil {
		t.Fatal(err)
	}
	prod, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{zqST, coef}, nil)
	loss, _ := sumAll(tape.Context(), prod[0])
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gZe := tape.Grad(ze)
	for i := 0; i < batch; i++ {
		for c := 0; c < d; c++ { // straight-through ⇒ grad_ze == coef
			if math.Abs(gZe.AtF64(i, c)-coef.AtF64(i, c)) > 1e-12 {
				t.Errorf("grad_ze[%d,%d]=%g want coef %g", i, c, gZe.AtF64(i, c), coef.AtF64(i, c))
			}
		}
	}
	if gCb := tape.Grad(vq.Codebook); gCb != nil {
		for i := range gCb.Numel() { // recon path must not reach the codebook
			if v := gCb.AtF64(tensor.Unravel(i, gCb.Shape())...); math.Abs(v) > 1e-12 {
				t.Errorf("codebook got recon gradient %g, want 0 (detached)", v)
			}
		}
	}
}

// §V16 tier-1 (Eq.3) + §V2 (the two-way stop-gradient split, verified against the
// EXACT closed-form gradients — finite-diff is invalid through stop_gradient):
// grad_ze comes only from β·commitment, grad_codebook only from the codebook loss.
func TestVQLossGradientRouting(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 42))
	const batch, k, d = 5, 4, 3
	const beta = 0.25
	ze := randMat(rng, batch, d)
	vq := nn.NewVectorQuantizer(tensor.F64, k, d, beta)
	for i := range vq.Codebook.Numel() {
		c := tensor.Unravel(i, vq.Codebook.Shape())
		vq.Codebook.SetF64(rng.NormFloat64(), c...)
	}
	tape := autograd.NewTape()
	zqST, loss, idx, err := vq.Quantize(tape.Context(), ze)
	if err != nil {
		t.Fatal(err)
	}
	_ = zqST
	// loss parity against an independent recompute.
	n := float64(batch * d)
	var eLoss, cLoss float64
	for i := 0; i < batch; i++ {
		for c := 0; c < d; c++ {
			zq := vq.Codebook.AtF64(idx[i], c)
			eLoss += (ze.AtF64(i, c) - zq) * (ze.AtF64(i, c) - zq)
			cLoss += (ze.AtF64(i, c) - zq) * (ze.AtF64(i, c) - zq)
		}
	}
	eLoss, cLoss = eLoss/n, cLoss/n
	if math.Abs(loss.AtF64()-(eLoss+beta*cLoss)) > 1e-9 {
		t.Errorf("loss=%g want %g", loss.AtF64(), eLoss+beta*cLoss)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	// grad_ze = (2β/N)(ze − zq)  [codebook term detaches ze].
	gZe := tape.Grad(ze)
	for i := 0; i < batch; i++ {
		for c := 0; c < d; c++ {
			zq := vq.Codebook.AtF64(idx[i], c)
			want := (2 * beta / n) * (ze.AtF64(i, c) - zq)
			if math.Abs(gZe.AtF64(i, c)-want) > 1e-9 {
				t.Errorf("grad_ze[%d,%d]=%g want commit-only %g", i, c, gZe.AtF64(i, c), want)
			}
		}
	}
	// grad_codebook[k] = Σ_{i:idx=k} (2/N)(e_k − ze_i)  [commitment detaches codebook].
	gCb := tape.Grad(vq.Codebook)
	wantCb := make([][]float64, k)
	for j := range wantCb {
		wantCb[j] = make([]float64, d)
	}
	for i := 0; i < batch; i++ {
		for c := 0; c < d; c++ {
			wantCb[idx[i]][c] += (2 / n) * (vq.Codebook.AtF64(idx[i], c) - ze.AtF64(i, c))
		}
	}
	for j := 0; j < k; j++ {
		for c := 0; c < d; c++ {
			if math.Abs(gCb.AtF64(j, c)-wantCb[j][c]) > 1e-9 {
				t.Errorf("grad_codebook[%d,%d]=%g want codebook-only %g", j, c, gCb.AtF64(j, c), wantCb[j][c])
			}
		}
	}
}

// §V2: with β=0 the commitment term is off, so the encoder receives NO gradient from
// the VQ loss (only the codebook does) — the routing is exactly separable.
func TestVQZeroBetaNoEncoderGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 43))
	const batch, k, d = 4, 3, 2
	ze := randMat(rng, batch, d)
	vq := nn.NewVectorQuantizer(tensor.F64, k, d, 0)
	for i := range vq.Codebook.Numel() {
		c := tensor.Unravel(i, vq.Codebook.Shape())
		vq.Codebook.SetF64(rng.NormFloat64(), c...)
	}
	tape := autograd.NewTape()
	_, loss, _, err := vq.Quantize(tape.Context(), ze)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	if g := tape.Grad(ze); g != nil {
		for i := range g.Numel() {
			if v := g.AtF64(tensor.Unravel(i, g.Shape())...); math.Abs(v) > 1e-12 {
				t.Errorf("β=0 encoder gradient %g, want 0", v)
			}
		}
	}
	if g := tape.Grad(vq.Codebook); g == nil {
		t.Error("codebook should still receive gradient from the codebook loss")
	}
}

func TestVQErrors(t *testing.T) {
	vq := nn.NewVectorQuantizer(tensor.F64, 4, 3, 0.25)
	if _, _, _, err := vq.Quantize(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 1})); err == nil {
		t.Error("rank-3: want error")
	}
	if _, _, _, err := vq.Quantize(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 5})); err == nil {
		t.Error("dim mismatch: want error")
	}
}

func ExampleVectorQuantizer() {
	// 2 codebook entries; the input row is nearest to entry 1, so it quantizes there.
	vq := nn.NewVectorQuantizer(tensor.F64, 2, 2, 0.25)
	vq.Codebook.SetF64(0, 0, 0)
	vq.Codebook.SetF64(0, 0, 1) // e_0 = (0,0)
	vq.Codebook.SetF64(5, 1, 0)
	vq.Codebook.SetF64(5, 1, 1) // e_1 = (5,5)
	ze := toT([][]float64{{4, 6}})
	zqST, _, idx, _ := vq.Quantize(backend.NewContext(), ze)
	fmt.Printf("idx=%d zq=[%.0f %.0f]\n", idx[0], zqST.AtF64(0, 0), zqST.AtF64(0, 1))
	// Output:
	// idx=1 zq=[5 5]
}
