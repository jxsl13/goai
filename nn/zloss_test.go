package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// zlossRefLoss is an independent recomputation of cross-entropy + PaLM z-loss over a [b,c] logit
// block: per row lse=logsumexp; loss += (lse − z[target]) + coeff·lse²; mean over rows.
func zlossRefLoss(zd []float64, ty []float64, b, c int, coeff float64) float64 {
	total := 0.0
	for i := range b {
		row := zd[i*c : (i+1)*c]
		m := math.Inf(-1)
		for _, v := range row {
			if v > m {
				m = v
			}
		}
		s := 0.0
		for _, v := range row {
			s += math.Exp(v - m)
		}
		lse := m + math.Log(s)
		total += lse - row[int(ty[i])] + coeff*lse*lse
	}
	return total / float64(b)
}

// §V16 tier-1: CrossEntropyZLoss equals cross-entropy plus coeff·mean(logsumexp²) exactly (Chowdhery
// et al. 2022 PaLM z-loss, §R113), and reduces to plain CrossEntropy when the coefficient is 0.
func TestCrossEntropyZLossValue(t *testing.T) {
	const b, c, coeff = 2, 3, 1e-4
	zd := []float64{2.0, -1.0, 0.5, 0.3, 1.2, -0.7}
	ty := []float64{0, 2}
	ctx := backend.NewContext()
	z := tensor.FromFloat64(tensor.Shape{b, c}, zd)
	tg := tensor.FromFloat64(tensor.Shape{b}, ty)

	loss, err := nn.CrossEntropyZLoss(ctx, z, tg, coeff)
	if err != nil {
		t.Fatal(err)
	}
	if want := zlossRefLoss(zd, ty, b, c, coeff); math.Abs(loss.AtF64()-want) > 1e-12 {
		t.Errorf("CE+z-loss = %.14g, want %.14g", loss.AtF64(), want)
	}
	// coeff 0 must equal plain CrossEntropy
	plain, _ := nn.CrossEntropy(ctx, z, tg)
	z0, _ := nn.CrossEntropyZLoss(ctx, z, tg, 0)
	if math.Abs(plain.AtF64()-z0.AtF64()) > 1e-12 {
		t.Errorf("CrossEntropyZLoss(0) = %.14g, want plain CE %.14g", z0.AtF64(), plain.AtF64())
	}
	// z-loss strictly increases the loss (lse² ≥ 0, coeff > 0)
	if !(loss.AtF64() > plain.AtF64()) {
		t.Errorf("z-loss must add a positive penalty: %.14g vs %.14g", loss.AtF64(), plain.AtF64())
	}
}

// §V2 GRADCHECK: the analytic gradient of CrossEntropyZLoss w.r.t. the logits (which includes the
// z-loss term 2·coeff·lse·softmax) matches central finite differences.
func TestCrossEntropyZLossGradcheck(t *testing.T) {
	const b, c, coeff, h, rtol = 2, 3, 0.05, 1e-6, 1e-4
	zd := []float64{2.0, -1.0, 0.5, 0.3, 1.2, -0.7}
	ty := []float64{0, 2}

	tape := autograd.NewTape()
	ctx := tape.Context()
	z := tensor.FromFloat64(tensor.Shape{b, c}, zd)
	tg := tensor.FromFloat64(tensor.Shape{b}, ty)
	loss, err := nn.CrossEntropyZLoss(ctx, z, tg, coeff)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gz := tape.Grad(z)
	if gz == nil {
		t.Fatal("nil grad for logits")
	}
	for j := range zd {
		p := append([]float64(nil), zd...)
		m := append([]float64(nil), zd...)
		p[j] += h
		m[j] -= h
		numeric := (zlossRefLoss(p, ty, b, c, coeff) - zlossRefLoss(m, ty, b, c, coeff)) / (2 * h)
		analytic := gz.AtF64(tensor.Unravel(j, tensor.Shape{b, c})...)
		if math.Abs(numeric-analytic) > rtol*math.Max(1, math.Abs(numeric)) {
			t.Errorf("z-loss grad[%d]: numeric %.10g vs analytic %.10g", j, numeric, analytic)
		}
	}
}

func TestCrossEntropyZLossValidation(t *testing.T) {
	ctx := backend.NewContext()
	z := tensor.New(tensor.F64, tensor.Shape{2, 3})
	tg := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 1})
	if _, err := nn.CrossEntropyZLoss(ctx, z, tg, -1e-4); err == nil {
		t.Error("negative z-loss coefficient must be rejected")
	}
}

// CrossEntropyZLoss adds the PaLM z-loss coeff·(logsumexp)² to cross-entropy, penalising the softmax
// log-partition so it stays near 0 — a standard stabilizer for large-model training. Here, on a
// confident logit row, the z-loss makes the total loss slightly larger than plain cross-entropy.
func ExampleCrossEntropyZLoss() {
	ctx := backend.NewContext()
	z := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{4, 0, 0})
	tg := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	ce, _ := nn.CrossEntropy(ctx, z, tg)
	zl, _ := nn.CrossEntropyZLoss(ctx, z, tg, 1e-4)
	fmt.Println(zl.AtF64() > ce.AtF64())
	// Output: true
}

// zlossRefStandalone is an independent recomputation of the standalone z-loss coeff·mean(lse²).
func zlossRefStandalone(zd []float64, b, c int, coeff float64) float64 {
	total := 0.0
	for i := range b {
		row := zd[i*c : (i+1)*c]
		m := math.Inf(-1)
		for _, v := range row {
			if v > m {
				m = v
			}
		}
		s := 0.0
		for _, v := range row {
			s += math.Exp(v - m)
		}
		lse := m + math.Log(s)
		total += lse * lse
	}
	return coeff * total / float64(b)
}

// §V16 tier-1: standalone ZLoss equals coeff·mean(logsumexp²) exactly (ST-MoE router z-loss form,
// §R113), takes only logits (no targets), and is 0 when the coefficient is 0.
func TestZLossValue(t *testing.T) {
	const b, c, coeff = 2, 4, 1e-3
	zd := []float64{2.0, -1.0, 0.5, 3.0, 0.3, 1.2, -0.7, 0.1}
	ctx := backend.NewContext()
	z := tensor.FromFloat64(tensor.Shape{b, c}, zd)
	got, err := nn.ZLoss(ctx, z, coeff)
	if err != nil {
		t.Fatal(err)
	}
	if want := zlossRefStandalone(zd, b, c, coeff); math.Abs(got.AtF64()-want) > 1e-12 {
		t.Errorf("ZLoss = %.14g, want %.14g", got.AtF64(), want)
	}
	if z0, _ := nn.ZLoss(ctx, z, 0); z0.AtF64() != 0 {
		t.Errorf("ZLoss(coeff 0) = %v, want 0", z0.AtF64())
	}
}

// §V16 tier-1 (consistency): the standalone ZLoss equals the z-loss component fused into
// CrossEntropyZLoss — i.e. CrossEntropyZLoss(z,t,coeff) − CrossEntropy(z,t) == ZLoss(z,coeff).
func TestZLossMatchesFusedComponent(t *testing.T) {
	const coeff = 5e-4
	ctx := backend.NewContext()
	z := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{2, -1, 0.5, 0.3, 1.2, -0.7})
	tg := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 2})
	ce, _ := nn.CrossEntropy(ctx, z, tg)
	cez, _ := nn.CrossEntropyZLoss(ctx, z, tg, coeff)
	zl, _ := nn.ZLoss(ctx, z, coeff)
	if fused, standalone := cez.AtF64()-ce.AtF64(), zl.AtF64(); math.Abs(fused-standalone) > 1e-12 {
		t.Errorf("fused z-loss component %.14g != standalone %.14g", fused, standalone)
	}
}

// §V2 GRADCHECK: the analytic gradient of the standalone ZLoss matches central finite differences.
func TestZLossGradcheck(t *testing.T) {
	const b, c, coeff, h, rtol = 2, 3, 0.1, 1e-6, 1e-4
	zd := []float64{2.0, -1.0, 0.5, 0.3, 1.2, -0.7}
	tape := autograd.NewTape()
	ctx := tape.Context()
	z := tensor.FromFloat64(tensor.Shape{b, c}, zd)
	loss, err := nn.ZLoss(ctx, z, coeff)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gz := tape.Grad(z)
	for j := range zd {
		p := append([]float64(nil), zd...)
		m := append([]float64(nil), zd...)
		p[j] += h
		m[j] -= h
		numeric := (zlossRefStandalone(p, b, c, coeff) - zlossRefStandalone(m, b, c, coeff)) / (2 * h)
		analytic := gz.AtF64(tensor.Unravel(j, tensor.Shape{b, c})...)
		if math.Abs(numeric-analytic) > rtol*math.Max(1, math.Abs(numeric)) {
			t.Errorf("zloss grad[%d]: numeric %.10g vs analytic %.10g", j, numeric, analytic)
		}
	}
}

func TestZLossValidation(t *testing.T) {
	ctx := backend.NewContext()
	if _, err := nn.ZLoss(ctx, tensor.New(tensor.F64, tensor.Shape{2, 3}), -1e-3); err == nil {
		t.Error("negative coefficient must be rejected")
	}
	if _, err := nn.ZLoss(ctx, tensor.New(tensor.F64, tensor.Shape{3}), 1e-3); err == nil {
		t.Error("rank-1 logits must be rejected")
	}
}

// ZLoss is the standalone log-Z regularizer for arbitrary logits — its primary use is the ST-MoE
// router z-loss, added alongside the load-balancing loss to keep the router's softmax normalizer
// near 0. Here it is applied to a batch of router gate logits over 3 experts.
func ExampleZLoss() {
	ctx := backend.NewContext()
	routerLogits := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{2, 0, -1, 1, 1, 0})
	l, _ := nn.ZLoss(ctx, routerLogits, 1e-3)
	fmt.Println(l.AtF64() > 0)
	// Output: true
}
