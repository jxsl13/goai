package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: the (α, β) constants match DeepNet Table 2 for each architecture.
func TestDeepNormConstants(t *testing.T) {
	// encoder-only, N=1: α=2^¼≈1.18921, β=8^{-¼}≈0.59460
	if a, b := nn.DeepNormEncoder(1); math.Abs(a-math.Pow(2, 0.25)) > 1e-12 || math.Abs(b-math.Pow(8, -0.25)) > 1e-12 {
		t.Errorf("encoder(1) = (%.8g,%.8g), want (%.8g,%.8g)", a, b, math.Pow(2, 0.25), math.Pow(8, -0.25))
	}
	// decoder-only, M=1000: α=(2000)^¼, β=(8000)^{-¼}
	if a, b := nn.DeepNormDecoder(1000); math.Abs(a-math.Pow(2000, 0.25)) > 1e-9 || math.Abs(b-math.Pow(8000, -0.25)) > 1e-12 {
		t.Errorf("decoder(1000) = (%.8g,%.8g), want (%.8g,%.8g)", a, b, math.Pow(2000, 0.25), math.Pow(8000, -0.25))
	}
	// encoder-decoder, N=12, M=12
	ae, be, ad, bd := nn.DeepNormEncoderDecoder(12, 12)
	nm := math.Pow(12, 4) * 12
	wantAe, wantBe := 0.81*math.Pow(nm, 1.0/16), 0.87*math.Pow(nm, -1.0/16)
	wantAd, wantBd := math.Pow(36, 0.25), math.Pow(144, -0.25)
	if math.Abs(ae-wantAe) > 1e-9 || math.Abs(be-wantBe) > 1e-9 || math.Abs(ad-wantAd) > 1e-9 || math.Abs(bd-wantBd) > 1e-9 {
		t.Errorf("encdec(12,12) = (%.6g,%.6g,%.6g,%.6g), want (%.6g,%.6g,%.6g,%.6g)", ae, be, ad, bd, wantAe, wantBe, wantAd, wantBd)
	}
}

// §V16 tier-1: DeepNorm.Forward(x, g) = LayerNorm(α·x + g), the defining residual.
func TestDeepNormResidual(t *testing.T) {
	const d = 4
	alpha := 1.5
	dn := nn.NewDeepNorm(tensor.F64, d, alpha)
	x := tensor.FromFloat64(tensor.Shape{2, d}, []float64{1, -2, 0.5, 3, 0.2, 1, -1, 2})
	g := tensor.FromFloat64(tensor.Shape{2, d}, []float64{0.5, 0.5, -1, 0, 1, -0.3, 0.4, 0.7})

	got, err := dn.Forward(backend.NewContext(), x, g)
	if err != nil {
		t.Fatal(err)
	}
	// reference: LayerNorm(α·x + g) using the same LN layer
	r := tensor.New(tensor.F64, x.Shape())
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		r.SetF64(alpha*x.AtF64(idx...)+g.AtF64(idx...), idx...)
	}
	want, _ := dn.LN.Forward(backend.NewContext(), r)
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		if math.Abs(got.AtF64(idx...)-want.AtF64(idx...)) > 1e-12 {
			t.Errorf("DeepNorm[%v] = %.10g, want %.10g", idx, got.AtF64(idx...), want.AtF64(idx...))
		}
	}
}

// §V16 tier-1: α=1 reduces to the standard post-LN residual LayerNorm(x + g).
func TestDeepNormAlphaOnePostLN(t *testing.T) {
	const d = 3
	dn := nn.NewDeepNorm(tensor.F64, d, 1)
	x := tensor.FromFloat64(tensor.Shape{1, d}, []float64{2, -1, 0.5})
	g := tensor.FromFloat64(tensor.Shape{1, d}, []float64{0.3, 0.6, -0.2})
	got, _ := dn.Forward(backend.NewContext(), x, g)

	sum := tensor.New(tensor.F64, x.Shape())
	for i := range d {
		sum.SetF64(x.AtF64(0, i)+g.AtF64(0, i), 0, i)
	}
	want, _ := dn.LN.Forward(backend.NewContext(), sum)
	for i := range d {
		if math.Abs(got.AtF64(0, i)-want.AtF64(0, i)) > 1e-12 {
			t.Errorf("α=1 [%d] = %.10g, want post-LN %.10g", i, got.AtF64(0, i), want.AtF64(0, i))
		}
	}
}

// §V16 tier-1: DeepNormInit is Xavier-uniform scaled by β — so with the same seed each
// value equals β·(the plain Xavier value); β=1 reproduces XavierUniform exactly.
func TestDeepNormInit(t *testing.T) {
	const fi, fo = 8, 6
	const beta, seed = 0.5, uint64(42)
	wb := tensor.New(tensor.F64, tensor.Shape{fi, fo})
	nn.DeepNormInit(wb, fi, fo, beta, seed)
	wx := tensor.New(tensor.F64, tensor.Shape{fi, fo})
	nn.XavierUniform(wx, fi, fo, seed)
	for i := range wb.Numel() {
		idx := tensor.Unravel(i, wb.Shape())
		if got, want := wb.AtF64(idx...), beta*wx.AtF64(idx...); math.Abs(got-want) > 1e-12 {
			t.Errorf("DeepNormInit[%v] = %.10g, want β·Xavier = %.10g", idx, got, want)
		}
	}
	// range is bounded by β·√(6/(fanIn+fanOut))
	bound := beta * math.Sqrt(6.0/float64(fi+fo))
	for i := range wb.Numel() {
		if v := math.Abs(wb.AtF64(tensor.Unravel(i, wb.Shape())...)); v > bound+1e-12 {
			t.Errorf("|DeepNormInit| = %.6g exceeds bound %.6g", v, bound)
		}
	}
}

func ExampleDeepNorm() {
	// A DeepNorm residual for a 12-layer decoder: x' = LayerNorm(α·x + sublayer(x)).
	alpha, _ := nn.DeepNormDecoder(12)
	dn := nn.NewDeepNorm(tensor.F64, 4, alpha)
	x := tensor.FromFloat64(tensor.Shape{1, 4}, []float64{1, 2, 3, 4})
	g := tensor.FromFloat64(tensor.Shape{1, 4}, []float64{0.1, -0.1, 0.2, -0.2}) // sublayer output
	y, _ := dn.Forward(backend.NewContext(), x, g)
	// output is unit-variance (LayerNorm) and centered
	var mean float64
	for i := range 4 {
		mean += y.AtF64(0, i)
	}
	mean /= 4
	if math.Abs(mean) < 1e-12 {
		mean = 0 // normalize ±0: amd64 libm rounds this to −1e−17, arm64 to +0
	}
	fmt.Printf("%.4f\n", mean)
	// Output:
	// 0.0000
}

func ExampleDeepNormEncoder() {
	// A 100-layer encoder up-scales the residual by α and down-scales init by β.
	alpha, beta := nn.DeepNormEncoder(100)
	fmt.Printf("alpha=%.4f beta=%.4f\n", alpha, beta)
	// Output:
	// alpha=3.7606 beta=0.1880
}
