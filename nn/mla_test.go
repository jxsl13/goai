package nn_test

import (
	"encoding/json"
	"fmt"
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

type mlaGolden struct {
	Seq, D, Heads, Dh, DR, Dc, Dcq int
	Base                           float64
	H                              []float64
	WDKV                           []float64 `json:"W_DKV"`
	WUK                            []float64 `json:"W_UK"`
	WUV                            []float64 `json:"W_UV"`
	WKR                            []float64 `json:"W_KR"`
	WDQ                            []float64 `json:"W_DQ"`
	WUQ                            []float64 `json:"W_UQ"`
	WQR                            []float64 `json:"W_QR"`
	WO                             []float64 `json:"W_O"`
	QC                             []float64 `json:"qC"`
	KC                             []float64 `json:"kC"`
	VC                             []float64 `json:"vC"`
	QRpre                          []float64 `json:"qRpre"`
	KRpre                          []float64 `json:"kRpre"`
	O                              []float64
	U                              []float64
}

func loadMLA(t *testing.T) mlaGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/mla.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g mlaGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func (g mlaGolden) opInputs() []*tensor.Tensor {
	hdh := g.Heads * g.Dh
	return []*tensor.Tensor{
		tensor.FromFloat64(tensor.Shape{g.Seq, hdh}, g.QC),
		tensor.FromFloat64(tensor.Shape{g.Seq, hdh}, g.KC),
		tensor.FromFloat64(tensor.Shape{g.Seq, hdh}, g.VC),
		tensor.FromFloat64(tensor.Shape{g.Seq, g.Heads * g.DR}, g.QRpre),
		tensor.FromFloat64(tensor.Shape{g.Seq, g.DR}, g.KRpre),
	}
}

// §V16 tier-1: the fused MLA attention (content + decoupled-RoPE score, internal
// per-head RoPE) matches the independent numpy reference at f64 1e-12.
func TestMLAOpParity(t *testing.T) {
	g := loadMLA(t)
	out, err := backend.Execute(backend.NewContext(), backend.OpMLA, g.opInputs(),
		backend.MLAAttrs{Heads: g.Heads, Causal: true})
	if err != nil {
		t.Fatal(err)
	}
	for k := range out[0].Numel() {
		idx := tensor.Unravel(k, out[0].Shape())
		if got, want := out[0].AtF64(idx...), g.O[k]; math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("mla O[%d]: got %.12g want %.12g", k, got, want)
		}
	}
}

// §V2: finite differences of Σ(MLA) w.r.t. all five inputs (qC,kC,vC and the
// pre-RoPE qR,kR — gradients pulled back through the internal rotation) match the
// analytic VJP.
func TestMLAOpGradCheck(t *testing.T) {
	g := loadMLA(t)
	ins := g.opInputs()
	attrs := backend.MLAAttrs{Heads: g.Heads, Causal: true}

	sumOut := func(xs []*tensor.Tensor) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpMLA, xs, attrs)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for k := range out[0].Numel() {
			s += out[0].AtF64(tensor.Unravel(k, out[0].Shape())...)
		}
		return s
	}
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), backend.OpMLA, ins, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	for n, in := range ins {
		grad := tape.Grad(in)
		if grad == nil {
			t.Fatalf("input %d received no gradient", n)
		}
		for k := range in.Numel() {
			idx := tensor.Unravel(k, in.Shape())
			o := in.AtF64(idx...)
			in.SetF64(o+h, idx...)
			lp := sumOut(ins)
			in.SetF64(o-h, idx...)
			lm := sumOut(ins)
			in.SetF64(o, idx...)
			if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("input %d grad[%d]: numeric %.8g vs analytic %.8g", n, k, num, ana)
			}
		}
	}
}

// §V16 tier-1: the full MLA layer (all projections + attention + output) matches
// the independent numpy reference at f64 1e-12.
func TestMLALayerForward(t *testing.T) {
	g := loadMLA(t)
	hdh := g.Heads * g.Dh
	m := &nn.MLA{
		WDKV:  tensor.FromFloat64(tensor.Shape{g.D, g.Dc}, g.WDKV),
		WUK:   tensor.FromFloat64(tensor.Shape{g.Dc, hdh}, g.WUK),
		WUV:   tensor.FromFloat64(tensor.Shape{g.Dc, hdh}, g.WUV),
		WKR:   tensor.FromFloat64(tensor.Shape{g.D, g.DR}, g.WKR),
		WDQ:   tensor.FromFloat64(tensor.Shape{g.D, g.Dcq}, g.WDQ),
		WUQ:   tensor.FromFloat64(tensor.Shape{g.Dcq, hdh}, g.WUQ),
		WQR:   tensor.FromFloat64(tensor.Shape{g.Dcq, g.Heads * g.DR}, g.WQR),
		WO:    tensor.FromFloat64(tensor.Shape{hdh, g.D}, g.WO),
		Heads: g.Heads, Dh: g.Dh, DR: g.DR, Causal: true,
	}
	u, err := m.Forward(backend.NewContext(), tensor.FromFloat64(tensor.Shape{g.Seq, g.D}, g.H))
	if err != nil {
		t.Fatal(err)
	}
	for k := range u.Numel() {
		idx := tensor.Unravel(k, u.Shape())
		if got, want := u.AtF64(idx...), g.U[k]; math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("mla u[%d]: got %.12g want %.12g", k, got, want)
		}
	}
}

// MLA's whole point: the per-token KV cache (latent dc + shared RoPE dR) is far
// smaller than MHA's (heads·2·dh).
func TestMLACacheReduction(t *testing.T) {
	m, err := nn.NewMLA(tensor.F64, 512, 8, 64, 32, 256, 384, true, 1)
	if err != nil {
		t.Fatal(err)
	}
	mla, mha := m.CacheBytesPerToken(256)
	if mla >= mha {
		t.Errorf("MLA cache %d should be smaller than MHA %d", mla, mha)
	}
	// 256+32 = 288 vs 8*2*64 = 1024 → ~72% reduction here
	if mla != 288 || mha != 1024 {
		t.Errorf("cache sizes = %d / %d, want 288 / 1024", mla, mha)
	}
}

// MLA (DeepSeek-V2) compresses the KV cache: instead of storing full per-head
// keys and values, it caches only a small shared latent plus a decoupled RoPE
// key. Here an 8-head layer caches 288 floats per token versus 1024 for standard
// MHA — a ~72% reduction (the paper's config reaches ~93%).
func ExampleMLA() {
	m, _ := nn.NewMLA(tensor.F64, 512 /*d*/, 8 /*heads*/, 64 /*dh*/, 32 /*dR*/, 256 /*dc*/, 384 /*dcq*/, true, 1)
	mla, mha := m.CacheBytesPerToken(256)
	fmt.Printf("KV cache per token: MLA %d vs MHA %d\n", mla, mha)
	// Output: KV cache per token: MLA 288 vs MHA 1024
}
