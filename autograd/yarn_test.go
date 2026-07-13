package autograd_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type yarnGolden struct {
	Q         []float64 `json:"q"`
	Seq       int       `json:"seq"`
	HD        int       `json:"hd"`
	Base      float64   `json:"base"`
	Scale     float64   `json:"scale"`
	OrigCtx   float64   `json:"orig_ctx"`
	BetaFast  float64   `json:"beta_fast"`
	BetaSlow  float64   `json:"beta_slow"`
	Low       float64   `json:"low"`
	High      float64   `json:"high"`
	Theta     []float64 `json:"theta"`
	InvFreq   []float64 `json:"inv_freq"`
	Y         []float64 `json:"y"`
	AttnScale float64   `json:"attn_scale"`
}

func loadYarn(t *testing.T) yarnGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/yarn.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g yarnGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func yarnAttrs(g yarnGolden) backend.RoPEAttrs {
	return backend.RoPEAttrs{
		Base: g.Base, YaRNScale: g.Scale, YaRNOrigCtx: g.OrigCtx,
		YaRNBetaFast: g.BetaFast, YaRNBetaSlow: g.BetaSlow,
	}
}

// §V16 tier-1: the YaRN forward matches the independent numpy paper reference at
// f64 1e-12, element for element.
func TestYaRNGoldenParity(t *testing.T) {
	g := loadYarn(t)
	q := tensor.FromFloat64(tensor.Shape{g.Seq, g.HD}, g.Q)
	out, err := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, yarnAttrs(g))
	if err != nil {
		t.Fatal(err)
	}
	for k := range out[0].Numel() {
		idx := tensor.Unravel(k, out[0].Shape())
		if got, want := out[0].AtF64(idx...), g.Y[k]; math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("yarn[%d]: got %.12g want %.12g", k, got, want)
		}
	}
}

// backend.RoPEFreqs reproduces the per-dimension NTK-by-parts blend: high-freq
// pairs (i<low) are extrapolated (θ_i unchanged), low-freq pairs (i≥high) are
// interpolated (θ_i/s), the band between is a linear ramp — matched to numpy.
func TestYaRNInvFreq(t *testing.T) {
	g := loadYarn(t)
	inv, posDiv := backend.RoPEFreqs(g.HD, yarnAttrs(g))
	if posDiv != 1 {
		t.Errorf("yarn posDiv %v, want 1 (interpolation folded into frequencies)", posDiv)
	}
	for i, want := range g.InvFreq {
		if math.Abs(inv[i]-want) > 1e-12 {
			t.Errorf("inv_freq[%d]: got %.12g want %.12g", i, inv[i], want)
		}
	}
	for i := range g.HD / 2 {
		switch {
		case float64(i) < g.Low: // extrapolated band
			if math.Abs(inv[i]-g.Theta[i]) > 1e-12 {
				t.Errorf("dim %d (i<low) must keep θ_i (extrapolate): %.12g vs %.12g", i, inv[i], g.Theta[i])
			}
		case float64(i) >= g.High: // interpolated band
			if math.Abs(inv[i]-g.Theta[i]/g.Scale) > 1e-12 {
				t.Errorf("dim %d (i≥high) must use θ_i/s (interpolate): %.12g vs %.12g", i, inv[i], g.Theta[i]/g.Scale)
			}
		}
	}
}

// §V2: the YaRN backward uses the same per-dimension angles, so the analytic VJP
// matches central finite differences of the scaled forward.
func TestYaRNGradCheck(t *testing.T) {
	g := loadYarn(t)
	attrs := yarnAttrs(g)
	seq, hd := g.Seq, g.HD
	data := make([]float64, seq*hd)
	for i := range data {
		data[i] = math.Sin(float64(i)*0.5 + 0.3)
	}
	loss := func(d []float64) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpRoPE,
			[]*tensor.Tensor{tensor.FromFloat64(tensor.Shape{seq, hd}, d)}, attrs)
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
	qT := tensor.FromFloat64(tensor.Shape{seq, hd}, data)
	out, err := backend.Execute(tape.Context(), backend.OpRoPE, []*tensor.Tensor{qT}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}
	gq := tape.Grad(qT)
	const h = 1e-6
	work := append([]float64(nil), data...)
	for k := range data {
		o := work[k]
		work[k] = o + h
		lp := loss(work)
		work[k] = o - h
		lm := loss(work)
		work[k] = o
		if num, ana := (lp-lm)/(2*h), gq.AtF64(tensor.Unravel(k, gq.Shape())...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", k, num, ana)
		}
	}
}

// The attention temperature factor m_scale = 0.1·ln(s)+1 (Peng et al. 2023 §3.3);
// s≤1 disables it.
func TestYaRNAttnScale(t *testing.T) {
	g := loadYarn(t)
	if got := backend.YaRNAttnScale(g.Scale); math.Abs(got-g.AttnScale) > 1e-12 {
		t.Errorf("attn scale got %.12g want %.12g", got, g.AttnScale)
	}
	if backend.YaRNAttnScale(1) != 1 || backend.YaRNAttnScale(0.5) != 1 {
		t.Error("attn scale must be 1 for s≤1")
	}
}

// YaRN (Peng et al. 2023) extends a RoPE model's context by "NTK-by-parts"
// interpolation: high-frequency dimensions (short wavelength, needed for local
// position) are left untouched, while low-frequency dimensions (long wavelength)
// are interpolated to keep their phases in the trained range. Here a model trained
// at context 2048 is extended 4×: the highest-frequency pair is extrapolated, so
// it is byte-identical to plain RoPE, and the accompanying attention temperature
// is 0.1·ln(4)+1.
func Example_ropeYaRN() {
	q := tensor.New(tensor.F64, tensor.Shape{4, 16})
	for p := range 4 {
		for j := range 16 {
			q.SetF64(1, p, j)
		}
	}
	attrs := backend.RoPEAttrs{YaRNScale: 4.0, YaRNOrigCtx: 2048.0}
	yarn, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, attrs)
	plain, _ := backend.Execute(backend.NewContext(), backend.OpRoPE, []*tensor.Tensor{q}, nil)

	same := math.Abs(yarn[0].AtF64(1, 0)-plain[0].AtF64(1, 0)) < 1e-12
	fmt.Println("high-freq dim extrapolated:", same)
	fmt.Printf("attention temperature: %.4f\n", backend.YaRNAttnScale(4))
	// Output:
	// high-freq dim extrapolated: true
	// attention temperature: 1.1386
}
