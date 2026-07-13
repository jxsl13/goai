package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// §V16 tier-1: the μP scaling rules match Table 8 exactly, relative to the base width —
// hidden & readout Adam LR and the readout multiplier all scale as 1/m (m = width/base),
// input LR is constant 1, and the attention scale is 1/head_dim (not 1/√head_dim).
func TestMuPScalingConstants(t *testing.T) {
	p := nn.NewMuP(128)
	cases := []struct {
		width int
		invM  float64 // 1/m
	}{{128, 1}, {256, 0.5}, {512, 0.25}, {64, 2}}
	for _, c := range cases {
		if got := p.HiddenLRScale(c.width); math.Abs(got-c.invM) > 1e-12 {
			t.Errorf("HiddenLRScale(%d) = %g, want %g", c.width, got, c.invM)
		}
		if got := p.ReadoutLRScale(c.width); math.Abs(got-c.invM) > 1e-12 {
			t.Errorf("ReadoutLRScale(%d) = %g, want %g", c.width, got, c.invM)
		}
		if got := p.ReadoutMult(c.width); math.Abs(got-c.invM) > 1e-12 {
			t.Errorf("ReadoutMult(%d) = %g, want %g", c.width, got, c.invM)
		}
		if got := p.WidthMult(c.width); math.Abs(got-1/c.invM) > 1e-12 {
			t.Errorf("WidthMult(%d) = %g, want %g", c.width, got, 1/c.invM)
		}
	}
	// input/embedding LR is width-independent
	if p.InputLRScale() != 1 {
		t.Errorf("InputLRScale = %g, want 1", p.InputLRScale())
	}
	// attention scale is 1/d, distinct from the standard 1/√d
	if got, want := p.AttentionScale(64), 1.0/64; math.Abs(got-want) > 1e-12 {
		t.Errorf("AttentionScale(64) = %g, want %g", got, want)
	}
	if p.AttentionScale(64) == nn.StdAttentionScale(64) {
		t.Error("μP attention scale 1/d must differ from standard 1/√d")
	}
	if math.Abs(nn.StdAttentionScale(64)-1.0/8) > 1e-12 {
		t.Errorf("StdAttentionScale(64) = %g, want 1/8", nn.StdAttentionScale(64))
	}
}

// §V16 tier-1 (the defining μP-vs-SP contrast): the readout learning-rate and multiplier
// SHRINK with width under μP (1/m), whereas Standard Parametrization keeps them constant
// — this is exactly what makes hyperparameters transfer across widths.
func TestMuPSPContrast(t *testing.T) {
	p := nn.NewMuP(64)
	// scaling up the width 8× shrinks the readout LR/mult 8× under μP...
	if got := p.ReadoutLRScale(512); math.Abs(got-1.0/8) > 1e-12 {
		t.Errorf("μP ReadoutLRScale(8×base) = %g, want 1/8", got)
	}
	// ...while SP (constant 1) does not scale at all — so the μP factor is strictly < 1.
	if p.ReadoutLRScale(512) >= 1 {
		t.Error("μP readout LR must shrink with width (SP would stay 1)")
	}
	if p.ReadoutMult(512) >= 1 {
		t.Error("μP readout multiplier must shrink with width (SP would stay 1)")
	}
}

func ExampleMuP() {
	// Base proxy width 256; at a 4× wider target the hidden/readout LR and the readout
	// multiplier are all scaled by 1/4, and attention uses 1/head_dim.
	p := nn.NewMuP(256)
	fmt.Printf("hiddenLR=%.2f readoutMult=%.2f attn(64)=%.5f\n",
		p.HiddenLRScale(1024), p.ReadoutMult(1024), p.AttentionScale(64))
	// Output:
	// hiddenLR=0.25 readoutMult=0.25 attn(64)=0.01562
}
