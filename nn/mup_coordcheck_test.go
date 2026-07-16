package nn_test

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// coordCfg is the shared coordinate-check configuration for the tests below: a 2-hidden-layer
// MLP swept over a 16× width range, trained a few Adam steps. Under μP every layer's coordinate
// scale stays width-invariant; under SP the hidden coordinate blows up with width.
func coordCfg(seed uint64) nn.CoordCheck {
	return nn.CoordCheck{
		Widths: []int{64, 128, 256, 512, 1024}, BaseWidth: 64,
		Depth: 2, InDim: 8, Batch: 16, Steps: 20, LR: 0.01, Seed: seed,
	}
}

func coordMaxRatio(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	return m
}

// §V16 tier-1 (the μP anchor — width invariance): under correct μP the per-layer activation
// coordinate RMS is ~width-invariant across a 16× width sweep, so the max/min ratio of every
// layer stays bounded by a small constant, and the OUTPUT logit RMS (μP's headline quantity)
// is essentially constant. Verified across several seeds so it is not a lucky draw.
func TestCoordCheckMuPWidthInvariant(t *testing.T) {
	for _, seed := range []uint64{1, 2, 3, 7, 42} {
		muP, err := coordCfg(seed).Run(true)
		if err != nil {
			t.Fatalf("seed %d: Run(μP): %v", seed, err)
		}
		ratio, err := nn.CoordCheckStable(muP)
		if err != nil {
			t.Fatalf("seed %d: CoordCheckStable: %v", seed, err)
		}
		if len(ratio) != 3 { // Depth hidden pre-activations + 1 output
			t.Fatalf("seed %d: got %d layer ratios, want 3", seed, len(ratio))
		}
		for l, r := range ratio {
			if r >= 3 {
				t.Errorf("seed %d: μP layer %d width-ratio %.3g ≥ 3 (should be ~1, width-invariant)", seed, l, r)
			}
		}
		if out := ratio[len(ratio)-1]; out >= 2 {
			t.Errorf("seed %d: μP output-logit width-ratio %.3g ≥ 2 (μP's headline: it should be ~constant)", seed, out)
		}
	}
}

// §V16 tier-1 (the contrast that makes the check meaningful — SP diverges): running the SAME
// sweep WITHOUT μP scaling (Standard Parametrization) makes at least one layer's coordinate
// RMS grow strongly with width, so SP's worst-layer width-ratio is far larger than μP's — the
// signature that μP is doing real work. Asserts SP ratio > 3× the μP ratio.
func TestCoordCheckSPDiverges(t *testing.T) {
	for _, seed := range []uint64{1, 2, 3, 7, 42} {
		c := coordCfg(seed)
		muP, err := c.Run(true)
		if err != nil {
			t.Fatalf("seed %d: Run(μP): %v", seed, err)
		}
		sp, err := c.Run(false)
		if err != nil {
			t.Fatalf("seed %d: Run(SP): %v", seed, err)
		}
		muRatio, err := nn.CoordCheckStable(muP)
		if err != nil {
			t.Fatalf("seed %d: stable(μP): %v", seed, err)
		}
		spRatio, err := nn.CoordCheckStable(sp)
		if err != nil {
			t.Fatalf("seed %d: stable(SP): %v", seed, err)
		}
		muMax, spMax := coordMaxRatio(muRatio), coordMaxRatio(spRatio)
		t.Logf("seed %d: μP worst-layer ratio %.3g (%v)  SP %.3g (%v)", seed, muMax, fmtRatio(muRatio), spMax, fmtRatio(spRatio))
		if spMax <= 3*muMax {
			t.Errorf("seed %d: SP worst-layer ratio %.3g not > 3× μP %.3g — μP should be far more width-stable", seed, spMax, muMax)
		}
	}
}

// §V16 (determinism, §V13): the same CoordCheck value produces a bit-identical table on every
// run — both the μP and the SP sweep.
func TestCoordCheckDeterministic(t *testing.T) {
	c := coordCfg(1)
	for _, muP := range []bool{true, false} {
		a, err := c.Run(muP)
		if err != nil {
			t.Fatal(err)
		}
		b, err := c.Run(muP)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(a.RMS, b.RMS) {
			t.Errorf("muP=%v: repeated Run gave different tables:\n a=%v\n b=%v", muP, a.RMS, b.RMS)
		}
		if !reflect.DeepEqual(a.Widths, c.Widths) {
			t.Errorf("muP=%v: table widths %v != config %v", muP, a.Widths, c.Widths)
		}
	}
}

// §V16 (μP rules are actually applied): the μP and SP sweeps must NOT be identical — if the
// [nn.MuP] readout multiplier and per-group learning rates were ignored, both runs would return
// the same table. The distinct, more-stable μP result is evidence the mup.go rules are wired in.
func TestCoordCheckMuPActuallyApplied(t *testing.T) {
	c := coordCfg(1)
	muP, _ := c.Run(true)
	sp, _ := c.Run(false)
	if reflect.DeepEqual(muP.RMS, sp.RMS) {
		t.Error("μP and SP tables are identical — the MuP scaling is not being applied")
	}
}

// §V16 (sanity / error paths): the check rejects degenerate configurations, and CoordCheckStable
// rejects tables it cannot form a ratio from.
func TestCoordCheckSanity(t *testing.T) {
	base := coordCfg(1)
	bad := []struct {
		name string
		mut  func(*nn.CoordCheck)
	}{
		{"one width", func(c *nn.CoordCheck) { c.Widths = []int{64} }},
		{"zero widths", func(c *nn.CoordCheck) { c.Widths = nil }},
		{"non-positive width", func(c *nn.CoordCheck) { c.Widths = []int{64, 0} }},
		{"repeated width", func(c *nn.CoordCheck) { c.Widths = []int{64, 64} }},
		{"bad base", func(c *nn.CoordCheck) { c.BaseWidth = 0 }},
		{"bad depth", func(c *nn.CoordCheck) { c.Depth = 0 }},
		{"bad indim", func(c *nn.CoordCheck) { c.InDim = 0 }},
		{"bad batch", func(c *nn.CoordCheck) { c.Batch = 0 }},
		{"negative steps", func(c *nn.CoordCheck) { c.Steps = -1 }},
		{"bad lr", func(c *nn.CoordCheck) { c.LR = 0 }},
	}
	for _, tc := range bad {
		c := base
		tc.mut(&c)
		if _, err := c.Run(true); err == nil {
			t.Errorf("%s: expected error, got nil", tc.name)
		}
	}

	if _, err := nn.CoordCheckStable(nil); err == nil {
		t.Error("CoordCheckStable(nil): expected error")
	}
	if _, err := nn.CoordCheckStable(&nn.CoordTable{}); err == nil {
		t.Error("CoordCheckStable(empty): expected error")
	}
	if _, err := nn.CoordCheckStable(&nn.CoordTable{RMS: [][]float64{{1, 2}}}); err == nil {
		t.Error("CoordCheckStable(one width): expected error")
	}
	if _, err := nn.CoordCheckStable(&nn.CoordTable{RMS: [][]float64{{1}, {0}}}); err == nil {
		t.Error("CoordCheckStable(non-positive RMS): expected error")
	}
	if _, err := nn.CoordCheckStable(&nn.CoordTable{RMS: [][]float64{{1, 2}, {1}}}); err == nil {
		t.Error("CoordCheckStable(ragged): expected error")
	}
	// A well-formed table yields one ratio per layer (per-column max/min).
	good, err := nn.CoordCheckStable(&nn.CoordTable{RMS: [][]float64{{2, 2}, {1, 8}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(good) != 2 || good[0] != 2 || good[1] != 4 { // col0 max2/min1=2, col1 max8/min2=4
		t.Errorf("ratios = %v, want [2 4]", good)
	}
}

func fmtRatio(v []float64) string {
	s := "["
	for i, x := range v {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("%.2g", x)
	}
	return s + "]"
}

// ExampleCoordCheck runs the μP coordinate check and prints the verdict: μP keeps every layer's
// coordinate scale width-invariant (small ratio across the width sweep), while Standard
// Parametrization lets a coordinate blow up — proving the μP scaling in nn.MuP works.
func ExampleCoordCheck() {
	c := nn.CoordCheck{
		Widths: []int{64, 128, 256, 512, 1024}, BaseWidth: 64,
		Depth: 2, InDim: 8, Batch: 16, Steps: 20, LR: 0.01, Seed: 1,
	}
	muP, _ := c.Run(true) // Maximal-Update Parametrization
	sp, _ := c.Run(false) // Standard Parametrization (the contrast)
	muRatio, _ := nn.CoordCheckStable(muP)
	spRatio, _ := nn.CoordCheckStable(sp)

	muMax, spMax := 0.0, 0.0
	for _, r := range muRatio {
		if r > muMax {
			muMax = r
		}
	}
	for _, r := range spRatio {
		if r > spMax {
			spMax = r
		}
	}
	fmt.Printf("μP width-stable (every layer ratio < 3): %v\n", muMax < 3)
	fmt.Printf("SP diverges (worst layer > 3× μP): %v\n", spMax > 3*muMax)
	// Output:
	// μP width-stable (every layer ratio < 3): true
	// SP diverges (worst layer > 3× μP): true
}
