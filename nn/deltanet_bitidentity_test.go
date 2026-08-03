package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestDeltaNetFamilyIsBitIdentical freezes both delta-rule recurrences. Merging their per-step
// passes over the state claims to change no value — each row's stages already ran in that order
// relative to each other — and a linear-attention variant is somewhere a small drift would be
// blamed on the normalization and never chased.
//
// The shapes bracket the state size, which is what the merge acts on: 32x32 is 8 KB and lives
// in L1 whatever you do, 128x128 is 131 KB and does not survive a pass.
func TestDeltaNetFamilyIsBitIdentical(t *testing.T) {
	cases := []struct {
		seq, dk, dv    int
		wantDN, wantGD uint64
	}{
		{8, 4, 4, 10396212524795270898, 8262904611513938596},
		{64, 32, 32, 1711795498884816630, 5199788972169960644},
		{48, 128, 128, 5532324204454043268, 17362182918204471820},
	}
	ctx := backend.NewContext() // Recorder == nil: the fused inference path
	for _, c := range cases {
		mk := func(fn func(i int) float64, r, cc int) *tensor.Tensor {
			tn := tensor.New(tensor.F64, tensor.Shape{r, cc})
			s := tn.Storage().F64()
			for i := range s {
				s[i] = fn(i)
			}
			return tn
		}
		q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.011) }, c.seq, c.dk)
		k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.017) }, c.seq, c.dk)
		v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.023) }, c.seq, c.dv)
		beta := mk(func(i int) float64 { return 0.4 + 0.3*math.Cos(float64(i)) }, c.seq, 1)
		alpha := mk(func(i int) float64 { return 0.9 + 0.05*math.Sin(float64(i)) }, c.seq, 1)
		digest := func(out *tensor.Tensor) uint64 {
			h := uint64(14695981039346656037)
			for _, x := range out.Storage().F64() {
				b := math.Float64bits(x)
				for s := 0; s < 64; s += 8 {
					h = (h ^ (b>>s)&0xff) * 1099511628211
				}
			}
			return h
		}
		dn, err := nn.DeltaNet(ctx, q, k, v, beta)
		if err != nil {
			t.Fatal(err)
		}
		gd, err := nn.GatedDeltaNet(ctx, q, k, v, alpha, beta)
		if err != nil {
			t.Fatal(err)
		}
		hdn, hgd := digest(dn), digest(gd)
		if hdn != c.wantDN {
			t.Fatalf("DeltaNet %dx%dx%d digest %d, want %d", c.seq, c.dk, c.dv, hdn, c.wantDN)
		}
		if hgd != c.wantGD {
			t.Fatalf("GatedDeltaNet %dx%dx%d digest %d, want %d", c.seq, c.dk, c.dv, hgd, c.wantGD)
		}
	}
}
