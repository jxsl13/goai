package nn

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// muonShapeSet pairs a parameter set with the digest of the parameters after three Muon steps.
//
// THE SMALL SET IS THE SERIAL ARM AND THE LARGE ONE ACTUALLY BANDS. The first version of this
// test used only the small shapes, whose total element count is under parallelRows' 1<<14
// threshold, so both arms ran serially and a mutation letting each band start one parameter
// early passed. The large set clears the gate.
//
// Three parameters, not two: with two, a band split that hands every parameter to its own
// worker looks the same as a correct one. Each set mixes shapes, and its second entry is taller
// than it is wide so newtonSchulz5 takes its transpose branch.
type muonShapeSet struct {
	name      string
	shapes    []tensor.Shape
	f64, f32d uint64
}

var muonSets = []muonShapeSet{
	{"below-gate", []tensor.Shape{{24, 40}, {40, 24}, {32, 32}}, archgold.Pick(9241251877628356013, 15826934832090508788), archgold.Pick(14877648412914339811, 14877648412914339811)},
	{"banded", []tensor.Shape{{72, 80}, {80, 72}, {76, 76}}, archgold.Pick(4743616716528603079, 7910985114552742753), archgold.Pick(12900181287849649047, 12900181287849649047)},
}

// TestMuonStepIsBitIdentical freezes the parameters after three Muon steps. Banding the
// parameter loop claims to change no value — each parameter reads its own gradient and writes
// its own buffers, and the Newton-Schulz iteration inside it is untouched — and an optimizer is
// the last place a small numeric drift would be noticed, since training absorbs it.
func TestMuonStepIsBitIdentical(t *testing.T) {
	for _, set := range muonSets {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			want := set.f64
			if dt == tensor.F32 {
				want = set.f32d
			}
			params := make([]*tensor.Tensor, len(set.shapes))
			grads := map[*tensor.Tensor]*tensor.Tensor{}
			for i, s := range set.shapes {
				params[i] = tensor.New(dt, s)
				g := tensor.New(dt, s)
				for r := range s[0] {
					for c := range s[1] {
						params[i].SetF64(math.Sin(float64(i*13+r*3+c))*0.4, r, c)
						g.SetF64(math.Cos(float64(i*7+r*5+c))*0.1, r, c)
					}
				}
				grads[params[i]] = g
			}
			opt := NewMuon(params, 0.02)
			for range 3 {
				err := opt.Step(func(p *tensor.Tensor) *tensor.Tensor { return grads[p] })
				if err != nil {
					t.Fatal(err)
				}
			}
			h := uint64(14695981039346656037)
			for _, p := range params {
				for i := range p.Numel() {
					idx := tensor.Unravel(i, p.Shape())
					b := math.Float64bits(p.AtF64(idx...))
					for s := 0; s < 64; s += 8 {
						h = (h ^ (b>>s)&0xff) * 1099511628211
					}
				}
			}
			if h != want {
				t.Fatalf("%s %v digest %d, want %d", set.name, dt, h, want)
			}
		}
	}
}
