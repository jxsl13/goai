package cpu

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/archgold"
	"github.com/jxsl13/goai/tensor"
)

// wkvOpDigest binds the actual executed inputs to the frozen dyadic fixture.
// Output metadata, finiteness, and unchanged inputs are checked before hashing.
func wkvOpDigest(t *testing.T, be backend.Name, in []*tensor.Tensor, wantInputSHA string) uint64 {
	t.Helper()
	if got := wkvInputSHA256(in); got != wantInputSHA {
		t.Fatalf("%v frozen input SHA: got %s want %s", be, got, wantInputSHA)
	}
	out := executeWKVDyadic(t, be, in)
	requireWKVFinite(t, be, out)
	if got := wkvInputSHA256(in); got != wantInputSHA {
		t.Fatalf("%v mutated frozen inputs: got %s want %s", be, got, wantInputSHA)
	}
	return wkvOutputDigest(out)
}

// Exact dyadic goldens were harvested with Go 1.27.1 in native CI run
// https://github.com/jxsl13/goai/actions/runs/34560190514 from source head
// b9867e2ba8c0b4fade00e91a74d308301c1729fb. Linux and Windows agree in both
// build modes; native macOS agrees with an independent M2 run. The CI merge
// 2032c7774b090a2668613515e1c1c68b08f7f809 has the identical source tree.
// No golden was obtained from emulation or a transcendental input fixture.
func TestWKVOpIsBitIdentical(t *testing.T) {
	if !archgold.Supported() {
		t.Skip(archgold.Reason)
	}
	// 37 channels exercises remainder bands; 96 exercises complete groups.
	cases := []struct {
		dt               tensor.Dtype
		seq, d           int
		inputSHA         string
		wantRef, wantCPU uint64
	}{
		{tensor.F32, 24, 37,
			"237d57a777afaf54dc842cb4ebca902d5dcd44a769f663c3223825500e293ab4",
			archgold.Pick(12746545039049088356, 12746545039049088356),
			archgold.PickSIMD(12746545039049088356, 12746545039049088356, 12746545039049088356, 12746545039049088356)},
		{tensor.F64, 24, 37,
			"66b4821f3b049d918411fbffab0511221b67481d75c7a547c73fb6b1448ae0e3",
			archgold.Pick(17367948250675808524, 6028244063516659357),
			archgold.PickSIMD(17367948250675808524, 6028244063516659357, 4909970916033361732, 7419203717802686153)},
		{tensor.F32, 64, 96,
			"7e11d15d2d3c8691a747c9bf2b37ee7f6ee33d3e34ed0399b13b2a485cd81fa7",
			archgold.Pick(3640097571888084831, 3640097571888084831),
			archgold.PickSIMD(3640097571888084831, 3640097571888084831, 3640097571888084831, 3640097571888084831)},
		{tensor.F64, 64, 96,
			"673f8d6b748f3a7290fa9cf02a7dba18fd0c11d55a16e1110f32362b3cf50721",
			archgold.Pick(18162273878488981657, 15771226142186551746),
			archgold.PickSIMD(18162273878488981657, 15771226142186551746, 7543576894518255023, 14264126157401123388)},
	}
	for _, c := range cases {
		for _, be := range []backend.Name{backend.Ref, backend.CPU} {
			t.Run(fmt.Sprintf("%s/%s/%dx%d", be, c.dt, c.seq, c.d), func(t *testing.T) {
				want := c.wantRef
				if be == backend.CPU {
					want = c.wantCPU
				}
				got := wkvOpDigest(t, be, wkvDyadicInputs(c.dt, c.seq, c.d), c.inputSHA)
				if got != want {
					t.Fatalf("output digest: got %d want %d", got, want)
				}
			})
		}
	}
}

// Preserve the old Sin/Cos vectors as reference-accuracy coverage. Their input
// bits are architecture-dependent, so they are deliberately not golden fixtures.
func wkvLegacyInputs(dt tensor.Dtype, seq, d int) []*tensor.Tensor {
	mk := func(shape tensor.Shape, fn func(i int) float64) *tensor.Tensor {
		x := tensor.New(dt, shape)
		n := x.Numel()
		if dt == tensor.F64 {
			s := x.Storage().F64()
			for i := range n {
				s[i] = fn(i)
			}
		} else {
			s := x.Storage().F32()
			for i := range n {
				s[i] = float32(fn(i))
			}
		}
		return x
	}
	k := mk(tensor.Shape{seq, d}, func(i int) float64 { return math.Sin(float64(i)*0.37) * 3 })
	v := mk(tensor.Shape{seq, d}, func(i int) float64 { return math.Cos(float64(i) * 0.21) })
	w := mk(tensor.Shape{d}, func(i int) float64 { return 0.5 + 0.01*float64(i%7) })
	u := mk(tensor.Shape{d}, func(i int) float64 { return -0.25 + 0.02*float64(i%5) })
	return []*tensor.Tensor{k, v, w, u}
}

func TestWKVLegacyFixtureReferenceParity(t *testing.T) {
	for _, shape := range [][2]int{{24, 37}, {64, 96}} {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			t.Run(fmt.Sprintf("%s/%dx%d", dt, shape[0], shape[1]), func(t *testing.T) {
				var outputs [2]*tensor.Tensor
				var inputSHA string
				for i, be := range []backend.Name{backend.Ref, backend.CPU} {
					in := wkvLegacyInputs(dt, shape[0], shape[1])
					before := wkvInputSHA256(in)
					if i == 0 {
						inputSHA = before
					} else if before != inputSHA {
						t.Fatal("legacy CPU/Ref input bits differ")
					}
					outputs[i] = executeWKVDyadic(t, be, in)
					requireWKVFinite(t, be, outputs[i])
					if wkvInputSHA256(in) != before {
						t.Fatalf("%v mutated legacy inputs", be)
					}
				}
				if wkvSIMDExperimentEnabled() && dt == tensor.F64 {
					requireWKVF64Relative(t, outputs[1], outputs[0])
				} else {
					requireWKVExact(t, "legacy CPU vs Ref", outputs[1], outputs[0])
				}
			})
		}
	}
}
