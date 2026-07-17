package nlp

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// jlensFromTensors implements the documented JLensFromPT naming assumption:
// square [d,d] tensors with trailing integer indices form the layer run;
// everything else (unembed refs, metadata tensors) is ignored. Pinned here so
// the §T812 golden verification has a single mapping seam to correct.
func TestJLensFromTensorsMapping(t *testing.T) {
	sq := func(d int, base float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{d, d})
		s := x.Storage().F64()
		for i := range s {
			s[i] = base + float64(i)
		}
		return x
	}
	ts := map[string]*tensor.Tensor{
		"jacobians.0": sq(3, 100),
		"jacobians.1": sq(3, 200),
		"jacobians.2": sq(3, 300),
		"unembed":     tensor.New(tensor.F64, tensor.Shape{3, 7}), // non-square: ignored
		"norm.gamma":  tensor.New(tensor.F64, tensor.Shape{3}),    // 1-D: ignored
	}
	jl, err := jlensFromTensors(ts)
	if err != nil {
		t.Fatal(err)
	}
	if jl.Dim != 3 || jl.Layers != 2 || len(jl.J) != 3 {
		t.Fatalf("got {dim %d, layers %d, %d matrices}", jl.Dim, jl.Layers, len(jl.J))
	}
	if jl.J[1].AtF64(0, 0) != 200 || jl.J[2].AtF64(2, 2) != 308 {
		t.Fatalf("layer order scrambled: J[1][0,0]=%g J[2][2,2]=%g", jl.J[1].AtF64(0, 0), jl.J[2].AtF64(2, 2))
	}

	// f32 artifacts are widened to the f64 the fit produces natively.
	f32 := tensor.New(tensor.F32, tensor.Shape{2, 2})
	f32.SetF64(1.5, 0, 1)
	jl, err = jlensFromTensors(map[string]*tensor.Tensor{"J.0": f32})
	if err != nil {
		t.Fatal(err)
	}
	if jl.J[0].Dtype() != tensor.F64 || jl.J[0].AtF64(0, 1) != 1.5 {
		t.Fatalf("f32 widening: dtype %v value %g", jl.J[0].Dtype(), jl.J[0].AtF64(0, 1))
	}

	// No indexed square tensors at all → the documented naming-assumption error.
	if _, err := jlensFromTensors(map[string]*tensor.Tensor{"w": sq(3, 0)}); err == nil {
		t.Fatal("expected error for un-indexed tensor map")
	}
	// A gap in the index run is rejected rather than silently misaligned.
	if _, err := jlensFromTensors(map[string]*tensor.Tensor{
		"jacobians.0": sq(3, 0), "jacobians.2": sq(3, 0),
	}); err == nil {
		t.Fatal("expected error for non-contiguous layer indices")
	}
	// Mixed dims across layers are rejected.
	if _, err := jlensFromTensors(map[string]*tensor.Tensor{
		"jacobians.0": sq(3, 0), "jacobians.1": sq(4, 0),
	}); err == nil {
		t.Fatal("expected error for inconsistent dims")
	}
}

// jlensTargets: full coverage when uncapped, evenly spaced (first and last
// included) when capped, deduplicated for tiny sequences.
func TestJLensTargets(t *testing.T) {
	cases := []struct {
		seq, max int
		want     []int
	}{
		{5, 0, []int{0, 1, 2, 3, 4}},
		{3, 8, []int{0, 1, 2}},
		{10, 3, []int{0, 4, 9}},
		{4, 1, []int{3}},
		{6, 2, []int{0, 5}},
	}
	for _, c := range cases {
		got := jlensTargets(c.seq, c.max)
		if len(got) != len(c.want) {
			t.Fatalf("jlensTargets(%d,%d) = %v, want %v", c.seq, c.max, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("jlensTargets(%d,%d) = %v, want %v", c.seq, c.max, got, c.want)
			}
		}
	}
}
