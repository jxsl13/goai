package nlp

import (
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// jlensFromTensors implements the JLensFromPT mapping, pinned by the §T812
// golden fixture against the real reference artifact: keys "J.N" (the .pt's
// int-keyed "J" dict flattened by format/pytorch) carry block-output-indexed
// Jacobians — reference layer N = GoAI layer N+1 — so K matrices yield
// Layers = K+1 with J[0] nil (never fitted by the reference) and J[Layers]
// the synthesized identity. Non-square and un-indexed tensors are ignored.
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
		"J.0":        sq(3, 100),
		"J.1":        sq(3, 200),
		"J.2":        sq(3, 300),
		"unembed":    tensor.New(tensor.F64, tensor.Shape{3, 7}), // non-square: ignored
		"norm.gamma": tensor.New(tensor.F64, tensor.Shape{3}),    // 1-D: ignored
	}
	jl, err := jlensFromTensors(ts)
	if err != nil {
		t.Fatal(err)
	}
	// 3 reference matrices → a 4-block model: Layers 4, J[0] nil, J[4] = I.
	if jl.Dim != 3 || jl.Layers != 4 || len(jl.J) != 5 {
		t.Fatalf("got {dim %d, layers %d, %d matrices}", jl.Dim, jl.Layers, len(jl.J))
	}
	if jl.J[0] != nil {
		t.Fatal("J[0] must be nil: the reference format has no post-embedding Jacobian")
	}
	if jl.J[1].AtF64(0, 0) != 100 || jl.J[2].AtF64(0, 0) != 200 || jl.J[3].AtF64(2, 2) != 308 {
		t.Fatalf("layer shift wrong: J[1][0,0]=%g J[2][0,0]=%g J[3][2,2]=%g",
			jl.J[1].AtF64(0, 0), jl.J[2].AtF64(0, 0), jl.J[3].AtF64(2, 2))
	}
	for a := range 3 {
		for b := range 3 {
			want := 0.0
			if a == b {
				want = 1
			}
			if jl.J[4].AtF64(a, b) != want {
				t.Fatalf("J[Layers] not identity at [%d,%d]", a, b)
			}
		}
	}

	// f16/f32 artifacts (the reference saves fp16 by default) are widened to
	// the f64 the fit produces natively.
	f32 := tensor.New(tensor.F32, tensor.Shape{2, 2})
	f32.SetF64(1.5, 0, 1)
	jl, err = jlensFromTensors(map[string]*tensor.Tensor{"J.0": f32})
	if err != nil {
		t.Fatal(err)
	}
	if jl.J[1].Dtype() != tensor.F64 || jl.J[1].AtF64(0, 1) != 1.5 {
		t.Fatalf("f32 widening: dtype %v value %g", jl.J[1].Dtype(), jl.J[1].AtF64(0, 1))
	}

	// No indexed square tensors at all → the documented format error.
	if _, err := jlensFromTensors(map[string]*tensor.Tensor{"w": sq(3, 0)}); err == nil {
		t.Fatal("expected error for un-indexed tensor map")
	}
	// A gap in the index run (non-default source_layers fit) is rejected
	// rather than silently misaligned.
	if _, err := jlensFromTensors(map[string]*tensor.Tensor{
		"J.0": sq(3, 0), "J.2": sq(3, 0),
	}); err == nil {
		t.Fatal("expected error for non-contiguous layer indices")
	}
	// Mixed dims across layers are rejected.
	if _, err := jlensFromTensors(map[string]*tensor.Tensor{
		"J.0": sq(3, 0), "J.1": sq(4, 0),
	}); err == nil {
		t.Fatal("expected error for inconsistent dims")
	}
}

// jlensTargets: full coverage when uncapped, evenly spaced (first and last
// index included) when capped, deduplicated for tiny valid-position lists.
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
