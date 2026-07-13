package ops_test

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func flatE(t *tensor.Tensor) []float64 {
	d := make([]float64, t.Numel())
	for i := range d {
		d[i] = t.AtF64(tensor.Unravel(i, t.Shape())...)
	}
	return d
}

// hand-computed einsum patterns: matmul, transpose, trace, outer, full sum.
func TestEinsumHand(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	b := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 0, 0, 1, 1, 1})
	mm, err := ops.Einsum("ij,jk->ik", a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !mm.Shape().Equal(tensor.Shape{2, 2}) {
		t.Fatalf("matmul shape %v", mm.Shape())
	}
	for i, w := range []float64{4, 5, 10, 11} {
		if flatE(mm)[i] != w {
			t.Errorf("matmul[%d]=%g want %g", i, flatE(mm)[i], w)
		}
	}

	tr, _ := ops.Einsum("ij->ji", a)
	if !tr.Shape().Equal(tensor.Shape{3, 2}) || tr.AtF64(2, 1) != 6 {
		t.Errorf("transpose wrong: shape %v [2,1]=%g", tr.Shape(), tr.AtF64(2, 1))
	}

	sq := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	trace, _ := ops.Einsum("ii->", sq)
	if trace.AtF64() != 5 {
		t.Errorf("trace = %g, want 5", trace.AtF64())
	}

	outer, _ := ops.Einsum("i,j->ij", tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2}), tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4}))
	for i, w := range []float64{3, 4, 6, 8} {
		if flatE(outer)[i] != w {
			t.Errorf("outer[%d]=%g want %g", i, flatE(outer)[i], w)
		}
	}

	sum, _ := ops.Einsum("ij->", a)
	if sum.AtF64() != 21 {
		t.Errorf("sum = %g, want 21", sum.AtF64())
	}
	rowsum, _ := ops.Einsum("ij->i", a)
	if rowsum.AtF64(0) != 6 || rowsum.AtF64(1) != 15 {
		t.Errorf("rowsum wrong: %v", flatE(rowsum))
	}
	// implicit output (indices appearing once, sorted) → matmul 'ik'
	imp, _ := ops.Einsum("ij,jk", a, b)
	if !imp.Shape().Equal(tensor.Shape{2, 2}) || flatE(imp)[3] != 11 {
		t.Errorf("implicit output wrong: shape %v", imp.Shape())
	}
}

// §V16 definitional: a range of contractions match the reference numpy.einsum.
// Skips without the .venv.
func TestEinsumVsNumpy(t *testing.T) {
	py := "../.venv/bin/python"
	if _, err := os.Stat(py); err != nil {
		t.Skip("no .venv python")
	}
	a := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1.5, -2, 3, 0.5, 4, -1})
	bt := tensor.FromFloat64(tensor.Shape{2, 3, 2}, []float64{1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6})
	v := tensor.FromFloat64(tensor.Shape{3}, []float64{2, -1, 0.5})
	cases := []struct {
		spec string
		ops  []*tensor.Tensor
		np   string // numpy operand expression
	}{
		{"ij,j->i", []*tensor.Tensor{a, v}, "A,V"},
		{"ij,ik->jk", []*tensor.Tensor{a, a}, "A,A"},
		{"bij->bji", []*tensor.Tensor{bt}, "B"},
		{"bij,bjk->bik", []*tensor.Tensor{bt, bt}, "B,B.transpose(0,2,1)@0+B"}, // handled below
	}
	pre := "import numpy as np;" +
		"A=np.array([1.5,-2,3,0.5,4,-1]).reshape(2,3);" +
		"V=np.array([2,-1,0.5]);" +
		"B=np.array([1,2,3,4,5,6,-1,-2,-3,-4,-5,-6]).reshape(2,3,2);"
	// only run the first three (self-contained numpy exprs)
	for _, c := range cases[:3] {
		got, err := ops.Einsum(c.spec, c.ops...)
		if err != nil {
			t.Fatalf("%s: %v", c.spec, err)
		}
		script := pre + fmt.Sprintf("print(np.einsum('%s',%s).flatten().tolist())", c.spec, c.np)
		out, err := exec.Command(py, "-c", script).CombinedOutput()
		if err != nil {
			t.Fatalf("numpy %s: %v\n%s", c.spec, err, out)
		}
		want := parseFloats(t, strings.TrimSpace(string(out)))
		gf := flatE(got)
		if len(gf) != len(want) {
			t.Fatalf("%s: len %d vs numpy %d", c.spec, len(gf), len(want))
		}
		for i := range gf {
			if math.Abs(gf[i]-want[i]) > 1e-10 {
				t.Errorf("%s [%d]: %g vs numpy %g", c.spec, i, gf[i], want[i])
			}
		}
	}
	// batched matmul B[2,3,2]·... use 'bij,bjk->bik' with a compatible second operand
	c := tensor.FromFloat64(tensor.Shape{2, 2, 4}, make([]float64, 16))
	for i := range 16 {
		c.SetF64(float64(i)-8, tensor.Unravel(i, c.Shape())...)
	}
	got, _ := ops.Einsum("bij,bjk->bik", bt, c)
	script := pre + "C=np.arange(16).reshape(2,2,4)-8;" +
		"print(np.einsum('bij,bjk->bik',B,C).flatten().tolist())"
	out, _ := exec.Command(py, "-c", script).CombinedOutput()
	want := parseFloats(t, strings.TrimSpace(string(out)))
	for i, gv := range flatE(got) {
		if math.Abs(gv-want[i]) > 1e-9 {
			t.Errorf("bmm[%d]: %g vs numpy %g", i, gv, want[i])
		}
	}
}

func parseFloats(t *testing.T, s string) []float64 {
	t.Helper()
	s = strings.Trim(s, "[]")
	if s == "" {
		return nil
	}
	var out []float64
	for _, p := range strings.Split(s, ",") {
		var f float64
		if _, err := fmt.Sscanf(strings.TrimSpace(p), "%g", &f); err != nil {
			t.Fatalf("parse %q: %v", p, err)
		}
		out = append(out, f)
	}
	return out
}

func TestEinsumErrors(t *testing.T) {
	a := tensor.New(tensor.F64, tensor.Shape{2, 3})
	cases := []struct {
		spec string
		ops  []*tensor.Tensor
	}{
		{"ij,jk->ik", []*tensor.Tensor{a}},                                             // operand count mismatch
		{"ij", []*tensor.Tensor{tensor.New(tensor.F64, tensor.Shape{2, 3, 4})}},        // subscript len != ndim
		{"ij,jk->il", []*tensor.Tensor{a, tensor.New(tensor.F64, tensor.Shape{3, 4})}}, // output index 'l' absent
		{"ij...->i", []*tensor.Tensor{a}},                                              // ellipsis
		{"iJ->i", []*tensor.Tensor{a}},                                                 // non-lowercase
	}
	for i, c := range cases {
		if _, err := ops.Einsum(c.spec, c.ops...); err == nil {
			t.Errorf("case %d (%q): expected error", i, c.spec)
		}
	}
	// inconsistent index size: 'ii' on a non-square matrix
	if _, err := ops.Einsum("ii->", a); err == nil {
		t.Error("non-square trace must error (inconsistent index size)")
	}
}

// Einsum evaluates Einstein-summation contractions, like numpy.einsum. "ij,jk->ik"
// is matrix multiplication.
func ExampleEinsum() {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	b := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{5, 6, 7, 8})
	c, _ := ops.Einsum("ij,jk->ik", a, b)
	fmt.Printf("%.0f %.0f %.0f %.0f\n", c.AtF64(0, 0), c.AtF64(0, 1), c.AtF64(1, 0), c.AtF64(1, 1))
	// Output: 19 22 43 50
}
