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

// numpy.var/std (ddof=0): full-reduction population variance of [1,2,3,4] is 1.25,
// std is √1.25.
func TestVarStdFull(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 3, 4})
	v, err := ops.Var(x, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(v.AtF64()-1.25) > 1e-12 {
		t.Errorf("Var = %g, want 1.25", v.AtF64())
	}
	s, _ := ops.Std(x, nil, false)
	if math.Abs(s.AtF64()-math.Sqrt(1.25)) > 1e-12 {
		t.Errorf("Std = %g, want %g", s.AtF64(), math.Sqrt(1.25))
	}
}

// numpy.var along an axis: for [[1,2,3],[4,5,6]], var over axis 0 (columns) is
// [2.25,2.25,2.25]; over axis 1 (rows) is [2/3, 2/3].
func TestVarAxis(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	v0, _ := ops.Var(x, []int{0}, false)
	if !v0.Shape().Equal(tensor.Shape{3}) {
		t.Fatalf("axis0 shape %v", v0.Shape())
	}
	for i := range 3 {
		if math.Abs(v0.AtF64(i)-2.25) > 1e-12 {
			t.Errorf("var axis0[%d]=%g, want 2.25", i, v0.AtF64(i))
		}
	}
	v1, _ := ops.Var(x, []int{1}, false)
	if !v1.Shape().Equal(tensor.Shape{2}) {
		t.Fatalf("axis1 shape %v", v1.Shape())
	}
	for i := range 2 {
		if math.Abs(v1.AtF64(i)-2.0/3.0) > 1e-12 {
			t.Errorf("var axis1[%d]=%g, want %g", i, v1.AtF64(i), 2.0/3.0)
		}
	}
	// keepdims retains the reduced axis as size 1
	vk, _ := ops.Var(x, []int{1}, true)
	if !vk.Shape().Equal(tensor.Shape{2, 1}) {
		t.Errorf("keepdims shape %v, want [2 1]", vk.Shape())
	}
}

// §V16 definitional: matches the reference numpy.var / numpy.std (ddof=0) at 1e-10.
// Skips when the .venv is absent.
func TestVarStdVsNumpy(t *testing.T) {
	py := "../.venv/bin/python"
	if _, err := os.Stat(py); err != nil {
		t.Skip("no .venv python for numpy cross-check")
	}
	xd := []float64{1.5, -2, 3.25, 0.5, 4, -1.75, 2.2, 0.1}
	x := tensor.FromFloat64(tensor.Shape{2, 4}, xd)
	gotV, _ := ops.Var(x, []int{1}, false)
	gotS, _ := ops.Std(x, []int{1}, false)

	script := "import numpy as np;" +
		"a=np.array([1.5,-2,3.25,0.5,4,-1.75,2.2,0.1]).reshape(2,4);" +
		"print(np.var(a,axis=1).tolist(), np.std(a,axis=1).tolist())"
	out, err := exec.Command(py, "-c", script).CombinedOutput()
	if err != nil {
		t.Fatalf("numpy failed: %v\n%s", err, out)
	}
	// parse "[v0, v1] [s0, s1]"
	txt := strings.TrimSpace(string(out))
	var v0, v1, s0, s1 float64
	if _, err := fmt.Sscanf(txt, "[%g, %g] [%g, %g]", &v0, &v1, &s0, &s1); err != nil {
		t.Fatalf("parse %q: %v", txt, err)
	}
	if math.Abs(gotV.AtF64(0)-v0) > 1e-10 || math.Abs(gotV.AtF64(1)-v1) > 1e-10 {
		t.Errorf("Var %v vs numpy [%g %g]", []float64{gotV.AtF64(0), gotV.AtF64(1)}, v0, v1)
	}
	if math.Abs(gotS.AtF64(0)-s0) > 1e-10 || math.Abs(gotS.AtF64(1)-s1) > 1e-10 {
		t.Errorf("Std %v vs numpy [%g %g]", []float64{gotS.AtF64(0), gotS.AtF64(1)}, s0, s1)
	}
}

// Var computes the population variance over an axis (numpy.var, ddof=0). Here each
// row of [[1,2,3],[4,5,6]] has variance 2/3 ≈ 0.6667.
func ExampleVar() {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	v, _ := ops.Var(x, []int{1}, false)
	fmt.Printf("%.4f %.4f\n", v.AtF64(0), v.AtF64(1))
	// Output: 0.6667 0.6667
}

// §V16 definitional (§R140 ddof follow-up): sample variance/std with Bessel's correction.
// [1,2,3,4]: SS=5, N=4 → ddof=1 gives 5/3, std √(5/3). ddof=0 reduces to population (5/4).
func TestVarStdDDofValues(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 3, 4})
	pop, _ := ops.VarDDof(x, nil, false, 0)
	if math.Abs(pop.AtF64()-1.25) > 1e-12 {
		t.Errorf("ddof=0 var = %g, want 1.25 (== population Var)", pop.AtF64())
	}
	samp, _ := ops.VarDDof(x, nil, false, 1)
	if math.Abs(samp.AtF64()-5.0/3.0) > 1e-12 {
		t.Errorf("ddof=1 var = %.12g, want %.12g", samp.AtF64(), 5.0/3.0)
	}
	sstd, _ := ops.StdDDof(x, nil, false, 1)
	if math.Abs(sstd.AtF64()-math.Sqrt(5.0/3.0)) > 1e-12 {
		t.Errorf("ddof=1 std = %.12g, want %.12g", sstd.AtF64(), math.Sqrt(5.0/3.0))
	}
	if _, err := ops.VarDDof(x, nil, false, 4); err == nil { // N−ddof=0
		t.Error("ddof=N: want error")
	}
	if _, err := ops.VarDDof(x, nil, false, -1); err == nil {
		t.Error("ddof<0: want error")
	}
}

// §V16 definitional: VarDDof matches numpy.var/std with ddof=1 over an axis. Skips without .venv.
func TestVarStdDDofVsNumpy(t *testing.T) {
	py := "../.venv/bin/python"
	if _, err := os.Stat(py); err != nil {
		t.Skip("no .venv python for numpy cross-check")
	}
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1.5, -2, 3.25, 0.5, 4, -1.75, 2.2, 0.1})
	gotV, _ := ops.VarDDof(x, []int{1}, false, 1)
	gotS, _ := ops.StdDDof(x, []int{1}, false, 1)
	script := "import numpy as np;" +
		"a=np.array([1.5,-2,3.25,0.5,4,-1.75,2.2,0.1]).reshape(2,4);" +
		"print(np.var(a,axis=1,ddof=1).tolist(), np.std(a,axis=1,ddof=1).tolist())"
	out, err := exec.Command(py, "-c", script).CombinedOutput()
	if err != nil {
		t.Fatalf("numpy failed: %v\n%s", err, out)
	}
	var v0, v1, s0, s1 float64
	if _, err := fmt.Sscanf(strings.TrimSpace(string(out)), "[%g, %g] [%g, %g]", &v0, &v1, &s0, &s1); err != nil {
		t.Fatalf("parse %q: %v", out, err)
	}
	if math.Abs(gotV.AtF64(0)-v0) > 1e-10 || math.Abs(gotV.AtF64(1)-v1) > 1e-10 {
		t.Errorf("VarDDof %v vs numpy [%g %g]", []float64{gotV.AtF64(0), gotV.AtF64(1)}, v0, v1)
	}
	if math.Abs(gotS.AtF64(0)-s0) > 1e-10 || math.Abs(gotS.AtF64(1)-s1) > 1e-10 {
		t.Errorf("StdDDof %v vs numpy [%g %g]", []float64{gotS.AtF64(0), gotS.AtF64(1)}, s0, s1)
	}
}

// §V2: keepdims retains the reduced axis as size 1, and per-axis ddof rescaling is the exact
// N/(N−ddof) factor over the population variance (here N=3 per row ⇒ factor 3/2).
func TestVarDDofKeepdimsAndAxes(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 6, 8})
	vk, _ := ops.VarDDof(x, []int{1}, true, 1)
	if !vk.Shape().Equal(tensor.Shape{2, 1}) {
		t.Fatalf("keepdims shape %v, want [2 1]", vk.Shape())
	}
	pop, _ := ops.Var(x, []int{1}, true)
	for i := range 2 {
		if got, want := vk.AtF64(i, 0), pop.AtF64(i, 0)*3.0/2.0; math.Abs(got-want) > 1e-12 {
			t.Errorf("row %d ddof=1 var %.12g != population·3/2 %.12g", i, got, want)
		}
	}
	// full 2-axis reduction: N=6, ddof=1 ⇒ population·6/5.
	full1, _ := ops.VarDDof(x, nil, false, 1)
	full0, _ := ops.Var(x, nil, false)
	if got, want := full1.AtF64(), full0.AtF64()*6.0/5.0; math.Abs(got-want) > 1e-12 {
		t.Errorf("full ddof=1 var %.12g != population·6/5 %.12g", got, want)
	}
}

func ExampleVarDDof() {
	// Sample (unbiased) variance of [1,2,3,4] with Bessel's correction: SS=5, N−1=3 ⇒ 5/3.
	x := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 3, 4})
	v, _ := ops.VarDDof(x, nil, false, 1)
	fmt.Printf("%.4f\n", v.AtF64())
	// Output: 1.6667
}
