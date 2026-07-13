package ops_test

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// numpy.cumsum: a 1-D prefix sum, and per-axis prefix sums of a matrix (axis 0 down
// columns, axis 1 across rows).
func TestCumsumForward(t *testing.T) {
	v, _ := ops.Cumsum(tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 3, 4}), 0)
	for i, w := range []float64{1, 3, 6, 10} {
		if v.AtF64(i) != w {
			t.Errorf("1d[%d]=%g want %g", i, v.AtF64(i), w)
		}
	}
	m := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	c0, _ := ops.Cumsum(m, 0)
	for i, w := range []float64{1, 2, 3, 5, 7, 9} {
		if c0.AtF64(tensor.Unravel(i, c0.Shape())...) != w {
			t.Errorf("axis0[%d]=%g want %g", i, c0.AtF64(tensor.Unravel(i, c0.Shape())...), w)
		}
	}
	c1, _ := ops.Cumsum(m, 1)
	for i, w := range []float64{1, 3, 6, 4, 9, 15} {
		if c1.AtF64(tensor.Unravel(i, c1.Shape())...) != w {
			t.Errorf("axis1[%d]=%g want %g", i, c1.AtF64(tensor.Unravel(i, c1.Shape())...), w)
		}
	}
	// negative axis == last
	cn, _ := ops.Cumsum(m, -1)
	if cn.AtF64(0, 2) != 6 || cn.AtF64(1, 2) != 15 {
		t.Errorf("negative-axis cumsum wrong")
	}
}

func TestCumsumErrors(t *testing.T) {
	if _, err := ops.Cumsum(tensor.New(tensor.F64, tensor.Shape{2, 3}), 2); err == nil {
		t.Error("axis out of range must error")
	}
	if _, err := ops.Cumsum(tensor.New(tensor.F64, tensor.Shape{}), 0); err == nil {
		t.Error("scalar (rank 0) must error")
	}
}

// §V16 definitional: matches numpy.cumsum along an axis. Skips without the .venv.
func TestCumsumVsNumpy(t *testing.T) {
	py := "../.venv/bin/python"
	if _, err := os.Stat(py); err != nil {
		t.Skip("no .venv python")
	}
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1.5, -2, 3, 0.5, 4, -1})
	got, _ := ops.Cumsum(x, 1)
	out, err := exec.Command(py, "-c",
		"import numpy as np; a=np.array([1.5,-2,3,0.5,4,-1]).reshape(2,3); print(np.cumsum(a,axis=1).flatten().tolist())").CombinedOutput()
	if err != nil {
		t.Fatalf("numpy: %v\n%s", err, out)
	}
	want := strings.TrimSpace(string(out))
	var g string
	{
		parts := make([]string, 6)
		for i := range 6 {
			parts[i] = fmt.Sprintf("%g", got.AtF64(tensor.Unravel(i, got.Shape())...))
		}
		g = "[" + strings.Join(parts, ", ") + "]"
	}
	if g != want {
		t.Errorf("cumsum %s vs numpy %s", g, want)
	}
}

// Cumsum returns the running total along an axis, like numpy.cumsum. Here the
// prefix sums of [1,2,3,4] are [1,3,6,10].
func ExampleCumsum() {
	x := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 3, 4})
	c, _ := ops.Cumsum(x, 0)
	fmt.Printf("%.0f %.0f %.0f %.0f\n", c.AtF64(0), c.AtF64(1), c.AtF64(2), c.AtF64(3))
	// Output: 1 3 6 10
}
