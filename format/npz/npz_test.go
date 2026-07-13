package npz

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func flat(t *tensor.Tensor) []float64 {
	d := make([]float64, t.Numel())
	for i := range d {
		d[i] = t.AtF64(tensor.Unravel(i, t.Shape())...)
	}
	return d
}

// §V15 round-trip: a set of named tensors survives Save→Load bit-exactly, with
// names, shapes and dtypes preserved.
func TestRoundTrip(t *testing.T) {
	in := map[string]*tensor.Tensor{
		"weight": tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
		"bias":   tensor.FromFloat64(tensor.Shape{3}, []float64{-1, 0.5, 2}),
		"scale":  tensor.New(tensor.F32, tensor.Shape{}),
	}
	in["scale"].SetF64(3.25)
	var buf bytes.Buffer
	if err := Save(&buf, in); err != nil {
		t.Fatal(err)
	}
	got, err := Load(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(in) {
		t.Fatalf("got %d arrays, want %d", len(got), len(in))
	}
	for name, want := range in {
		g, ok := got[name]
		if !ok {
			t.Fatalf("missing array %q", name)
		}
		if !g.Shape().Equal(want.Shape()) || g.Dtype() != want.Dtype() {
			t.Errorf("%q: shape/dtype %v/%v != %v/%v", name, g.Shape(), g.Dtype(), want.Shape(), want.Dtype())
		}
		fg, fw := flat(g), flat(want)
		for i := range fw {
			if fg[i] != fw[i] {
				t.Errorf("%q elem %d: %g != %g", name, i, fg[i], fw[i])
			}
		}
	}
}

// §V16 definitional: an archive written by the reference numpy.savez loads
// byte-faithfully (names, shapes, dtypes, values).
func TestLoadNumpyReference(t *testing.T) {
	got, err := LoadFile("testdata/ref.npz")
	if err != nil {
		t.Fatal(err)
	}
	w, ok := got["weight"]
	if !ok || !w.Shape().Equal(tensor.Shape{2, 2}) || w.Dtype() != tensor.F64 {
		t.Fatalf("weight missing/wrong: %v", got["weight"])
	}
	for i, v := range []float64{1, 2, 3, 4} {
		if flat(w)[i] != v {
			t.Errorf("weight[%d]=%g want %g", i, flat(w)[i], v)
		}
	}
	b, ok := got["bias"]
	if !ok || !b.Shape().Equal(tensor.Shape{2}) || b.Dtype() != tensor.F32 {
		t.Fatalf("bias missing/wrong: %v", got["bias"])
	}
	for i, v := range []float64{0.5, -0.5} {
		if flat(b)[i] != v {
			t.Errorf("bias[%d]=%g want %g", i, flat(b)[i], v)
		}
	}
}

// §V16 definitional: an archive GoAI writes is read back identically by the
// reference numpy.load. Skips when the .venv is absent.
func TestSaveReadableByNumpy(t *testing.T) {
	py := "../../.venv/bin/python"
	if _, err := os.Stat(py); err != nil {
		t.Skip("no .venv python for numpy cross-check")
	}
	in := map[string]*tensor.Tensor{
		"a": tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1.5, -2, 3.25, 4}),
		"b": tensor.FromFloat64(tensor.Shape{3}, []float64{7, 8, 9}),
	}
	path := filepath.Join(t.TempDir(), "go.npz")
	if err := SaveFile(path, in); err != nil {
		t.Fatal(err)
	}
	script := fmt.Sprintf("import numpy as np; z=np.load(%q); print(sorted(z.files), z['a'].flatten().tolist(), z['b'].tolist())", path)
	out, err := exec.Command(py, "-c", script).CombinedOutput()
	if err != nil {
		t.Fatalf("numpy load failed: %v\n%s", err, out)
	}
	if want := "['a', 'b'] [1.5, -2.0, 3.25, 4.0] [7.0, 8.0, 9.0]\n"; string(out) != want {
		t.Errorf("numpy read %q, want %q", string(out), want)
	}
}

func TestLoadErrors(t *testing.T) {
	// not a zip
	if _, err := Load(bytes.NewReader([]byte("not a zip archive")), 17); err == nil {
		t.Error("non-zip must error")
	}
	// empty
	if _, err := Load(bytes.NewReader(nil), 0); err == nil {
		t.Error("empty must error")
	}
}

// Save writes a set of named tensors to a NumPy .npz archive (like numpy.savez);
// Load reads them back into a map. A round-trip preserves names and values.
func ExampleSave() {
	arrays := map[string]*tensor.Tensor{
		"w": tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2}),
	}
	var buf bytes.Buffer
	_ = Save(&buf, arrays)
	got, _ := Load(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	fmt.Println(got["w"].Shape(), got["w"].AtF64(1))
	// Output: (2) 2
}
