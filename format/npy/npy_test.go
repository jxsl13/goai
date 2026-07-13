package npy

import (
	"bytes"
	"encoding/binary"
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

// §V15 round-trip: any tensor survives Save→Load bit-exactly, across shapes and
// both dtypes (scalar, 1-D with the trailing-comma tuple, and N-D).
func TestRoundTrip(t *testing.T) {
	cases := []*tensor.Tensor{
		tensor.FromFloat64(tensor.Shape{}, []float64{42.5}),
		tensor.FromFloat64(tensor.Shape{5}, []float64{1, -2, 3.5, 0, -0.25}),
		tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
		tensor.New(tensor.F32, tensor.Shape{2, 2, 2}),
	}
	cases[3].SetF64(1.5, 0, 0, 0)
	cases[3].SetF64(-9.25, 1, 1, 1)
	for i, in := range cases {
		var buf bytes.Buffer
		if err := Save(&buf, in); err != nil {
			t.Fatalf("case %d save: %v", i, err)
		}
		out, err := Load(&buf)
		if err != nil {
			t.Fatalf("case %d load: %v", i, err)
		}
		if !out.Shape().Equal(in.Shape()) || out.Dtype() != in.Dtype() {
			t.Fatalf("case %d: shape/dtype %v/%v != %v/%v", i, out.Shape(), out.Dtype(), in.Shape(), in.Dtype())
		}
		fi, fo := flat(in), flat(out)
		for j := range fi {
			if fi[j] != fo[j] {
				t.Errorf("case %d elem %d: %g != %g", i, j, fo[j], fi[j])
			}
		}
	}
}

// §V16 definitional: arrays written by the reference numpy load byte-faithfully.
func TestLoadNumpyReference(t *testing.T) {
	f64, err := LoadFile("testdata/ref_f64.npy")
	if err != nil {
		t.Fatal(err)
	}
	if !f64.Shape().Equal(tensor.Shape{2, 3}) || f64.Dtype() != tensor.F64 {
		t.Fatalf("f64 ref shape/dtype %v/%v", f64.Shape(), f64.Dtype())
	}
	want64 := []float64{1.5, -2.0, 3.25, 4.0, 5.5, -6.75}
	for i, w := range want64 {
		if flat(f64)[i] != w {
			t.Errorf("f64[%d]=%g want %g", i, flat(f64)[i], w)
		}
	}
	f32, err := LoadFile("testdata/ref_f32.npy")
	if err != nil {
		t.Fatal(err)
	}
	if !f32.Shape().Equal(tensor.Shape{4}) || f32.Dtype() != tensor.F32 {
		t.Fatalf("f32 ref shape/dtype %v/%v", f32.Shape(), f32.Dtype())
	}
	for i, w := range []float64{0.5, 1.5, 2.5, -3.5} {
		if flat(f32)[i] != w {
			t.Errorf("f32[%d]=%g want %g", i, flat(f32)[i], w)
		}
	}
}

// §V16 definitional: a file GoAI writes is read back identically by the reference
// numpy — byte-faithful in the other direction. Skips when the .venv is absent.
func TestSaveReadableByNumpy(t *testing.T) {
	py := "../../.venv/bin/python"
	if _, err := os.Stat(py); err != nil {
		t.Skip("no .venv python for numpy cross-check")
	}
	in := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1.5, -2, 3.25, 4, 5.5, -6.75})
	path := filepath.Join(t.TempDir(), "go.npy")
	if err := SaveFile(path, in); err != nil {
		t.Fatal(err)
	}
	script := fmt.Sprintf("import numpy as np; a=np.load(%q); print(a.dtype, list(a.shape), a.flatten().tolist())", path)
	out, err := exec.Command(py, "-c", script).CombinedOutput()
	if err != nil {
		t.Fatalf("numpy load failed: %v\n%s", err, out)
	}
	got := string(out)
	if want := "float64 [2, 3] [1.5, -2.0, 3.25, 4.0, 5.5, -6.75]"; got != want+"\n" {
		t.Errorf("numpy read %q, want %q", got, want)
	}
}

func TestLoadErrors(t *testing.T) {
	bad := [][]byte{
		{},
		[]byte("\x93NUMPYxx"),                // bad magic tail (version bytes) — magic ok, truncated
		append([]byte("XXXXXX"), 1, 0, 5, 0), // bad magic
		append([]byte("\x93NUMPY"), 9, 0),    // unsupported version
	}
	for i, b := range bad {
		if _, err := Load(bytes.NewReader(b)); err == nil {
			t.Errorf("case %d: expected error", i)
		}
	}
	// big-endian descr must be rejected
	if _, err := dtypeFromDescr(">f8"); err == nil {
		t.Error("big-endian must error")
	}
}

func TestSaveUnsupportedDtype(t *testing.T) {
	// F16/F32/F64 supported; an unknown dtype must error via descrFor.
	if _, err := descrFor(tensor.Dtype(255)); err == nil {
		t.Error("unsupported dtype must error")
	}
	// BF16 has no standard numpy descriptor, so it is deliberately rejected.
	if _, err := descrFor(tensor.BF16); err == nil {
		t.Error("BF16 must error (no standard numpy descr)")
	}
	if _, err := dtypeFromDescr("<f2"); err != nil {
		t.Errorf("<f2 (float16) must be accepted: %v", err)
	}
}

// Save writes a tensor to the NumPy .npy format; Load reads it back. A round-trip
// preserves shape, dtype and values.
func ExampleSave() {
	in := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	var buf bytes.Buffer
	_ = Save(&buf, in)
	out, _ := Load(&buf)
	fmt.Println(out.Shape(), out.AtF64(1, 1))
	// Output: (2, 2) 4
}

// f16From builds an F16 tensor from exactly-representable float values (via SetF64).
func f16From(shape tensor.Shape, vals []float64) *tensor.Tensor {
	t := tensor.New(tensor.F16, shape)
	for i, v := range vals {
		t.SetF64(v, tensor.Unravel(i, shape)...)
	}
	return t
}

// §V15 (numpy.lib.format): F16 tensors round-trip through .npy bit-exactly — the raw IEEE binary16
// bits are stored verbatim (incl. +0/normal/subnormal/NaN/±Inf patterns).
func TestRoundTripF16(t *testing.T) {
	in := tensor.New(tensor.F16, tensor.Shape{2, 3})
	copy(in.Storage().U16(), []uint16{0x0000, 0x3C00, 0x0001, 0x7E00, 0x7C00, 0xFC00})
	var buf bytes.Buffer
	if err := Save(&buf, in); err != nil {
		t.Fatal(err)
	}
	out, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if out.Dtype() != tensor.F16 || !out.Shape().Equal(in.Shape()) {
		t.Fatalf("meta drift: %v%v", out.Dtype(), out.Shape())
	}
	gb, ib := out.Storage().U16(), in.Storage().U16()
	for i := range ib {
		if gb[i] != ib[i] {
			t.Errorf("bit drift at %d: %04x vs %04x", i, gb[i], ib[i])
		}
	}
}

// §V15 / §V16 definitional: byte-faithful numpy.float16 interop both ways. Skips without .venv.
func TestF16NumpyInterop(t *testing.T) {
	py := "../../.venv/bin/python"
	if _, err := os.Stat(py); err != nil {
		t.Skip("no .venv python for numpy cross-check")
	}
	dir := t.TempDir()
	// GoAI → numpy: exactly-representable half values must read back as float16 with the same values.
	vals := []float64{1, -2, 0.5, 0.25, 3.5, -0.125}
	goPath := filepath.Join(dir, "go_f16.npy")
	if err := SaveFile(goPath, f16From(tensor.Shape{2, 3}, vals)); err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(py, "-c", fmt.Sprintf(
		"import numpy as np; a=np.load(%q); print(a.dtype, list(a.shape), a.flatten().tolist())", goPath)).CombinedOutput()
	if err != nil {
		t.Fatalf("numpy load failed: %v\n%s", err, out)
	}
	if want := "float16 [2, 3] [1.0, -2.0, 0.5, 0.25, 3.5, -0.125]\n"; string(out) != want {
		t.Errorf("numpy read %q, want %q", out, want)
	}
	// numpy → GoAI: a numpy-written float16 array must load bit-exactly.
	npPath := filepath.Join(dir, "np_f16.npy")
	mk, err := exec.Command(py, "-c", fmt.Sprintf(
		"import numpy as np; np.save(%q, np.array([[1,-2,0.5],[0.25,3.5,-0.125]], dtype=np.float16))", npPath)).CombinedOutput()
	if err != nil {
		t.Fatalf("numpy save failed: %v\n%s", err, mk)
	}
	got, err := LoadFile(npPath)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dtype() != tensor.F16 || !got.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("loaded %v%v, want F16[2 3]", got.Dtype(), got.Shape())
	}
	for i, want := range vals {
		if g := got.AtF64(tensor.Unravel(i, got.Shape())...); g != want {
			t.Errorf("elem %d = %g, want %g", i, g, want)
		}
	}
}

// §V15 fuzz: arbitrary F16 tensors survive Save→Load bit-exactly (bits stored verbatim).
func FuzzRoundTripF16(f *testing.F) {
	f.Add(2, 3, []byte{0, 0x3C, 0, 0x40, 1, 0, 0, 0x7E, 0, 0x7C, 0, 0xFC})
	f.Fuzz(func(t *testing.T, d0, d1 int, raw []byte) {
		if d0 < 0 || d1 < 0 || d0 > 24 || d1 > 24 {
			return
		}
		n := d0 * d1
		if len(raw) < n*2 {
			return
		}
		in := tensor.New(tensor.F16, tensor.Shape{d0, d1})
		bits := in.Storage().U16()
		for i := range bits {
			bits[i] = binary.LittleEndian.Uint16(raw[i*2:])
		}
		var buf bytes.Buffer
		if err := Save(&buf, in); err != nil {
			t.Fatalf("save: %v", err)
		}
		out, err := Load(&buf)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if out.Dtype() != tensor.F16 || !out.Shape().Equal(in.Shape()) {
			t.Fatalf("meta drift: %v%v", out.Dtype(), out.Shape())
		}
		gb := out.Storage().U16()
		for i := range bits {
			if gb[i] != bits[i] {
				t.Fatalf("bit drift at %d: %04x vs %04x", i, gb[i], bits[i])
			}
		}
	})
}
