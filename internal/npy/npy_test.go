package npy

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// Reading real numpy-generated .npy files (validates the parser against the
// actual format, §T10). Regenerate with `make golden`.
func TestReadNumpyFiles(t *testing.T) {
	m64, err := ReadFile("testdata/mat_f64.npy")
	if err != nil {
		t.Fatal(err)
	}
	if m64.Dtype() != tensor.F64 || !m64.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("mat_f64: dtype %v shape %v", m64.Dtype(), m64.Shape())
	}
	want := [][3]float64{{1.5, -2.0, 3.25}, {4.0, 5.5, -6.75}}
	for i := range 2 {
		for j := range 3 {
			if m64.AtF64(i, j) != want[i][j] {
				t.Errorf("mat_f64(%d,%d)=%v want %v", i, j, m64.AtF64(i, j), want[i][j])
			}
		}
	}

	m32, err := ReadFile("testdata/mat_f32.npy")
	if err != nil {
		t.Fatal(err)
	}
	if m32.Dtype() != tensor.F32 || !m32.Shape().Equal(tensor.Shape{3, 4}) {
		t.Fatalf("mat_f32: dtype %v shape %v", m32.Dtype(), m32.Shape())
	}
	if m32.AtF64(0, 0) != 0 || m32.AtF64(2, 3) != 11*0.5 {
		t.Errorf("mat_f32 values wrong: (0,0)=%v (2,3)=%v", m32.AtF64(0, 0), m32.AtF64(2, 3))
	}

	v, err := ReadFile("testdata/vec_f64.npy")
	if err != nil {
		t.Fatal(err)
	}
	if !v.Shape().Equal(tensor.Shape{3}) || v.AtF64(2) != 3 {
		t.Errorf("vec: shape %v (2)=%v", v.Shape(), v.AtF64(2))
	}
}

// Go write → Go read round-trip preserves dtype, shape, and values.
func TestRoundTrip(t *testing.T) {
	cases := []*tensor.Tensor{
		tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
		tensor.FromFloat32(tensor.Shape{4}, []float32{1.5, -2.5, 3.5, 0}),
		tensor.FromFloat64(tensor.Shape{}, []float64{42}), // scalar
	}
	for _, in := range cases {
		var buf bytes.Buffer
		if err := Write(&buf, in); err != nil {
			t.Fatal(err)
		}
		out, err := Read(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if out.Dtype() != in.Dtype() || !out.Shape().Equal(in.Shape()) {
			t.Fatalf("round-trip meta: %v %v vs %v %v", out.Dtype(), out.Shape(), in.Dtype(), in.Shape())
		}
		for i := range in.Numel() {
			idx := tensor.Unravel(i, in.Shape())
			if out.AtF64(idx...) != in.AtF64(idx...) {
				t.Errorf("round-trip value [%d]: %v vs %v", i, out.AtF64(idx...), in.AtF64(idx...))
			}
		}
	}
}

// Non-contiguous input is materialized on write.
func TestWriteNonContiguous(t *testing.T) {
	m := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	tr, _ := m.Transpose(0, 1) // (3,2) non-contiguous
	var buf bytes.Buffer
	if err := Write(&buf, tr); err != nil {
		t.Fatal(err)
	}
	out, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{3, 2}) || out.AtF64(0, 1) != 4 {
		t.Errorf("transposed write: shape %v (0,1)=%v", out.Shape(), out.AtF64(0, 1))
	}
}

func TestBadMagic(t *testing.T) {
	if _, err := Read(bytes.NewReader([]byte("not a npy file at all"))); err == nil {
		t.Error("bad magic must error")
	}
}
