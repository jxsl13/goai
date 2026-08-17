package safetensors_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// mkMultiTensor writes a multi-dtype safetensors file and returns its path.
func mkMultiTensor(t *testing.T) string {
	t.Helper()
	f64 := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1.5, -2, 3.25, 4, 5.5, -6.75})
	f32 := tensor.Zeros(tensor.F32, tensor.Shape{4})
	for i, v := range []float64{0.5, -1.5, 2.5, -3.5} {
		f32.SetF64(v, i)
	}
	f16 := tensor.Zeros(tensor.F16, tensor.Shape{3})
	for i, v := range []float64{1.5, -2.5, 0.75} {
		f16.SetF64(v, i)
	}
	m := map[string]*tensor.Tensor{"big.f64": f64, "small.f32": f32, "half.f16": f16}
	p := t.TempDir() + "/multi.safetensors"
	if err := safetensors.SaveFile(p, m, map[string]string{"k": "v"}); err != nil {
		t.Fatal(err)
	}
	return p
}

// Names reads the header only and describes every tensor without loading data.
func TestNamesHeaderOnly(t *testing.T) {
	p := mkMultiTensor(t)
	infos, meta, err := safetensors.Names(p)
	if err != nil {
		t.Fatal(err)
	}
	if meta["k"] != "v" {
		t.Errorf("metadata not returned: %v", meta)
	}
	// Sorted by name: big.f64, half.f16, small.f32.
	want := []struct {
		name  string
		dt    tensor.Dtype
		shape tensor.Shape
	}{
		{"big.f64", tensor.F64, tensor.Shape{2, 3}},
		{"half.f16", tensor.F16, tensor.Shape{3}},
		{"small.f32", tensor.F32, tensor.Shape{4}},
	}
	if len(infos) != len(want) {
		t.Fatalf("got %d infos, want %d", len(infos), len(want))
	}
	for i, w := range want {
		got := infos[i]
		if got.Name != w.name || got.Dtype != w.dt || !got.Shape.Equal(w.shape) {
			t.Errorf("info[%d] = {%s %v %v}, want {%s %v %v}", i, got.Name, got.Dtype, got.Shape, w.name, w.dt, w.shape)
		}
	}
}

// LoadTensor must return each tensor bit-identical to LoadFile — proving the
// single-tensor seek path reuses the same decode, not a divergent one.
func TestLoadTensorBitIdenticalToLoadFile(t *testing.T) {
	p := mkMultiTensor(t)
	all, _, err := safetensors.LoadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	for name, whole := range all {
		one, err := safetensors.LoadTensor(p, name)
		if err != nil {
			t.Fatalf("LoadTensor(%q): %v", name, err)
		}
		if one.Dtype() != whole.Dtype() || !one.Shape().Equal(whole.Shape()) {
			t.Fatalf("%s: dtype/shape mismatch %v/%v vs %v/%v", name, one.Dtype(), one.Shape(), whole.Dtype(), whole.Shape())
		}
		n := whole.Numel()
		for i := range n {
			idx := tensor.Unravel(i, whole.Shape())
			if math.Float64bits(one.AtF64(idx...)) != math.Float64bits(whole.AtF64(idx...)) {
				t.Errorf("%s elem %d: LoadTensor %v != LoadFile %v", name, i, one.AtF64(idx...), whole.AtF64(idx...))
				break
			}
		}
	}
}

// LoadTensor against the independent reference F16/BF16 fixture (T901): a single
// tensor pulled by name decodes to the exact reference values.
func TestLoadTensorFromReferenceHalfFloat(t *testing.T) {
	want := [2][3]float64{{1.5, -2.5, 3.5}, {-4.25, 0.75, -0.125}}
	for _, name := range []string{"h16", "hbf16"} {
		x, err := safetensors.LoadTensor("testdata/ref_halffloat.safetensors", name)
		if err != nil {
			t.Fatalf("LoadTensor(%q): %v", name, err)
		}
		for i := range 2 {
			for j := range 3 {
				if x.AtF64(i, j) != want[i][j] {
					t.Errorf("%s(%d,%d) = %v, want %v", name, i, j, x.AtF64(i, j), want[i][j])
				}
			}
		}
	}
}

func TestLoadTensorErrors(t *testing.T) {
	p := mkMultiTensor(t)
	if _, err := safetensors.LoadTensor(p, "nope"); err == nil {
		t.Error("want error for a missing tensor name")
	}
	if _, _, err := safetensors.Names("/no/such/file.safetensors"); err == nil {
		t.Error("want error for a missing file")
	}
}

func TestLoadTensorZeroSizedTensor(t *testing.T) {
	p := t.TempDir() + "/zero.safetensors"
	if err := safetensors.SaveFile(p, map[string]*tensor.Tensor{
		"zero": tensor.Zeros(tensor.F32, tensor.Shape{2, 0, 3}),
	}, nil); err != nil {
		t.Fatal(err)
	}
	x, err := safetensors.LoadTensor(p, "zero")
	if err != nil {
		t.Fatal(err)
	}
	if x.Numel() != 0 || !x.Shape().Equal(tensor.Shape{2, 0, 3}) {
		t.Fatalf("zero tensor = shape %v, numel %d", x.Shape(), x.Numel())
	}
}

// The whole point: LoadTensor pulling a small tensor from a file with a large one
// must NOT allocate the large tensor. A 64 MiB tensor sits next to a tiny one;
// loading the tiny one allocates far less than 64 MiB.
func TestLoadTensorDoesNotMaterializeOthers(t *testing.T) {
	big := tensor.Zeros(tensor.F32, tensor.Shape{16 << 20}) // 64 MiB
	tiny := tensor.Zeros(tensor.F64, tensor.Shape{2})
	tiny.SetF64(3.5, 0)
	p := t.TempDir() + "/mixed.safetensors"
	if err := safetensors.SaveFile(p, map[string]*tensor.Tensor{"big": big, "tiny": tiny}, nil); err != nil {
		t.Fatal(err)
	}
	var m0, m1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m0)
	x, err := safetensors.LoadTensor(p, "tiny")
	runtime.ReadMemStats(&m1)
	if err != nil {
		t.Fatal(err)
	}
	if x.AtF64(0) != 3.5 {
		t.Errorf("tiny[0] = %v, want 3.5", x.AtF64(0))
	}
	if got := (m1.TotalAlloc - m0.TotalAlloc) >> 20; got > 8 {
		t.Errorf("LoadTensor(tiny) allocated %d MiB — it materialized the 64 MiB neighbour instead of seeking", got)
	}
}
