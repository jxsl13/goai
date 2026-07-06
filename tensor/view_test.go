package tensor

import "testing"

// §V6/§C5: a view shares storage — writing through the view is visible in the
// parent, proving zero-copy semantics.
func TestViewsShareStorage(t *testing.T) {
	base := FromFloat64(Shape{2, 3}, []float64{0, 0, 0, 0, 0, 0})

	tr, err := base.Transpose(0, 1) // shape (3,2), strides (1,3)
	if err != nil {
		t.Fatal(err)
	}
	if !tr.Shape().Equal(Shape{3, 2}) {
		t.Fatalf("transpose shape %v, want (3, 2)", tr.Shape())
	}
	if tr.IsContiguous() {
		t.Error("transposed view must be non-contiguous")
	}
	tr.SetF64(9, 2, 0) // maps to base (0,2)
	if got := base.AtF64(0, 2); got != 9 {
		t.Fatalf("write via transpose not visible in base: base(0,2)=%v", got)
	}
}

func TestTransposeMapping(t *testing.T) {
	base := FromFloat64(Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	tr, _ := base.Transpose(0, 1)
	// tr(j,i) == base(i,j)
	for i := range 2 {
		for j := range 3 {
			if tr.AtF64(j, i) != base.AtF64(i, j) {
				t.Fatalf("transpose mismatch at (%d,%d)", i, j)
			}
		}
	}
}

func TestSlice(t *testing.T) {
	base := FromFloat64(Shape{4, 3}, []float64{
		0, 1, 2,
		3, 4, 5,
		6, 7, 8,
		9, 10, 11,
	})
	rows, err := base.Slice(0, 1, 3) // rows 1..2 → shape (2,3)
	if err != nil {
		t.Fatal(err)
	}
	if !rows.Shape().Equal(Shape{2, 3}) {
		t.Fatalf("slice shape %v, want (2,3)", rows.Shape())
	}
	if rows.AtF64(0, 0) != 3 || rows.AtF64(1, 2) != 8 {
		t.Fatalf("slice values wrong: (0,0)=%v (1,2)=%v", rows.AtF64(0, 0), rows.AtF64(1, 2))
	}
	// zero-copy: mutate slice → visible in base
	rows.SetF64(99, 0, 0)
	if base.AtF64(1, 0) != 99 {
		t.Fatalf("slice write not visible in base: %v", base.AtF64(1, 0))
	}
}

func TestReshapeContiguous(t *testing.T) {
	base := FromFloat64(Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	r, err := base.Reshape(Shape{3, 2})
	if err != nil {
		t.Fatal(err)
	}
	if r.AtF64(0, 0) != 1 || r.AtF64(2, 1) != 6 {
		t.Fatalf("reshape values wrong: (0,0)=%v (2,1)=%v", r.AtF64(0, 0), r.AtF64(2, 1))
	}
	// shares storage
	r.SetF64(-1, 0, 0)
	if base.AtF64(0, 0) != -1 {
		t.Error("reshape must share storage")
	}
}

func TestReshapeNonContiguousErrors(t *testing.T) {
	base := FromFloat64(Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	tr, _ := base.Transpose(0, 1)
	if _, err := tr.Reshape(Shape{6}); err == nil {
		t.Error("reshape of non-contiguous view must error")
	}
	// Contiguous() then reshape works and preserves logical order
	c := tr.Contiguous()
	if !c.IsContiguous() {
		t.Fatal("Contiguous() must produce contiguous tensor")
	}
	flat, err := c.Reshape(Shape{6})
	if err != nil {
		t.Fatal(err)
	}
	// transposed logical order of (2,3) 1..6 is column-major: 1,4,2,5,3,6
	want := []float64{1, 4, 2, 5, 3, 6}
	for i := range 6 {
		if flat.AtF64(i) != want[i] {
			t.Fatalf("contiguous+reshape order wrong at %d: got %v want %v", i, flat.AtF64(i), want[i])
		}
	}
}

func TestReshapeMismatchErrors(t *testing.T) {
	base := New(F64, Shape{2, 3})
	if _, err := base.Reshape(Shape{2, 2}); err == nil {
		t.Error("reshape changing numel must error")
	}
}

func TestPermute(t *testing.T) {
	base := New(F64, Shape{2, 3, 4})
	// fill with pos so we can check mapping
	n := base.Numel()
	for pos := range n {
		idx := Unravel(pos, base.Shape())
		base.SetF64(float64(pos), idx...)
	}
	p, err := base.Permute(2, 0, 1) // shape (4,2,3)
	if err != nil {
		t.Fatal(err)
	}
	if !p.Shape().Equal(Shape{4, 2, 3}) {
		t.Fatalf("permute shape %v, want (4,2,3)", p.Shape())
	}
	// p(a,b,c) == base(b,c,a)
	if p.AtF64(3, 1, 2) != base.AtF64(1, 2, 3) {
		t.Error("permute index mapping wrong")
	}
	if _, err := base.Permute(0, 0, 1); err == nil {
		t.Error("non-permutation must error")
	}
}

func TestViewBoundsErrors(t *testing.T) {
	base := New(F64, Shape{2, 3})
	if _, err := base.Slice(5, 0, 1); err == nil {
		t.Error("bad axis must error")
	}
	if _, err := base.Slice(0, 0, 9); err == nil {
		t.Error("out-of-range bound must error")
	}
	if _, err := base.Transpose(0, 9); err == nil {
		t.Error("bad transpose axis must error")
	}
}
