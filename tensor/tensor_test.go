package tensor

import "testing"

func TestNewZeroed(t *testing.T) {
	x := New(F64, Shape{2, 3})
	if x.Dtype() != F64 || !x.Shape().Equal(Shape{2, 3}) {
		t.Fatalf("unexpected dtype/shape: %v %v", x.Dtype(), x.Shape())
	}
	if !x.IsContiguous() || x.Numel() != 6 {
		t.Fatalf("want contiguous numel 6, got contig=%v numel=%d", x.IsContiguous(), x.Numel())
	}
	for i := range 2 {
		for j := range 3 {
			if v := x.AtF64(i, j); v != 0 {
				t.Fatalf("New must zero-init, got %v at (%d,%d)", v, i, j)
			}
		}
	}
}

func TestFromFloatAndAccess(t *testing.T) {
	x := FromFloat64(Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	// row-major: (1,2)=6
	if got := x.AtF64(1, 2); got != 6 {
		t.Fatalf("AtF64(1,2) = %v, want 6", got)
	}
	x.SetF64(42, 0, 1)
	if got := x.AtF64(0, 1); got != 42 {
		t.Fatalf("SetF64 then AtF64 = %v, want 42", got)
	}

	f32 := FromFloat32(Shape{3}, []float32{1.5, 2.5, 3.5})
	if got := f32.AtF64(2); got != 3.5 {
		t.Fatalf("f32 AtF64(2) = %v, want 3.5", got)
	}
}

func TestFromFloatLenMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("data/numel mismatch must panic")
		}
	}()
	FromFloat64(Shape{2, 2}, []float64{1, 2, 3})
}

func TestScalarTensor(t *testing.T) {
	s := New(F64, Shape{})
	if s.Numel() != 1 || s.Ndim() != 0 {
		t.Fatalf("scalar: numel=%d ndim=%d, want 1,0", s.Numel(), s.Ndim())
	}
	s.SetF64(7) // no indices
	if got := s.AtF64(); got != 7 {
		t.Fatalf("scalar AtF64 = %v, want 7", got)
	}
}

func TestStorageDtypeGuards(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("F32() on F64 storage must panic")
		}
	}()
	New(F64, Shape{2}).Storage().F32()
}

func TestCast(t *testing.T) {
	x := FromFloat64(Shape{2, 2}, []float64{1.5, -2.25, 3.0, 0.5})
	f := x.Cast(F32)
	if f.Dtype() != F32 || !f.Shape().Equal(Shape{2, 2}) {
		t.Fatalf("cast meta %v %v", f.Dtype(), f.Shape())
	}
	if f.AtF64(0, 1) != -2.25 { // exactly representable in f32
		t.Errorf("cast value %v", f.AtF64(0, 1))
	}
	back := f.Cast(F64)
	if back.Dtype() != F64 || back.AtF64(1, 0) != 3.0 {
		t.Errorf("round cast %v %v", back.Dtype(), back.AtF64(1, 0))
	}
}
