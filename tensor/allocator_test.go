package tensor

import "testing"

func TestSizeClass(t *testing.T) {
	cases := map[int]int{1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 8: 3, 9: 4}
	for n, want := range cases {
		if got := sizeClass(n); got != want {
			t.Errorf("sizeClass(%d) = %d, want %d (cap %d)", n, got, want, 1<<got)
		}
		if 1<<sizeClass(n) < n {
			t.Errorf("class capacity %d < n %d", 1<<sizeClass(n), n)
		}
	}
}

func TestHeapAllocZeroed(t *testing.T) {
	a := Heap()
	f64 := a.Alloc(F64, 4).([]float64)
	if len(f64) != 4 {
		t.Fatalf("len %d, want 4", len(f64))
	}
	for _, v := range f64 {
		if v != 0 {
			t.Fatal("Heap.Alloc must be zeroed")
		}
	}
	a.Free(f64) // no-op, must not panic
	if a.Alignment() != 0 {
		t.Error("heap alignment is natural (0)")
	}
}

// §V6: a pooled buffer, once freed, is reused on the next same-class Alloc and
// is re-zeroed (no stale data leaks).
func TestPoolReuseAndZero(t *testing.T) {
	p := NewPool()
	b1 := p.Alloc(F64, 6).([]float64) // class 3, cap 8
	for i := range b1 {
		b1[i] = float64(i + 1) // dirty it
	}
	p.Free(b1)

	b2 := p.Alloc(F64, 5).([]float64) // same class 3 → should reuse b1's array
	if cap(b2) != 8 {
		t.Fatalf("expected reused cap 8, got %d", cap(b2))
	}
	if &b1[:1][0] != &b2[:1][0] {
		t.Error("expected pool to reuse the same backing array")
	}
	for _, v := range b2 {
		if v != 0 {
			t.Fatal("pooled buffer must be re-zeroed on Alloc")
		}
	}
}

func TestPoolEmptyAndForeign(t *testing.T) {
	p := NewPool()
	e := p.Alloc(F32, 0).([]float32)
	if len(e) != 0 {
		t.Fatalf("empty alloc len %d", len(e))
	}
	// Freeing a foreign (non-power-of-two cap) buffer must be a safe no-op.
	p.Free(make([]float64, 3)) // cap 3, dropped
	p.Free("not a buffer")     // ignored
}

func TestPoolAlignmentAdvisory(t *testing.T) {
	p := NewPool(WithAlignment(64))
	if p.Alignment() != 64 {
		t.Errorf("Alignment() = %d, want 64 (advisory)", p.Alignment())
	}
	// advisory only: allocation still succeeds and is usable
	if got := len(p.Alloc(F32, 10).([]float32)); got != 10 {
		t.Errorf("alloc len %d, want 10", got)
	}
}

func TestDeviceBasics(t *testing.T) {
	if CPU().Kind() != KindCPU || CPU().String() != "cpu" {
		t.Error("CPU device identity wrong")
	}
	if KindCUDA.String() != "cuda" || KindMetal.String() != "metal" {
		t.Error("device kind strings wrong")
	}
	// nil allocator falls back to heap
	d := NewCPUDevice(nil)
	if d.Allocator() == nil {
		t.Error("nil allocator must fall back to heap")
	}
}

// A tensor built on a pooled CPU device uses that pool, and Release returns the
// buffer for reuse.
func TestTensorOnPooledDeviceRelease(t *testing.T) {
	p := NewPool()
	dev := NewCPUDevice(p)
	x := NewOn(dev, F64, Shape{4})
	if x.Device().Kind() != KindCPU {
		t.Fatal("device kind lost")
	}
	x.SetF64(3, 0)
	x.Storage().Release()
	// reuse: next alloc of the same class should recycle
	y := p.Alloc(F64, 4).([]float64)
	if cap(y) != 4 {
		t.Fatalf("expected class-2 cap 4, got %d", cap(y))
	}
	if y[0] != 0 {
		t.Error("released+realloc must be zeroed")
	}
}

func TestViewsInheritDevice(t *testing.T) {
	dev := NewCPUDevice(NewPool())
	x := NewOn(dev, F64, Shape{2, 3})
	tr, _ := x.Transpose(0, 1)
	if tr.Device() != dev {
		t.Error("view must inherit device")
	}
}
