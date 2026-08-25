package tensor

import (
	"sync"
	"testing"
)

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
	// Under the race detector sync.Pool DROPS items at random by design (to shake
	// out reuse bugs), so backing-array identity is only guaranteed without it (§B51).
	if !raceEnabled && &b1[:1][0] != &b2[:1][0] {
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
	before := &x.Storage().F64()[0]
	x.Storage().Release()
	// reuse: next alloc of the same class should recycle
	y := p.Alloc(F64, 4).([]float64)
	if cap(y) != 4 {
		t.Fatalf("expected class-2 cap 4, got %d", cap(y))
	}
	if y[0] != 0 {
		t.Error("released+realloc must be zeroed")
	}
	if !raceEnabled && before != &y[0] {
		t.Error("Storage release must remain reusable through the public Pool path")
	}
	p.Free(y)
	z := NewOn(dev, F64, Shape{4})
	if z.Storage().pool == nil {
		t.Fatal("pooled F64 Storage must retain its release token")
	}
	if !raceEnabled && before != &z.Storage().F64()[0] {
		t.Error("public Pool release must remain reusable through the Storage path")
	}
	z.Storage().Release()
	z.Storage().Release() // idempotent even after returning a token
}

func TestPooledStorageReleaseTokenZeroesBothDtypes(t *testing.T) {
	p := NewPool()
	for _, dtype := range []Dtype{F32, F64} {
		t.Run(dtype.String(), func(t *testing.T) {
			first := newStorageWith(p, dtype, 5) // class 3, cap 8
			block := first.pool
			if block == nil {
				t.Fatal("pooled Storage did not retain a release token")
			}
			switch dtype {
			case F32:
				for i := range first.f32[:cap(first.f32)] {
					first.f32[:cap(first.f32)][i] = 17
				}
			case F64:
				for i := range first.f64[:cap(first.f64)] {
					first.f64[:cap(first.f64)][i] = -23
				}
			}
			first.Release()

			second := newStorageWith(p, dtype, 7)
			if !raceEnabled && second.pool != block {
				t.Error("same-class Storage allocation did not reuse its release token")
			}
			for i := 0; i < second.Len(); i++ {
				if got := second.atF64(i); got != 0 {
					t.Fatalf("reused Storage element %d = %g, want zero", i, got)
				}
			}
			second.Release()
		})
	}

	for _, dtype := range []Dtype{F16, BF16} {
		storage := newStorageWith(p, dtype, 8)
		if storage.pool != nil {
			t.Fatalf("%v must keep the shared-U16 non-pooled carve-out", dtype)
		}
	}
}

func TestPooledStorageReleaseTokenConcurrent(t *testing.T) {
	p := NewPool()
	var wg sync.WaitGroup
	for worker := 0; worker < 24; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			dtype := F32
			if worker%2 != 0 {
				dtype = F64
			}
			for iteration := 0; iteration < 500; iteration++ {
				storage := newStorageWith(p, dtype, 65+iteration%32)
				for i := 0; i < storage.Len(); i++ {
					if got := storage.atF64(i); got != 0 {
						t.Errorf("worker %d iteration %d element %d = %g, want zero", worker, iteration, i, got)
						storage.Release()
						return
					}
					storage.setF64(i, float64(worker+1))
				}
				storage.Release()
			}
		}(worker)
	}
	wg.Wait()
}

func TestViewsInheritDevice(t *testing.T) {
	dev := NewCPUDevice(NewPool())
	x := NewOn(dev, F64, Shape{2, 3})
	tr, _ := x.Transpose(0, 1)
	if tr.Device() != dev {
		t.Error("view must inherit device")
	}
}
