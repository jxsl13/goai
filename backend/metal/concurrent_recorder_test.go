//go:build darwin && cgo

package metal

import (
	"reflect"
	"testing"
)

func TestConcurrentRecorderToggleDependenciesAndBoundaries(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	previous := SetConcurrentDecodeRecorder(false)
	defer SetConcurrentDecodeRecorder(previous)
	control, err := NewConcurrentRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if control.concurrent {
		t.Fatal("disabled control unexpectedly opened a concurrent recorder")
	}
	control.Free()

	SetConcurrentDecodeRecorder(true)
	candidate, err := NewConcurrentRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if !candidate.concurrent {
		t.Fatal("enabled candidate did not open a concurrent recorder")
	}
	defer candidate.Free()

	in := []float32{-3, -1, 0, 2, 5}
	zeros := make([]float32, len(in))
	x, err := NewDeviceBufferF32(in)
	if err != nil {
		t.Fatal(err)
	}
	defer x.Release()
	tmp, err := NewDeviceBufferF32(zeros)
	if err != nil {
		t.Fatal(err)
	}
	defer tmp.Release()
	copyBuf, err := NewDeviceBufferF32(zeros)
	if err != nil {
		t.Fatal(err)
	}
	defer copyBuf.Release()
	out, err := NewDeviceBufferF32(zeros)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Release()

	if err := candidate.Unary(x, tmp, 0); err != nil { // neg
		t.Fatal(err)
	}
	if err := candidate.Barrier(); err != nil {
		t.Fatal(err)
	}
	if err := candidate.Unary(tmp, tmp, 8); err != nil { // abs; depends on neg
		t.Fatal(err)
	}
	if err := candidate.Blit(tmp, 0, copyBuf, 0, len(in)); err != nil { // closes compute encoder
		t.Fatal(err)
	}
	if err := candidate.Unary(copyBuf, out, 0); err != nil { // opens a new compute encoder after blit
		t.Fatal(err)
	}
	if err := candidate.Finish(); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(in))
	if err := out.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	want := []float32{-3, -1, 0, -2, -5}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("concurrent dependency/boundary result=%v want %v", got, want)
	}

	open, err := NewConcurrentRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := open.Unary(x, tmp, 0); err != nil {
		t.Fatal(err)
	}
	open.Free() // must close and release an uncommitted active encoder safely
}
