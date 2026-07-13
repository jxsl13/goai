//go:build vulkan && cgo

package vulkan

import (
	"testing"
)

// §T381 (ADR-0019 Phase 2, Vulkan port foundation, mirrors Metal §T367): a device-resident buffer
// round-trips (upload → download) and UploadF32 overwrites in place — the storage primitive the
// Vulkan Recorder will chain ops over so intermediates stay resident (de-risked in §T380).
func TestVulkanDeviceBufferRoundTrip(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const n = 4096
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i)*0.5 - 3.0
	}
	b, err := NewDeviceBufferF32(src)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Release()
	if b.Len() != n {
		t.Fatalf("Len %d != %d", b.Len(), n)
	}
	got := make([]float32, n)
	if err := b.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("round-trip [%d]: %v != %v", i, got[i], src[i])
		}
	}
	// UploadF32 overwrites in place (buffer reuse across decode steps).
	src2 := make([]float32, n)
	for i := range src2 {
		src2[i] = float32(n-i) * 0.25
	}
	if err := b.UploadF32(src2); err != nil {
		t.Fatal(err)
	}
	if err := b.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for i := range src2 {
		if got[i] != src2[i] {
			t.Fatalf("after upload [%d]: %v != %v", i, got[i], src2[i])
		}
	}
}
