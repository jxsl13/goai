package backend

import (
	"slices"
	"testing"
)

// §T46: Default() picks the highest-preference REGISTERED backend; unregistered
// preference names are skipped; the reference is the guaranteed final fallback.
func TestPreferenceSelection(t *testing.T) {
	saved := Preference()
	defer SetPreference(saved...)

	// Two fake accel backends "on this machine".
	gpuHi := &mockBackend{name: "gpu-hi", table: map[kernelKeyT]Kernel{}}
	gpuLo := &mockBackend{name: "gpu-lo", table: map[kernelKeyT]Kernel{}}
	Register(gpuHi)
	Register(gpuLo)

	// Prefer gpu-hi over gpu-lo over the (registered) reference.
	SetPreference("absent-cuda", "gpu-hi", "gpu-lo")
	if got := Default().Name(); got != "gpu-hi" {
		t.Errorf("Default = %q, want gpu-hi (first registered in preference)", got)
	}

	// Drop gpu-hi from preference → next registered wins.
	SetPreference("absent-cuda", "gpu-lo")
	if got := Default().Name(); got != "gpu-lo" {
		t.Errorf("Default = %q, want gpu-lo", got)
	}

	// No preference entry registered → guaranteed reference fallback.
	SetPreference("absent-cuda", "absent-metal")
	if got := Default().Name(); got != Reference().Name() {
		t.Errorf("Default = %q, want reference %q as fallback", got, Reference().Name())
	}

	// cleanup registry entries so other tests see a clean map
	delete(registry, "gpu-hi")
	delete(registry, "gpu-lo")
}

// The default preference order encodes descending performance and lists the CPU
// backend, with GPU backends ahead of it (validated §R46 + on-host benchmark).
func TestDefaultPreferenceOrder(t *testing.T) {
	want := []Name{CUDA, Metal, Vulkan, CPU}
	// A fresh process starts with exactly this order; other tests may have
	// appended via RegisterDefault, so assert the GPU→cpu prefix relationship.
	got := Preference()
	ci := slices.Index(got, CUDA)
	mi := slices.Index(got, Metal)
	vi := slices.Index(got, Vulkan)
	pi := slices.Index(got, CPU)
	if ci < 0 || mi < 0 || vi < 0 || pi < 0 {
		t.Fatalf("preference %v missing a canonical backend (want superset of %v)", got, want)
	}
	if !(ci < mi && mi < vi && vi < pi) {
		t.Errorf("preference order %v: want cuda<metal<vulkan<cpu (descending performance)", got)
	}
}

// RegisterDefault appends unknown names to preference so a custom optimized
// backend is chosen over the bare reference, without displacing GPU preference.
func TestRegisterDefaultAppends(t *testing.T) {
	saved := Preference()
	defer SetPreference(saved...)

	custom := &mockBackend{name: "custom-cpu", table: map[kernelKeyT]Kernel{}}
	RegisterDefault(custom)
	defer delete(registry, "custom-cpu")

	if !slices.Contains(Preference(), "custom-cpu") {
		t.Error("RegisterDefault must add the backend to the preference order")
	}
	// With only custom-cpu (of the appended names) registered and GPU names
	// absent, it beats the reference.
	SetPreference("absent-gpu", "custom-cpu")
	if got := Default().Name(); got != "custom-cpu" {
		t.Errorf("Default = %q, want custom-cpu over reference", got)
	}
}
