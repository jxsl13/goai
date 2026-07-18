package backend

import (
	"fmt"
	"slices"
	"sync"
)

var (
	regMu     sync.RWMutex
	registry  = map[Name]Backend{}
	reference Backend // the Pure-Go truth backend; fallback target (§I4, §V9)

	// preference is the descending-performance order Default() auto-selects from:
	// the first REGISTERED name wins (§T46). Discrete-GPU cuBLAS beats Apple MPS
	// and portable Vulkan, all beat CPU-SIMD; the reference is the guaranteed final
	// fallback (never listed — it is tried after every preference entry misses).
	// Validated vs ONNX Runtime EP priority, PyTorch cuda>mps>cpu, and ggml's
	// scored backend registry (§R46), and by on-host benchmark: Vulkan matmul beat
	// CPU-SIMD 1.5×–3.4× on an Apple M2 Pro. NPUs are inference-only (§R45) → no
	// NPU op backend exists to list here. Override with SetPreference.
	preference = []Name{CUDA, Metal, Vulkan, CPU}
)

// Register adds b under its name. It panics on a duplicate or empty name —
// registration happens in init(), so a clash is a build-time programming error.
// Build-tagged accel backends call this from init() ONLY when their device is
// present, so registration doubles as runtime auto-detection (§I4, §V4): the
// registry reflects exactly what can run on this machine.
func Register(b Backend) {
	regMu.Lock()
	defer regMu.Unlock()
	name := b.Name()
	if name == "" {
		panic("backend: empty name")
	}
	if _, dup := registry[name]; dup {
		panic(fmt.Sprintf("backend: duplicate registration %q", name))
	}
	registry[name] = b
}

// RegisterReference registers b and marks it the reference backend: the source
// of numeric truth (§V9) and the guaranteed final fallback when no preferred
// backend is registered and when an active backend lacks a kernel (§I4).
func RegisterReference(b Backend) {
	Register(b)
	regMu.Lock()
	reference = b
	regMu.Unlock()
}

// RegisterDefault registers b and ensures its name participates in preference
// selection: if b's name is not already in the preference order it is appended
// (just before the reference fallback), so a registered optimized backend is
// always chosen over the bare reference. Selection among multiple registered
// backends is by preference order, NOT registration order (see Default,
// SetPreference). Kept for API stability (§V8).
func RegisterDefault(b Backend) {
	Register(b)
	regMu.Lock()
	defer regMu.Unlock()
	if !slices.Contains(preference, b.Name()) {
		preference = append(preference, b.Name())
	}
}

// Get returns the backend registered under name.
func Get(name Name) (Backend, bool) {
	regMu.RLock()
	defer regMu.RUnlock()
	b, ok := registry[name]
	return b, ok
}

// Reference returns the reference backend, or nil if none is registered.
func Reference() Backend {
	regMu.RLock()
	defer regMu.RUnlock()
	return reference
}

// Available lists the names of all registered backends (feature detection).
// Sorted for determinism. The pure-Go cpu backend always registers; the
// device-backed accel backends (metal, cuda, vulkan) are build-tagged and
// self-register only when their device is present, so this reflects what can
// actually run on this machine (§I4, §V4).
func Available() []Name {
	regMu.RLock()
	defer regMu.RUnlock()
	names := make([]Name, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	slices.Sort(names)
	return names
}

// Preference returns a copy of the current descending-performance selection order
// Default() uses. See SetPreference to change it.
func Preference() []Name {
	regMu.RLock()
	defer regMu.RUnlock()
	return append([]Name(nil), preference...)
}

// SetPreference overrides the descending-performance order Default() selects
// from. Names need not all be registered — unregistered names are skipped at
// selection time, so it is safe to list every backend you might build with (e.g.
// SetPreference(CUDA, CPU) to prefer CUDA but never the GPU-less Vulkan path).
// The reference backend remains the guaranteed final fallback regardless. Call it
// once at startup before creating contexts/tapes.
func SetPreference(order ...Name) {
	regMu.Lock()
	defer regMu.Unlock()
	preference = append([]Name(nil), order...)
}

// Default returns the backend to use when the caller does not choose one — the
// highest-preference REGISTERED backend (auto-detection: GPU backends register
// only when their device is present), else the reference. This is what
// NewContext and autograd.NewTape use, so simply building with an accel backend's
// tag makes real work run on the GPU with no code change. Panics if nothing is
// registered — a build must import at least backend/ref.
func Default() Backend {
	regMu.RLock()
	defer regMu.RUnlock()
	for _, name := range preference {
		if b, ok := registry[name]; ok {
			return b
		}
	}
	if reference != nil {
		return reference
	}
	panic("backend: no backend registered (import backend/ref)")
}
