package backend

import (
	"fmt"
	"sort"
	"sync"
)

var (
	regMu     sync.RWMutex
	registry  = map[string]Backend{}
	reference Backend // the Pure-Go truth backend; fallback target (§I4, §V9)
	preferred Backend // optimized default, if registered (e.g. cpu); overrides reference
)

// Register adds b under its name. It panics on a duplicate or empty name —
// registration happens in init(), so a clash is a build-time programming error.
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
// of numeric truth (§V9) and the fallback target when an active backend lacks a
// kernel (§I4).
func RegisterReference(b Backend) {
	Register(b)
	regMu.Lock()
	reference = b
	regMu.Unlock()
}

// RegisterDefault registers b and marks it the preferred Default — the optimized
// backend real programs use (e.g. cpu). The reference remains the fallback target
// (§I4). Last registration wins.
func RegisterDefault(b Backend) {
	Register(b)
	regMu.Lock()
	preferred = b
	regMu.Unlock()
}

// Get returns the backend registered under name.
func Get(name string) (Backend, bool) {
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
// Sorted for determinism. Build-tagged accel backends self-register in init(),
// so this reflects what can actually run on this machine (§I4, §V4).
func Available() []string {
	regMu.RLock()
	defer regMu.RUnlock()
	names := make([]string, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Default returns the backend to use when the caller does not choose one: the
// preferred optimized backend if one registered (§I4), else the reference.
// Panics if nothing is registered — a build must import at least the reference.
func Default() Backend {
	regMu.RLock()
	defer regMu.RUnlock()
	if preferred != nil {
		return preferred
	}
	if reference != nil {
		return reference
	}
	panic("backend: no backend registered (import backend/ref)")
}
