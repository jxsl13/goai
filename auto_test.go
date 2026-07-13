//go:build darwin && cgo

package goai_test

import (
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"

	_ "github.com/jxsl13/goai" // importing the library is all it takes
)

// §T47: on macOS, a plain cgo build (no build tag) auto-registers the Metal
// backend simply by importing the library, and backend.Default() selects it —
// zero build tags, zero backend-selection code. This is the answer to "must the
// user pass build tags?": no, not on macOS.
func TestMetalAutoRegisteredWithoutTag(t *testing.T) {
	if !slices.Contains(backend.Available(), backend.Metal) {
		t.Skip("no Metal GPU on this machine — skipping (§V4)")
	}
	if got := backend.Default().Name(); got != backend.Metal {
		t.Errorf("Default() = %q, want metal (auto-selected, no -tags)", got)
	}
}
