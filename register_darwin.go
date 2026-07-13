//go:build cgo

package goai

// On macOS with cgo — the default for a native `go build`/`go test` — the
// Metal/MPS GPU backend needs NO build tag: its frameworks ship with every macOS
// (§R47). Importing goai therefore auto-registers Metal, and backend.Default()
// selects it when a GPU is present (§T47) — zero build tags, zero selection code.
// A `CGO_ENABLED=0` build excludes this file, so the pure-Go path stays green
// (§V7).
import _ "github.com/jxsl13/goai/backend/metal"
