//go:build darwin && cgo && !arm64

package metal

import "github.com/jxsl13/goai/backend"

// Unmeasured Darwin architectures preserve the direct Metal route.
func measuredHostUnaryMaxElements(backend.Op) int { return 0 }
