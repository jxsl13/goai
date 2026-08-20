//go:build darwin && cgo && !arm64

package metal

import "github.com/jxsl13/goai/backend"

func expectedMeasuredHostUnaryMaxElements(backend.Op) int { return 0 }
