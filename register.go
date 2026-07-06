package goai

// Importing goai self-registers the always-available backends: the Pure-Go
// reference (source of truth + fallback, §V9/§I4) and the optimized cpu backend
// (preferred Default, §T11). Accel backends (GPU) are opt-in via build tags.
import (
	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)
