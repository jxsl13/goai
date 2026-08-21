//go:build !(amd64 && goexperiment.simd)

package simd

// Apple arm64 NEON groups two adjacent channels. Portable builds are scalar and
// can exercise the same stricter boundary without changing their schedule.
var wkvRangeChunkSizes = [...]int{2, 4, 8, 16, 64}
