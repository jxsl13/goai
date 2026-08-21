//go:build amd64 && goexperiment.simd

package simd

// AVX WKV groups four adjacent channels; splitting a group would compare the
// vector exponential with the scalar tail rather than the same range schedule.
var wkvRangeChunkSizes = [...]int{4, 8, 16, 64}
