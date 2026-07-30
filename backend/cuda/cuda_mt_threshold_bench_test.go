//go:build cuda && cgo && linux

package cuda_test

import "testing"

// Small-M benches to tune each quant format's GEMV↔MT routing threshold (Q4_K/Q5_K/Q6_K already
// route at M>=2; these six still gate at M>=8). n=5632 is the ffn_up shape.
func BenchmarkIQ2XXSM2_5632(b *testing.B) { benchIQ2XXSM(b, 2, 2048, 5632) }
func BenchmarkIQ2XXSM4_5632(b *testing.B) { benchIQ2XXSM(b, 4, 2048, 5632) }
func BenchmarkIQ3SM2_5632(b *testing.B)   { benchIQ3SM(b, 2, 2048, 5632) }
func BenchmarkIQ3SM4_5632(b *testing.B)   { benchIQ3SM(b, 4, 2048, 5632) }
func BenchmarkIQ4XSM2_5632(b *testing.B)  { benchIQ4XSM(b, 2, 2048, 5632) }
func BenchmarkIQ4XSM4_5632(b *testing.B)  { benchIQ4XSM(b, 4, 2048, 5632) }
func BenchmarkMXFP4M2_5632(b *testing.B)  { benchMXFP4M(b, 2, 2048, 5632) }
func BenchmarkMXFP4M4_5632(b *testing.B)  { benchMXFP4M(b, 4, 2048, 5632) }
func BenchmarkQ2KM2_5632(b *testing.B)    { benchQ2KM(b, 2, 2048, 5632) }
func BenchmarkQ2KM4_5632(b *testing.B)    { benchQ2KM(b, 4, 2048, 5632) }
func BenchmarkQ3KM2_5632(b *testing.B)    { benchQ3KM(b, 2, 2048, 5632) }
func BenchmarkQ3KM4_5632(b *testing.B)    { benchQ3KM(b, 4, 2048, 5632) }
