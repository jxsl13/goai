package cpu_test

import "testing"

// Large-expert-count MoE combine: the per-(token) normalized-weight division is invariant across the
// model dim, so hoisting it out of the d-loop saves (d−1)·e divisions/token — a win that scales with e
// (DeepSeek-V3 = 256 experts). Marginal at e=8 (memory-bound 8-way gather), ~7-8% at e=256.
func BenchmarkMoECombine_e64(b *testing.B)  { benchMoECombine(b, 512, 2048, 64, false) }
func BenchmarkMoECombine_e256(b *testing.B) { benchMoECombine(b, 256, 2048, 256, false) }
