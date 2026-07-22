package tensor

import (
	"math/rand"
	"testing"
)

// BenchmarkCastF16toF32Varied casts a 512×512 F16 tensor of VARIED values (real f16
// weights, not the zero tensor New gives) — the honest test of the lookup table under
// random 256 KiB access, not the all-L1 zero case.
func BenchmarkCastF16toF32Varied(b *testing.B) {
	x := New(F16, Shape{512, 512})
	u := x.storage.U16()
	rng := rand.New(rand.NewSource(1))
	for i := range u {
		u[i] = f32ToF16(rng.Float32()*4 - 2) // finite f16 in [-2,2)
	}
	b.ResetTimer()
	for range b.N {
		_ = x.Cast(F32)
	}
}
