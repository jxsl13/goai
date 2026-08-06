//go:build cuda && cgo && linux

package cuda

import "testing"

// benchQ4KResidualAccSmallM measures the SMALL-BATCH (m>1) residual epilogue two ways, to test
// whether recordAdd should extend the fused Acc path past m==1 now that Acc routes to the
// weight-read-once MT (#1013):
//
//	fused   = QMatMulResidentAccQ4K (MT beta=1, ONE launch into dst)
//	unfused = QMatMulResidentQ4K into scratch (MT beta=0) + a separate Binary add (the split
//	          recordAdd currently takes for m>1)
//
// Both route the GEMM to the SAME MT kernel; the only difference is the fused path skips the add
// launch + the scratch HBM round-trip. Batched 64 ops/Wait (production decode pattern) so the
// per-op launch cost isn't masked by a full GPU sync per op.
func benchQ4KResidualAccSmallM(b *testing.B, m, k, n int, fused bool) {
	if !Available() {
		b.Skip("no gpu")
	}
	raw := make([]byte, (k*n/256)*144) // GGUF Q4_K = 144 bytes / 256-weight super-block
	rq, err := NewResidentBQ4KFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	rec, err := NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	da, _ := NewDeviceF32(m, k)
	dst, _ := NewDeviceF32(m, n)
	scratch, _ := NewDeviceF32(m, n)
	defer func() { rq.Free(); rec.Free(); da.Free(); dst.Free(); scratch.Free() }()
	const opsPerToken = 64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < opsPerToken; j++ {
			if fused {
				if err := rec.QMatMulResidentAccQ4K(da, rq, dst, m); err != nil {
					b.Fatal(err)
				}
			} else {
				if err := rec.QMatMulResidentQ4K(da, rq, scratch, m); err != nil {
					b.Fatal(err)
				}
				if err := rec.Binary(dst, scratch, dst, recBinaryAdd); err != nil {
					b.Fatal(err)
				}
			}
		}
		if err := rec.Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

// o_proj decode dims (K=N=hidden 2048). m=2 is the LOW end of the m<6 fused band — MT least amortized.
func BenchmarkQ4KAccSmallMFusedO_2(b *testing.B) { benchQ4KResidualAccSmallM(b, 2, 2048, 2048, true) }
func BenchmarkQ4KAccSmallMUnfusedO_2(b *testing.B) {
	benchQ4KResidualAccSmallM(b, 2, 2048, 2048, false)
}

// o_proj decode dims (K=N=hidden 2048), small batch m=4 (speculative / small-batch decode)
func BenchmarkQ4KAccSmallMFusedO_4(b *testing.B) { benchQ4KResidualAccSmallM(b, 4, 2048, 2048, true) }
func BenchmarkQ4KAccSmallMUnfusedO_4(b *testing.B) {
	benchQ4KResidualAccSmallM(b, 4, 2048, 2048, false)
}

// ffn_down decode dims (K=intermediate 5632, N=hidden 2048)
func BenchmarkQ4KAccSmallMFusedDown_4(b *testing.B) {
	benchQ4KResidualAccSmallM(b, 4, 5632, 2048, true)
}
func BenchmarkQ4KAccSmallMUnfusedDown_4(b *testing.B) {
	benchQ4KResidualAccSmallM(b, 4, 5632, 2048, false)
}
