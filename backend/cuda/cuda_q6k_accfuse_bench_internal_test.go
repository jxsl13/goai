//go:build cuda && cgo && linux

package cuda

import "testing"

// benchQ6KResidualAcc measures the fused Q6_K decode-residual epilogue (QMatMulResidentAccQ6K,
// 1 kernel, no scratch) against the un-fused fallback recordAdd previously took for Q6_K weights
// (QMatMulResidentQ6K into scratch + a separate cu_add_f32 = 2 kernels + a scratch buffer). This
// is the Q6_K twin of the merged #930/#932 Q8/Q4_K residual fusion; Q4_K_M models store
// attn_v/ffn_down as Q6_K, so a non-Llama Q4_K_M served via the generic Decoder hit the fallback
// on every ffn-down residual. Fused does strictly less work, so it must not be slower.
func benchQ6KResidualAcc(b *testing.B, k, n int, fused bool) {
	if !Available() {
		b.Skip("no gpu")
	}
	raw := make([]byte, (k*n/256)*210) // Q6_K = 210 bytes / 256-weight super-block
	rq, err := NewResidentBQ6KFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	rec, err := NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	da, err := NewDeviceF32(1, k)
	if err != nil {
		b.Fatal(err)
	}
	dst, err := NewDeviceF32(1, n)
	if err != nil {
		b.Fatal(err)
	}
	scratch, err := NewDeviceF32(1, n)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { rq.Free(); rec.Free(); da.Free(); dst.Free(); scratch.Free() }()

	// Model one decode token: ~32 layers × 2 residual-add projections = 64 fused-eligible ops
	// batched into a single command buffer, then ONE Wait — the production pattern. A per-op Wait
	// instead makes each launch pay a full GPU sync, which dominates and masks the fusion delta.
	const opsPerToken = 64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < opsPerToken; j++ {
			if fused {
				if err := rec.QMatMulResidentAccQ6K(da, rq, dst, 1); err != nil {
					b.Fatal(err)
				}
			} else {
				if err := rec.QMatMulResidentQ6K(da, rq, scratch, 1); err != nil {
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

// ffn_down decode dims (Qwen2.5-3B-class: K=intermediate 11008, N=hidden 2048)
func BenchmarkQ6KResidualAccFusedDown(b *testing.B)   { benchQ6KResidualAcc(b, 11008, 2048, true) }
func BenchmarkQ6KResidualAccUnfusedDown(b *testing.B) { benchQ6KResidualAcc(b, 11008, 2048, false) }

// o_proj decode dims (K=N=hidden 2048)
func BenchmarkQ6KResidualAccFusedO(b *testing.B)   { benchQ6KResidualAcc(b, 2048, 2048, true) }
func BenchmarkQ6KResidualAccUnfusedO(b *testing.B) { benchQ6KResidualAcc(b, 2048, 2048, false) }

// benchQ5KResidualAcc is the Q5_K twin (Q5_K_M stores o_proj as Q5_K). Same scalar-GEMV+beta
// epilogue as Q6_K, so it should show the same ~1.8µs/op launch+round-trip saving.
func benchQ5KResidualAcc(b *testing.B, k, n int, fused bool) {
	if !Available() {
		b.Skip("no gpu")
	}
	raw := make([]byte, (k*n/256)*q5kBlockBytes)
	rq, err := NewResidentBQ5KFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	rec, err := NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	da, err := NewDeviceF32(1, k)
	if err != nil {
		b.Fatal(err)
	}
	dst, err := NewDeviceF32(1, n)
	if err != nil {
		b.Fatal(err)
	}
	scratch, err := NewDeviceF32(1, n)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { rq.Free(); rec.Free(); da.Free(); dst.Free(); scratch.Free() }()
	const opsPerToken = 64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < opsPerToken; j++ {
			if fused {
				if err := rec.QMatMulResidentAccQ5K(da, rq, dst, 1); err != nil {
					b.Fatal(err)
				}
			} else {
				if err := rec.QMatMulResidentQ5K(da, rq, scratch, 1); err != nil {
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

// o_proj decode dims (Q5_K_M's o_proj is Q5_K)
func BenchmarkQ5KResidualAccFusedO(b *testing.B)   { benchQ5KResidualAcc(b, 2048, 2048, true) }
func BenchmarkQ5KResidualAccUnfusedO(b *testing.B) { benchQ5KResidualAcc(b, 2048, 2048, false) }
