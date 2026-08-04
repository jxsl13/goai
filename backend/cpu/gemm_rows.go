package cpu

// gemmF32RowsPortable computes C rows [loRow,hiRow) of C[m,n] = A[m,k]·B[k,n],
// f32-native scalar, store semantics (rows are cleared then SAXPY-accumulated
// in ascending p order). It is the portable body behind the gemmF32Rows shim:
// the fallback on non-FMA amd64 and the (dead-code) default-build definition —
// the f32NativeKernels const gates all callers to the perf builds.
func gemmF32RowsPortable(A, B, C []float32, loRow, hiRow, k, n int) {
	gemmF32RowsColsPortable(A, B, C, loRow, hiRow, k, n, 0, n)
}

// gemmF32RowsColsPortable is gemmF32RowsPortable restricted to columns
// [jLo,jHi) — columns outside the span are left untouched.
func gemmF32RowsColsPortable(A, B, C []float32, loRow, hiRow, k, n, jLo, jHi int) {
	//perfscan:ignore PS3043 portable non-FMA/dead-code fallback; FMA build uses asm kernel
	for i := loRow; i < hiRow; i++ {
		ci := C[i*n+jLo : i*n+jHi]
		clear(ci)
		ai := A[i*k : (i+1)*k]
		//perfscan:ignore PS1007,PS3049 portable fallback; f32NativeKernels gates callers to asm | portable non-FMA/dead-code fallback; asm build is h
		for p := 0; p < k; p++ {
			ap := ai[p]
			bp := B[p*n+jLo : p*n+jHi]
			for j, bv := range bp {
				//perfscan:ignore PS3075 portable non-FMA/dead-code fallback; asm build is hot path
				ci[j] += ap * bv
			}
		}
	}
}
