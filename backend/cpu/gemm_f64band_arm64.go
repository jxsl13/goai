//go:build arm64 && goexperiment.simd

package cpu

// gemmF64Band keeps a 4x2 output tile in scalar FP registers for the complete
// ascending-p reduction. This isolates the store-once loop restructuring from
// the NEON tile that may replace it after independent measurement.
func gemmF64Band(A, B, C []float64, loRow, hiRow, k, n int) {
	if k == 0 || loRow >= hiRow || n == 0 {
		return
	}
	if n == 1 {
		gemmF64BandPortable(A, B, C, loRow, hiRow, k, n)
		return
	}

	// Keep one k-by-nc panel of B near the M2 performance cores' private L2.
	// The split is only over independent columns; each output still visits p in
	// exactly the portable order.
	const l2Target = 256 << 10
	nc := n
	if k*n*8 > l2Target {
		nc = l2Target / (8 * k) / 2 * 2
		if nc < 2 {
			nc = 2
		}
	}
	for jb := 0; jb < n; jb += nc {
		gemmF64BandColsArm64Go(A, B, C, loRow, hiRow, k, n, jb, min(jb+nc, n))
	}
}

func gemmF64BandColsArm64Go(A, B, C []float64, loRow, hiRow, k, n, jLo, jHi int) {
	i := loRow
	for ; i+3 < hiRow; i += 4 {
		ar0 := A[(i+0)*k : (i+1)*k]
		ar1 := A[(i+1)*k : (i+2)*k]
		ar2 := A[(i+2)*k : (i+3)*k]
		ar3 := A[(i+3)*k : (i+4)*k]
		c0 := C[(i+0)*n : (i+1)*n]
		c1 := C[(i+1)*n : (i+2)*n]
		c2 := C[(i+2)*n : (i+3)*n]
		c3 := C[(i+3)*n : (i+4)*n]

		j := jLo
		for ; j+1 < jHi; j += 2 {
			v00, v01 := c0[j], c0[j+1]
			v10, v11 := c1[j], c1[j+1]
			v20, v21 := c2[j], c2[j+1]
			v30, v31 := c3[j], c3[j+1]
			bo := j
			for p := 0; p < k; p++ {
				b0, b1 := B[bo], B[bo+1]
				v00 += ar0[p] * b0
				v01 += ar0[p] * b1
				v10 += ar1[p] * b0
				v11 += ar1[p] * b1
				v20 += ar2[p] * b0
				v21 += ar2[p] * b1
				v30 += ar3[p] * b0
				v31 += ar3[p] * b1
				bo += n
			}
			c0[j], c0[j+1] = v00, v01
			c1[j], c1[j+1] = v10, v11
			c2[j], c2[j+1] = v20, v21
			c3[j], c3[j+1] = v30, v31
		}
		if j < jHi {
			v0, v1, v2, v3 := c0[j], c1[j], c2[j], c3[j]
			bo := j
			for p := 0; p < k; p++ {
				bv := B[bo]
				v0 += ar0[p] * bv
				v1 += ar1[p] * bv
				v2 += ar2[p] * bv
				v3 += ar3[p] * bv
				bo += n
			}
			c0[j], c1[j], c2[j], c3[j] = v0, v1, v2, v3
		}
	}

	for ; i < hiRow; i++ {
		ar := A[i*k : (i+1)*k]
		ci := C[i*n : (i+1)*n]
		j := jLo
		for ; j+1 < jHi; j += 2 {
			v0, v1 := ci[j], ci[j+1]
			bo := j
			for p := 0; p < k; p++ {
				v0 += ar[p] * B[bo]
				v1 += ar[p] * B[bo+1]
				bo += n
			}
			ci[j], ci[j+1] = v0, v1
		}
		if j < jHi {
			v := ci[j]
			bo := j
			for p := 0; p < k; p++ {
				v += ar[p] * B[bo]
				bo += n
			}
			ci[j] = v
		}
	}
}
