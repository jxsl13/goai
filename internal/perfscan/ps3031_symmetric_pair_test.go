package main

import "testing"

func symmetricPairFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "symmetric-pair-computed-twice" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3031_SymmetricPairComputedTwice is the measured shape: an eigh VJP's final loop
// forming G[i,j] and G[j,i] for every pair and storing half their sum, so half the cubic work was
// recomputing sums it had already made. Two kernels went about a third faster on this.
func TestDetectPS3031_SymmetricPairComputedTwice(t *testing.T) {
	src := `package p

func abar(v, tmp [][]float64, n int, out *T) {
	for i := range n {
		for j := range n {
			var g, gt float64
			for a := range n {
				g += v[i][a] * tmp[a][j]
				gt += v[j][a] * tmp[a][i]
			}
			out.SetF64(0.5*(g+gt), i, j)
		}
	}
}`
	fs := symmetricPairFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The commutativity argument is what licenses the transform, and the store requirement is what
	// keeps a reader from applying it where the mirror is not free.
	if !containsAll(fs[0].msg, "IEEE addition is commutative", "REQUIRED, not incidental") {
		t.Fatalf("message omits the licence or the precondition:\n%s", fs[0].msg)
	}
}

// TestDetectPS3031_SilentOnTriangle pins the applied form — the inner loop started at the outer
// index, both positions written.
func TestDetectPS3031_SilentOnTriangle(t *testing.T) {
	src := `package p

func abar(v, tmp [][]float64, n int, out *T) {
	for i := range n {
		for j := i; j < n; j++ {
			var g, gt float64
			for a := range n {
				g += v[i][a] * tmp[a][j]
				gt += v[j][a] * tmp[a][i]
			}
			m := 0.5 * (g + gt)
			out.SetF64(m, i, j)
			out.SetF64(m, j, i)
		}
	}
}`
	if fs := symmetricPairFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a triangle loop is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3031_SilentWithoutSymmetricStore pins the PRECONDITION, and it is the clause that
// took this check from 27 findings to 1. The fixture keeps the mirrored pair of accumulations and
// changes only what is written: a difference rather than a sum. The mirror is then not a duplicate
// — it is the other half of a quantity that is antisymmetric, and halving the work would change
// the result.
func TestDetectPS3031_SilentWithoutSymmetricStore(t *testing.T) {
	src := `package p

func skew(v, tmp [][]float64, n int, out *T) {
	for i := range n {
		for j := range n {
			var g, gt float64
			for a := range n {
				g += v[i][a] * tmp[a][j]
				gt += v[j][a] * tmp[a][i]
			}
			out.SetF64(0.5*(g-gt), i, j)
		}
	}
}`
	if fs := symmetricPairFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an antisymmetric store makes the mirror load-bearing:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3031_SilentOnUnmirroredAccumulators pins that the two sums must actually be each
// other's mirror. The fixture keeps two accumulators and a symmetric store, and changes one index
// so the second sum is an unrelated quantity — an unrolled band kernel looks like this, and an
// earlier version of the check reported several of them.
func TestDetectPS3031_SilentOnUnmirroredAccumulators(t *testing.T) {
	src := `package p

func band(v, tmp [][]float64, n int, out *T) {
	for i := range n {
		for j := range n {
			var g, gt float64
			for a := range n {
				g += v[i][a] * tmp[a][j]
				gt += v[j][a] * tmp[a][j]
			}
			out.SetF64(0.5*(g+gt), i, j)
		}
	}
}`
	if fs := symmetricPairFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the second sum is not the first one mirrored:\n%s",
			len(fs), fs[0].msg)
	}
}
