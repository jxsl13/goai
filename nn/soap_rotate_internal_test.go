package nn

import (
	"math"
	"math/rand"
	"testing"
)

// rotateForwardRef and rotateBackRef are the SOAP basis rotations EXACTLY as they
// stood before the ikj rewrite: a serial dot accumulator per output element. They
// are the bit-identity oracles. Keep them frozen — editing them to match a new
// implementation would make the gate self-fulfilling.
func rotateForwardRef(ql, g, qr [][]float64) [][]float64 {
	m, n := len(ql), len(qr)
	t := zeroMat(m, n)
	for k := range m {
		for j := range n {
			var acc float64
			for i := range m {
				acc += ql[i][k] * g[i][j]
			}
			t[k][j] = acc
		}
	}
	out := zeroMat(m, n)
	for k := range m {
		for l := range n {
			var acc float64
			for j := range n {
				acc += t[k][j] * qr[j][l]
			}
			out[k][l] = acc
		}
	}
	return out
}

func rotateBackRef(ql, nmat, qr [][]float64) [][]float64 {
	m, n := len(ql), len(qr)
	t := zeroMat(m, n)
	for i := range m {
		for l := range n {
			var acc float64
			for k := range m {
				acc += ql[i][k] * nmat[k][l]
			}
			t[i][l] = acc
		}
	}
	out := zeroMat(m, n)
	for i := range m {
		for j := range n {
			var acc float64
			for l := range n {
				acc += t[i][l] * qr[j][l]
			}
			out[i][j] = acc
		}
	}
	return out
}

// randRows builds a deterministic [r,c] jagged matrix with a wide exponent spread,
// so any reassociation of the accumulation rounds differently and is caught.
func randRows(rng *rand.Rand, r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := range m {
		m[i] = make([]float64, c)
		for j := range m[i] {
			m[i][j] = rng.NormFloat64() * math.Pow(2, float64(rng.Intn(21)-10))
		}
	}
	return m
}

// rotateShapes are the (m,n) pairs the gates run over. m==n and m!=n both matter:
// the second product of each rotation is square in one index and rectangular in the
// other, and an ikj rewrite that confuses them would still pass on square input.
var rotateShapes = []struct{ m, n int }{{1, 1}, {2, 3}, {3, 2}, {4, 4}, {7, 5}, {5, 7}, {16, 16}, {13, 21}}

// TestRotateForwardCrossReferenceExact holds the rewritten rotation bit-identical to
// the pre-rewrite dot form, tolerance 0 (§V22) — the ikj order accumulates the same
// products per output in the same ascending order, so the bits must be equal.
func TestRotateForwardCrossReferenceExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260728))
	for _, d := range rotateShapes {
		ql, g, qr := randRows(rng, d.m, d.m), randRows(rng, d.m, d.n), randRows(rng, d.n, d.n)
		got, want := rotateForward(ql, g, qr), rotateForwardRef(ql, g, qr)
		assertRowsBitsEqual(t, got, want, "rotateForward", d.m, d.n)
	}
}

func TestRotateBackCrossReferenceExact(t *testing.T) {
	rng := rand.New(rand.NewSource(31415))
	for _, d := range rotateShapes {
		ql, nm, qr := randRows(rng, d.m, d.m), randRows(rng, d.m, d.n), randRows(rng, d.n, d.n)
		got, want := rotateBack(ql, nm, qr), rotateBackRef(ql, nm, qr)
		assertRowsBitsEqual(t, got, want, "rotateBack", d.m, d.n)
	}
}

// TestRotateIntoMatchesAllocatingTwin holds the pooled *Into variants bit-identical
// to their allocating twins — the comment on them claims exactly that, and it is the
// hot path SOAP actually runs. It also feeds DIRTY scratch: the ikj form accumulates
// with += instead of overwriting with =, so a missing zeroing step would leak the
// previous call's values, and only a dirty buffer catches that.
func TestRotateIntoMatchesAllocatingTwin(t *testing.T) {
	rng := rand.New(rand.NewSource(2718))
	for _, d := range rotateShapes {
		ql, g, qr := randRows(rng, d.m, d.m), randRows(rng, d.m, d.n), randRows(rng, d.n, d.n)

		out, tmp := randRows(rng, d.m, d.n), randRows(rng, d.m, d.n) // deliberately dirty
		rotateForwardInto(out, tmp, ql, g, qr)
		assertRowsBitsEqual(t, out, rotateForwardRef(ql, g, qr), "rotateForwardInto", d.m, d.n)

		out, tmp = randRows(rng, d.m, d.n), randRows(rng, d.m, d.n)
		rotateBackInto(out, tmp, ql, g, qr)
		assertRowsBitsEqual(t, out, rotateBackRef(ql, g, qr), "rotateBackInto", d.m, d.n)
	}
}

func assertRowsBitsEqual(t *testing.T, got, want [][]float64, what string, m, n int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s %dx%d: %d rows != %d", what, m, n, len(got), len(want))
	}
	for i := range want {
		for j := range want[i] {
			if math.Float64bits(got[i][j]) != math.Float64bits(want[i][j]) {
				t.Fatalf("%s %dx%d: [%d][%d] differs: got %v (%#x) want %v (%#x)",
					what, m, n, i, j, got[i][j], math.Float64bits(got[i][j]),
					want[i][j], math.Float64bits(want[i][j]))
			}
		}
	}
}

// shampooPrecondRef is Shampoo's Ĝ = L^{−1/4}·G·R^{−1/4} EXACTLY as it stood before
// the ikj rewrite: a serial dot accumulator per output. Frozen oracle — see the note
// on rotateForwardRef.
func shampooPrecondRef(li, gm, ri [][]float64, m, n int) [][]float64 {
	t := zeroMat(m, n)
	for i := range m {
		for j := range n {
			var acc float64
			for k := range n {
				acc += gm[i][k] * ri[k][j]
			}
			t[i][j] = acc
		}
	}
	out := zeroMat(m, n)
	for i := range m {
		for j := range n {
			var acc float64
			for k := range m {
				acc += li[i][k] * t[k][j]
			}
			out[i][j] = acc
		}
	}
	return out
}

// TestShampooPrecondCrossReferenceExact gates the Shampoo preconditioned gradient at
// tolerance 0. It exists because a deliberate reassociation of that product turned NO
// existing test in this package red — the path was bit-identity-blind, so the ikj
// rewrite's exactness claim had nothing holding it.
func TestShampooPrecondCrossReferenceExact(t *testing.T) {
	rng := rand.New(rand.NewSource(161803))
	for _, d := range rotateShapes {
		li, gm, ri := randRows(rng, d.m, d.m), randRows(rng, d.m, d.n), randRows(rng, d.n, d.n)
		gh, tmp := randRows(rng, d.m, d.n), randRows(rng, d.m, d.n) // deliberately dirty
		shampooPrecondInto(gh, tmp, li, gm, ri, d.m, d.n)
		assertRowsBitsEqual(t, gh, shampooPrecondRef(li, gm, ri, d.m, d.n), "shampooPrecond", d.m, d.n)
	}
}
