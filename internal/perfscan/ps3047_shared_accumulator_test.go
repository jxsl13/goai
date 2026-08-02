package main

import "testing"

func sharedAccumFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "one-shared-accumulator-blocks-split" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3047_OneSharedGradient is the measured shape: an attention backward whose per-head
// gradients land in head-chosen columns and whose one decoupled-key gradient is accumulated by
// every head at the same address.
//
// The head offset is DERIVED — hc := h*dh — and the private writes index through it rather than
// through h. A syntactic test for the loop variable classifies those as shared, which inverts the
// finding and makes the check silent on exactly this shape.
func TestDetectPS3047_OneSharedGradient(t *testing.T) {
	src := `package p

func back(dq, dk, dv, dqR, dkR, q, k, v, g []float64, heads, dh, dR, seq, cols int) {
	for h := range heads {
		hc := h * dh
		for i := range seq {
			for j := range seq {
				dS := q[i]*k[j] + g[j]
				for d := range dh {
					dq[i*cols+hc+d] += dS * k[j*cols+hc+d]
					dk[j*cols+hc+d] += dS * q[i*cols+hc+d]
					dv[j*cols+hc+d] += dS * v[i*cols+hc+d]
				}
				for e := range dR {
					dqR[(i*heads+h)*dR+e] += dS
					dkR[j*dR+e] += dS
				}
			}
		}
	}
}`
	fs := sharedAccumFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The three things the transform needs: what to do, why it stays exact, and the two ways to
	// get it wrong — an unbounded recording buffer and a fold that loses the original bounds.
	if !containsAll(fs[0].msg, "RECORD AND FOLD", "BIT-IDENTICAL", "SIZE THE RECORDING BUFFER FIRST",
		"THE FOLD MUST REPRODUCE THE ORIGINAL BOUNDS") {
		t.Fatalf("message omits the transform, the exactness claim or a failure mode:\n%s", fs[0].msg)
	}
}

// TestDetectPS3047_SilentWhenNothingIsShared pins that the finding is about an EXCEPTION. A loop
// whose every accumulator is indexed by its dimension can fan out with no restructuring, and the
// record-and-fold machinery would be pure cost.
func TestDetectPS3047_SilentWhenNothingIsShared(t *testing.T) {
	src := `package p

func back(dq, dk, dv, q, k, v, g []float64, heads, dh, seq, cols int) {
	for h := range heads {
		hc := h * dh
		for i := range seq {
			for j := range seq {
				dS := q[i]*k[j] + g[j]
				for d := range dh {
					dq[i*cols+hc+d] += dS * k[j*cols+hc+d]
					dk[j*cols+hc+d] += dS * q[i*cols+hc+d]
					dv[j*cols+hc+d] += dS * v[i*cols+hc+d]
				}
			}
		}
	}
}`
	if fs := sharedAccumFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — every accumulator is already private:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3047_SilentOnAnOrdinaryReduction pins the other end. One private destination against
// several shared ones is not a loop held back by an exception; it is a reduction, and the answer
// there is a different transform.
func TestDetectPS3047_SilentOnAnOrdinaryReduction(t *testing.T) {
	src := `package p

func fold(acc, tot, cnt, dq, x []float64, heads, seq, dh int) {
	for h := range heads {
		for i := range seq {
			for d := range dh {
				dq[h*dh+d] += x[i]
				acc[d] += x[i]
				tot[d] += x[i] * x[i]
				cnt[d] += 1
			}
		}
	}
}`
	if fs := sharedAccumFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — three shared accumulators is a reduction:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3047_RoundedStoreCountsAsPrivate pins the f32 accumulator spelling. When every add
// must round through float32 the write is an explicit store, not +=, and reading only += left the
// check silent on the f32 arm of the function it was written for. The same pattern on the SHARED
// side would also match per-iteration scratch such as `a[j] = math.Exp(a[j] - m)`, so it counts
// only where the index names the dimension.
func TestDetectPS3047_RoundedStoreCountsAsPrivate(t *testing.T) {
	src := `package p

func back32(dq, dk, dv []float32, dqR, dkR []float64, a []float64, heads, dh, dR, seq, cols int) {
	for h := range heads {
		hc := h * dh
		for i := range seq {
			for j := range seq {
				a[j] = a[j] * 2
				dS := a[j]
				for d := range dh {
					dq[i*cols+hc+d] = float32(float64(dq[i*cols+hc+d]) + dS)
					dk[j*cols+hc+d] = float32(float64(dk[j*cols+hc+d]) + dS)
					dv[j*cols+hc+d] = float32(float64(dv[j*cols+hc+d]) + dS)
				}
				for e := range dR {
					dqR[(i*heads+h)*dR+e] += dS
					dkR[j*dR+e] += dS
				}
			}
		}
	}
}`
	if fs := sharedAccumFindingsIn(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1 — rounded stores are accumulations too", len(fs))
	}
}
