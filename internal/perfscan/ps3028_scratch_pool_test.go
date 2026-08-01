package main

import "testing"

func scratchPoolFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "unpooled-fully-overwritten-scratch" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3028_UnpooledFullyOverwrittenScratch is the measured shape: an attention backward's
// per-head contribution buffer, 16.7 MB at 8 heads and 512x512, freshly zeroed on every call and
// then written slot by slot with plain assignments.
func TestDetectPS3028_UnpooledFullyOverwrittenScratch(t *testing.T) {
	src := `package p

func bwd(heads, sq, sk int, score func(h, i, j int) float64) []float64 {
	buf := make([]float64, heads*sq*sk)
	for h := range heads {
		for i := range sq {
			base := (h*sq + i) * sk
			for j := range sk {
				buf[base+j] = score(h, i, j)
			}
		}
	}
	return buf
}`
	fs := scratchPoolFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Two things must reach the reader: that this is a resource win and not a speedup, so nobody
	// judges it on ns/op; and that the scanner cannot see coverage, so the overwrite has to be
	// proven before shipping.
	if !containsAll(fs[0].msg, "EXPECT NO SPEEDUP", "PROVE THE OVERWRITE") {
		t.Fatalf("message omits the resource framing or the proof instruction:\n%s", fs[0].msg)
	}
}

// TestDetectPS3028_SilentOnAccumulator pins the PLAIN-ASSIGNMENT requirement. One compound write
// makes the buffer an accumulator, and a recycled accumulator has to be cleared first — which is
// the very cost this check exists to remove, so recycling buys nothing there. The fixture differs
// from the positive by one character.
func TestDetectPS3028_SilentOnAccumulator(t *testing.T) {
	src := `package p

func bwd(heads, sq, sk int, score func(h, i, j int) float64) []float64 {
	buf := make([]float64, heads*sq*sk)
	for h := range heads {
		for i := range sq {
			base := (h*sq + i) * sk
			for j := range sk {
				buf[base+j] += score(h, i, j)
			}
		}
	}
	return buf
}`
	if fs := scratchPoolFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an accumulator needs clearing anyway:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3028_SilentOnSmallScratch pins the SIZE. A buffer sized by one or two dimensions is
// the ordinary per-call row or plane that every kernel allocates; zeroing it is not the cost, and
// reporting them would bury the one buffer that matters under dozens that do not.
func TestDetectPS3028_SilentOnSmallScratch(t *testing.T) {
	src := `package p

func bwd(sq, sk int, score func(i, j int) float64) []float64 {
	buf := make([]float64, sq*sk)
	for i := range sq {
		for j := range sk {
			buf[i*sk+j] = score(i, j)
		}
	}
	return buf
}`
	if fs := scratchPoolFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — two dimensions is an ordinary scratch", len(fs))
	}
}

// TestDetectPS3028_SilentWhenAlreadyPooled pins the POOL exclusion. A function that already talks
// to a pool has had this judgment made by whoever wrote it, and a second opinion from a scanner
// that cannot see which buffer is pooled is noise.
func TestDetectPS3028_SilentWhenAlreadyPooled(t *testing.T) {
	src := `package p

func bwd(heads, sq, sk int, score func(h, i, j int) float64) []float64 {
	rows := rowPool.Get().([]float64)
	buf := make([]float64, heads*sq*sk)
	for h := range heads {
		for i := range sq {
			base := (h*sq + i) * sk
			for j := range sk {
				buf[base+j] = score(h, i, j) + rows[j]
			}
		}
	}
	return buf
}`
	if fs := scratchPoolFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the function already recycles:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3028_SilentWhenNeverIndexWritten pins that the buffer must actually be written BY
// INDEX in this function. A buffer merely passed along proves nothing about coverage here, and
// advising a pool for it would be advice given without evidence.
func TestDetectPS3028_SilentWhenNeverIndexWritten(t *testing.T) {
	src := `package p

func bwd(heads, sq, sk int) []float64 {
	buf := make([]float64, heads*sq*sk)
	fill(buf, heads, sq, sk)
	return buf
}`
	if fs := scratchPoolFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing here shows the buffer is fully written", len(fs))
	}
}
