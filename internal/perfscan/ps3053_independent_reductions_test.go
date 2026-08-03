package main

import "testing"

func indepReductionFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "independent-reductions-one-at-a-time" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3053_PerItemDotOverASharedRow is the measured shape: a scan that walks one row of
// stored data at a time and, for each query, reduces that row into a scalar with its own
// accumulator.
func TestDetectPS3053_PerItemDotOverASharedRow(t *testing.T) {
	src := `package p

type ent struct {
	i int
	s float64
}

func scan(keys [][]float64, qt [][]float64, out []ent) {
	for i := range keys {
		row := keys[i]
		for j, q := range qt {
			var s float64
			for d, rv := range row {
				s += q[d] * rv
			}
			out[j] = ent{i, s}
		}
	}
}`
	fs := indepReductionFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The message must say why this is exact where PS3010's version of the same observation is
	// not, and it must warn about what the gate can miss — both came out of the measurement.
	if !containsAll(fs[0].msg, "TAKE FOUR ITEMS PER PASS", "BIT-IDENTICAL", "PS3010",
		"DESIGN THE MUTATION THAT GATES IT WITH CARE") {
		t.Fatalf("message omits the fix, the exactness distinction or the gate warning:\n%s", fs[0].msg)
	}
}

// TestDetectPS3053_SilentWithoutASharedOuterRow pins the nesting the finding rests on. A single
// reduction with no sibling to interleave with has nothing to gain: the chains it would overlap
// do not exist. Without this condition the check reported 111 sites tree-wide instead of 18.
func TestDetectPS3053_SilentWithoutASharedOuterRow(t *testing.T) {
	src := `package p

func norms(rows [][]float64, out []float64) {
	for i, row := range rows {
		var s float64
		for _, rv := range row {
			s += rv * rv
		}
		out[i] = s
	}
}`
	if fs := indepReductionFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — one reduction per row has no sibling:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3053_SilentWhenTheSourceIsPerItem pins the other half. If each item reduces over its
// OWN data rather than a shared row, interleaving buys no reuse of the source — the loads are
// distinct — and the transform is only the register pressure.
func TestDetectPS3053_SilentWhenTheSourceIsPerItem(t *testing.T) {
	src := `package p

func pairs(keys [][]float64, qt [][]float64, out []float64) {
	for i := range keys {
		base := keys[i]
		_ = base
		for j, q := range qt {
			var s float64
			for d, qv := range q {
				s += qv * float64(d)
			}
			out[j] = s
		}
	}
}`
	if fs := indepReductionFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the source belongs to the item:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3053_SilentWhenAlreadyGrouped pins the APPLIED form: the item loop advancing by more
// than one, which is how several accumulators end up interleaved.
func TestDetectPS3053_SilentWhenAlreadyGrouped(t *testing.T) {
	src := `package p

func scan(keys [][]float64, qt [][]float64, out []float64) {
	for i := range keys {
		row := keys[i]
		for j := 0; j+3 < len(qt); j += 4 {
			var s0, s1 float64
			for d, rv := range row {
				s0 += qt[j+0][d] * rv
				s1 += qt[j+1][d] * rv
			}
			out[j+0], out[j+1] = s0, s1
		}
		_ = i
	}
}`
	if fs := indepReductionFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the items are already taken in groups:\n%s", len(fs), fs[0].msg)
	}
}
