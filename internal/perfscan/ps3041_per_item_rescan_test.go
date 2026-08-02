package main

import "testing"

func perItemRescanFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "per-item-rescan-of-shared-collection" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3041_ScanOneCallDeep is the measured shape: a loop over query rows whose body
// calls a sibling method that walks the whole key bank. Neither the loop nor the callee mentions
// the other's data, so every row re-reads the same memory. Tiling it went -24%.
func TestDetectPS3041_ScanOneCallDeep(t *testing.T) {
	src := `package p

type mem struct{ keys [][]float64 }

func (m *mem) search(q []float64) int {
	best := -1
	for i := range m.keys {
		row := m.keys[i]
		var s float64
		for d := range row {
			s += q[d] * row[d]
		}
		if s > 0 {
			best = i
		}
	}
	return best
}

func (m *mem) gather(qs []float64, t, dim int, out []int) {
	for ti := 0; ti < t; ti++ {
		q := qs[ti*dim : ti*dim+dim]
		obase := ti * dim
		out[obase] = m.search(q)
	}
}`
	// The output is reached through an offset computed once, not out[ti]. That is how the
	// measured site is written, and it matters: with an out[ti] in the body the fixture passes
	// even if the check ignores per-item slice WINDOWS entirely, which is the form the real loop
	// uses to read its query row.
	fs := perItemRescanFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// Two things have to survive into the message: that the diagnosis is bandwidth and must be
	// confirmed before rewriting, and that the tile must not reassociate — that is what keeps the
	// existing goldens as the gate instead of forcing a tolerance argument.
	if !containsAll(fs[0].msg, "BANDWIDTH", "CONFIRM THE DIAGNOSIS FIRST", "MUST NOT REASSOCIATE") {
		t.Fatalf("message omits the diagnosis or the exactness condition:\n%s", fs[0].msg)
	}
}

// TestDetectPS3041_ScanInTheLoopItself pins the same shape without the call hop.
func TestDetectPS3041_ScanInTheLoopItself(t *testing.T) {
	src := `package p

type mem struct{ keys [][]float64 }

func (m *mem) gather(qs []float64, t, dim int, out []float64) {
	for ti := 0; ti < t; ti++ {
		q := qs[ti*dim : ti*dim+dim]
		var acc float64
		for i := range m.keys {
			row := m.keys[i]
			for d := range row {
				acc += q[d] * row[d]
			}
		}
		obase := ti * dim
		out[obase] = acc
	}
}`
	if fs := perItemRescanFindingsIn(t, src); len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
}

// TestDetectPS3041_SilentWhenTiled pins the APPLIED form. The block loop still calls the same
// scan over the same collection, so without a test for the stride the check would go on
// reporting the site after it had been fixed — the worst failure mode a scan rule has, because
// it teaches the reader to ignore it.
func TestDetectPS3041_SilentWhenTiled(t *testing.T) {
	src := `package p

type mem struct{ keys [][]float64 }

func (m *mem) searchTile(qt [][]float64, out []int) {
	for i := range m.keys {
		row := m.keys[i]
		for j, q := range qt {
			var s float64
			for d := range row {
				s += q[d] * row[d]
			}
			if s > 0 {
				out[j] = i
			}
		}
	}
}

func (m *mem) gather(qs []float64, t, dim int, out []int) {
	const b = 16
	qt := make([][]float64, b)
	for base := 0; base < t; base += b {
		for j := range qt {
			ti := base + j
			qt[j] = qs[ti*dim : ti*dim+dim]
		}
		m.searchTile(qt, out[base:])
	}
}`
	if fs := perItemRescanFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a block loop is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3041_SilentOnRangeOverIntField pins the narrowing that made the check usable. A
// receiver field ranged over as an INTEGER — `for d := range m.dim` — is not a collection walk,
// and the first version reported eight of those for every real finding because it only asked
// whether the body indexed SOMETHING by the range key. The collection's own elements must be read.
func TestDetectPS3041_SilentOnRangeOverIntField(t *testing.T) {
	src := `package p

type layer struct{ dim int }

func (l *layer) run(xs []float64, n int, out []float64) {
	for i := 0; i < n; i++ {
		x := xs[i*l.dim : i*l.dim+l.dim]
		var s float64
		for d := range l.dim {
			s += x[d] * x[d]
		}
		out[i] = s
	}
}`
	if fs := perItemRescanFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a range over an int field is not a collection walk:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3041_SilentOnParameterCollection pins that the re-streamed data must be state the
// RECEIVER holds. A collection passed in as a parameter is sized by the caller and is usually the
// per-call working set, not a bank every item re-reads; reporting those would bury the finding
// this check exists for.
func TestDetectPS3041_SilentOnParameterCollection(t *testing.T) {
	src := `package p

type mem struct{ n int }

func (m *mem) gather(keys [][]float64, qs []float64, t, dim int, out []float64) {
	for ti := 0; ti < t; ti++ {
		q := qs[ti*dim : ti*dim+dim]
		var acc float64
		for i := range keys {
			row := keys[i]
			for d := range row {
				acc += q[d] * row[d]
			}
		}
		out[ti] = acc
	}
}`
	if fs := perItemRescanFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a parameter collection is not shared receiver state:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3041_SilentWithoutPerItemWork records a KNOWN BLIND SPOT rather than a design goal.
// The check demands that the item loop index or slice something by its variable, which is how it
// tells per-item work from a counter. A loop that reaches its data only through METHOD CALLS
// taking the index — the AtF64/SetF64 fallback arm of the site this was measured on — therefore
// reads as no per-item work and is not reported. Widening to "the variable appears in any call
// argument" was rejected: it fires on every bounds-checked helper in the tree.
func TestDetectPS3041_SilentWithoutPerItemWork(t *testing.T) {
	src := `package p

type mem struct{ keys [][]float64 }

func (m *mem) at(i, d int) float64 { return m.keys[i][d] }

func (m *mem) gather(q *mem, t, dim int, out *mem) {
	for ti := 0; ti < t; ti++ {
		var acc float64
		for i := range m.keys {
			row := m.keys[i]
			for d := range row {
				acc += q.at(ti, d) * row[d]
			}
		}
		out.set(ti, acc)
	}
}

func (m *mem) set(i int, v float64) { m.keys[i][0] = v }`
	if fs := perItemRescanFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the documented blind spot has closed, update this test:\n%s",
			len(fs), fs[0].msg)
	}
}
