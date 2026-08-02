package main

import "testing"

func recursiveSplitFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "recursive-split-alloc" {
			out = append(out, f)
		}
	}
	return out
}

// TestDetectPS3039_RecursiveSplitAlloc is the measured shape: a CART builder allocating both sides
// of every split and recursing into them. Those two allocations are a per-node cost, so they scale
// with the tree; replacing them with an in-place partition took a forest fit's allocations down
// 45.5% and its bytes 63.9%.
func TestDetectPS3039_RecursiveSplitAlloc(t *testing.T) {
	src := `package p

func (b *builder) buildIdx(idx []int, depth int) *node {
	n := len(idx)
	nd := &node{leaf: true}
	feat, thr, ok := b.bestSplit(idx)
	if !ok {
		return nd
	}
	left := make([]int, 0, n)
	right := make([]int, 0, n)
	for _, i := range idx {
		if b.x[i][feat] <= thr {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	nd.left = b.buildIdx(left, depth+1)
	nd.right = b.buildIdx(right, depth+1)
	return nd
}`
	fs := recursiveSplitFindings(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The safety argument and the gating instruction both have to survive: the transform is only
	// obviously correct once you see why the in-place write cannot clobber, and it is only
	// verifiable against a golden, since the usual property tests pass for a different tree.
	if !containsAll(fs[0].msg, "cannot clobber an unread element", "EXACT GOLDEN", "per NODE") {
		t.Fatalf("message omits the safety argument, the gate or the scaling:\n%s", fs[0].msg)
	}
}

// TestDetectPS3039_SilentOnAppliedForm pins the applied form: one reused buffer, an in-place
// compaction, and two subslices handed to the recursion.
func TestDetectPS3039_SilentOnAppliedForm(t *testing.T) {
	src := `package p

func (b *builder) buildIdx(idx []int, depth int) *node {
	n := len(idx)
	nd := &node{leaf: true}
	feat, thr, ok := b.bestSplit(idx)
	if !ok {
		return nd
	}
	if cap(b.partIdx) < n {
		b.partIdx = make([]int, n)
	}
	rbuf := b.partIdx[:n]
	mid, r := 0, 0
	for _, i := range idx {
		if b.x[i][feat] <= thr {
			idx[mid] = i
			mid++
		} else {
			rbuf[r] = i
			r++
		}
	}
	copy(idx[mid:], rbuf[:r])
	nd.left = b.buildIdx(idx[:mid], depth+1)
	nd.right = b.buildIdx(idx[mid:], depth+1)
	return nd
}`
	if fs := recursiveSplitFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the partition is already in place:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3039_SilentWithoutRecursion pins that the calls must be RECURSIVE. Two buffers built
// once and passed to some other function cost once, not once per node, and the whole argument for
// this transform is that the cost multiplies with the recursion.
func TestDetectPS3039_SilentWithoutRecursion(t *testing.T) {
	src := `package p

func split(x [][]float64, idx []int, feat int, thr float64) (*node, *node) {
	n := len(idx)
	left := make([]int, 0, n)
	right := make([]int, 0, n)
	for _, i := range idx {
		if x[i][feat] <= thr {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	return leafOf(left), leafOf(right)
}`
	if fs := recursiveSplitFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing recurses:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3039_SilentOnASingleBuffer pins that BOTH sides must be allocated. A recursion that
// builds one buffer and reuses its input for the other is already half in place, and the in-place
// rewrite has nothing left to remove.
func TestDetectPS3039_SilentOnASingleBuffer(t *testing.T) {
	src := `package p

func (b *builder) walk(idx []int, depth int) *node {
	nd := &node{}
	sel := make([]int, 0, len(idx))
	for _, i := range idx {
		if b.keep(i) {
			sel = append(sel, i)
		}
	}
	nd.child = b.walk(sel, depth+1)
	return nd
}`
	if fs := recursiveSplitFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — only one side is allocated:\n%s", len(fs), fs[0].msg)
	}
}
