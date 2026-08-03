package main

import (
	"strings"
	"testing"
)

func compositeKeyMsgs(t *testing.T, src string) []string {
	t.Helper()
	var out []string
	for _, f := range scanSrc(t, src) {
		if f.category == "composite-key-map-probe" {
			out = append(out, f.msg)
		}
	}
	return out
}

// TestPS3004FiresOnCompositeKey is the positive floor. Two shapes because both occur in this tree: a
// named struct key (the backend's kernel dispatch table) and an array key (the BPE merge map).
//
// Neither fixture puts the probe inside a loop, which is the point — the measured site was a helper
// CALLED from a loop, and a loop gate would have missed it.
func TestPS3004FiresOnCompositeKey(t *testing.T) {
	for _, c := range []struct{ name, src, want string }{
		{"struct-key", `package p

func lookup(b *B, op Op, dt Dtype) (K, bool) {
	k, ok := b.table[kernelKey{op, dt}]
	return k, ok
}`, "kernelKey"},
		{"array-key", `package p

func rank(t *T, a, b string) int {
	if rk, ok := t.mergeRank[[2]string{a, b}]; ok {
		return rk
	}
	return 0
}`, "[2]string"},
	} {
		t.Run(c.name, func(t *testing.T) {
			msgs := compositeKeyMsgs(t, c.src)
			if len(msgs) != 1 {
				t.Fatalf("%d findings, want 1", len(msgs))
			}
			if !strings.Contains(msgs[0], c.want) {
				t.Fatalf("message does not name the key type %q: %s", c.want, msgs[0])
			}
			if !strings.Contains(msgs[0], "-38.19%") {
				t.Fatalf("message omits the measured result: %s", msgs[0])
			}
		})
	}
}

// Silence floors, one per clause.
func TestPS3004Silent(t *testing.T) {
	quiet := func(name, src string) {
		t.Run(name, func(t *testing.T) {
			if msgs := compositeKeyMsgs(t, src); len(msgs) != 0 {
				t.Fatalf("%s: expected silence, got: %s", name, msgs[0])
			}
		})
	}

	// CLAUSE: the index must be a COMPOSITE LITERAL. A plain string or int key takes Go's
	// specialized hasher and is not what this check is about — flagging it would cover most map
	// use in the tree.
	quiet("plain-key", `package p

func lookup(m map[string]int, k string) int {
	return m[k]
}`)

	// CLAUSE: it must be an INDEX expression. Constructing a composite literal and passing it
	// somewhere costs no hash at all.
	quiet("literal-not-indexed", `package p

func build(op Op, dt Dtype) kernelKey {
	k := kernelKey{op, dt}
	return k
}`)

	// CLAUSE: a slice or array indexed by an integer is not a map probe, whatever else the
	// function does with composite literals.
	quiet("slice-index", `package p

func get(s []int, i int) int {
	_ = kernelKey{}
	return s[i]
}`)
}
