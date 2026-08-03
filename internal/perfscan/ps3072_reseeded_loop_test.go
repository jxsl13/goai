package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// reseededLoopFindingsIn primes the fan-out registry from the fixture itself, so each fixture
// is silent or not for a reason contained in its own source rather than in whatever an earlier
// test left in the package-level map.
func reseededLoopFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fanoutReg = map[string]map[string]bool{}
	fansOutReg = map[string]map[string]bool{}
	collectFanoutHelpers([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "serial-loop-reseeded-from-the-item" {
			out = append(out, fnd)
		}
	}
	return out
}

// fixture is the measured shape with one condition varied per test: a token loop that reseeds a
// generator from the previous token and does per-item work under it, in a package that has a
// fan-out helper this function never calls.
const reseedFixture = `package p

import (
	"math/rand/v2"
	"sync"
)

func parallelChunks(n, work int, body func(lo, hi int)) {
	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); body(0, n) }()
	wg.Wait()
}

func detect(key uint64, tokens []int, vocab, gsz int) int {
	perm := make([]int, vocab)
	pcg := rand.NewPCG(0, 0)
	rng := rand.New(pcg)
	green := 0
	for i := 1; i < len(tokens); i++ {
		pcg.Seed(key, uint64(tokens[i-1]))
		for k := 0; k < gsz; k++ {
			j := k + rng.IntN(vocab-k)
			perm[k], perm[j] = perm[j], perm[k]
			if perm[k] == tokens[i] {
				green++
				break
			}
		}
	}
	return green
}`

func TestDetectPS3072_ReseededSerialLoop(t *testing.T) {
	fs := reseededLoopFindingsIn(t, reseedFixture)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The banding instruction, what still crosses iterations, and the precondition the win
	// actually rests on — all three came out of the measurement.
	if !containsAll(fs[0].msg, "BAND IT", "integer addition is order-free", "CHECK THE RESTORE") {
		t.Fatalf("message omits the transform, the surviving carry or the precondition:\n%s",
			fs[0].msg)
	}
}

// TestDetectPS3072_SilentOnAConstantSeed pins the condition the whole finding rests on. A
// generator reseeded from a CONSTANT is genuinely carried forward: iteration i+1 sees the state
// iteration i left, and banding would change every value. Everything else here fires.
func TestDetectPS3072_SilentOnAConstantSeed(t *testing.T) {
	src := replaceOnce(t, reseedFixture,
		"pcg.Seed(key, uint64(tokens[i-1]))", "pcg.Seed(key, 99)")
	if fs := reseededLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a constant seed really is a chain:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3072_SilentWhenTheStateIsPerIteration pins that the object must be declared
// OUTSIDE the loop. A generator built inside the body carries nothing and there is nothing to
// say about it.
func TestDetectPS3072_SilentWhenTheStateIsPerIteration(t *testing.T) {
	src := replaceOnce(t, reseedFixture, `	pcg := rand.NewPCG(0, 0)
	rng := rand.New(pcg)
	green := 0
	for i := 1; i < len(tokens); i++ {
		pcg.Seed(key, uint64(tokens[i-1]))`, `	green := 0
	for i := 1; i < len(tokens); i++ {
		pcg := rand.NewPCG(0, 0)
		rng := rand.New(pcg)
		pcg.Seed(key, uint64(tokens[i-1]))`)
	if fs := reseededLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this state never leaves the iteration:\n%s",
			len(fs), fs[0].msg)
	}
}

// TestDetectPS3072_SilentWhenTheFunctionAlreadyFansOut pins the suppression. The reseed is still
// there and still per-item; the function has already made the choice this check argues for.
func TestDetectPS3072_SilentWhenTheFunctionAlreadyFansOut(t *testing.T) {
	src := replaceOnce(t, reseedFixture, "	green := 0\n	for i := 1;",
		"	green := 0\n	parallelChunks(len(tokens), len(tokens)*gsz, func(lo, hi int) {})\n	for i := 1;")
	if fs := reseededLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — this function already fans out:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3072_SilentOnAStraightLineBody pins the work floor. A reseed over a body with no
// inner loop costs a few nanoseconds an item, and a fork per band would not earn them back.
func TestDetectPS3072_SilentOnAStraightLineBody(t *testing.T) {
	src := replaceOnce(t, reseedFixture, `		for k := 0; k < gsz; k++ {
			j := k + rng.IntN(vocab-k)
			perm[k], perm[j] = perm[j], perm[k]
			if perm[k] == tokens[i] {
				green++
				break
			}
		}`, `		green += rng.IntN(vocab) + perm[0]`)
	if fs := reseededLoopFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no per-item work to band:\n%s", len(fs), fs[0].msg)
	}
}

// replaceOnce edits one fixture into another and fails if the anchor is not unique — a silent
// no-op edit produces a fixture identical to the firing one, which then passes for no reason.
func replaceOnce(t *testing.T, src, old, new string) string {
	t.Helper()
	if n := strings.Count(src, old); n != 1 {
		t.Fatalf("anchor occurs %d times, want 1: %q", n, old)
	}
	return strings.Replace(src, old, new, 1)
}
