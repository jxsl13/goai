package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func queueingFanoutFindingsIn(t *testing.T, src string) []finding {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fanoutReg = map[string]map[string]bool{}
	collectFanoutHelpers([]*ast.File{f})
	var out []finding
	for _, fnd := range scanFile(fset, f, testSets(t)) {
		if fnd.category == "fanout-queues-jobs-it-does-not-need-to" {
			out = append(out, fnd)
		}
	}
	return out
}

// TestDetectPS3069_QueueingFanout is the measured shape: workers pulling from an unbuffered
// channel, with no path that skips the queue when the jobs already fit them.
func TestDetectPS3069_QueueingFanout(t *testing.T) {
	src := `package p

import "sync"

func parallelBuild(n int, work func(t int) error) error {
	workers := 8
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range jobs {
				_ = work(t)
			}
		}()
	}
	for t := 0; t < n; t++ {
		jobs <- t
	}
	close(jobs)
	wg.Wait()
	return nil
}`
	fs := queueingFanoutFindingsIn(t, src)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The honest expectation is the part a reader most needs: the profile share and the
	// measured payoff differ by more than an order of magnitude here.
	if !containsAll(fs[0].msg, "EXPECT THE SMALLER NUMBER, NOT THE PROFILE'S",
		"Parked threads are sampled", "AND RE-SWEEP THE WORK GATES AFTERWARDS") {
		t.Fatalf("message omits the honest expectation:\n%s", fs[0].msg)
	}
}

// TestDetectPS3069_SilentWithADirectPath pins the fix. A helper that spawns one goroutine per
// job when they already fit the workers has a second go statement and skips the queue.
func TestDetectPS3069_SilentWithADirectPath(t *testing.T) {
	src := `package p

import "sync"

func parallelBuild(n int, work func(t int) error) error {
	workers := 8
	var wg sync.WaitGroup
	if n <= workers {
		for t := 0; t < n; t++ {
			wg.Add(1)
			go func(t int) { defer wg.Done(); _ = work(t) }(t)
		}
		wg.Wait()
		return nil
	}
	jobs := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range jobs {
				_ = work(t)
			}
		}()
	}
	for t := 0; t < n; t++ {
		jobs <- t
	}
	close(jobs)
	wg.Wait()
	return nil
}`
	if fs := queueingFanoutFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — the direct path is there:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3069_SilentWithoutAQueue pins that the finding is about the CHANNEL. A helper
// that splits the range and spawns per chunk has no queue to skip.
func TestDetectPS3069_SilentWithoutAQueue(t *testing.T) {
	src := `package p

import "sync"

func parallelBands(n, work int, body func(lo, hi int)) {
	var wg sync.WaitGroup
	chunk := (n + 7) / 8
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); body(lo, hi) }(lo, hi)
	}
	wg.Wait()
}`
	if fs := queueingFanoutFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — no queue:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3069_SilentOnAPlainFunction pins that only REGISTERED fan-out helpers qualify —
// a function whose last parameter is not a callback over work is not one, however it dispatches.
func TestDetectPS3069_SilentOnAPlainFunction(t *testing.T) {
	src := `package p

import "sync"

func drain(n int, sink []int) {
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range jobs {
				sink[t] = t
			}
		}()
	}
	for t := 0; t < n; t++ {
		jobs <- t
	}
	close(jobs)
	wg.Wait()
}`
	if fs := queueingFanoutFindingsIn(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — not a fan-out helper:\n%s", len(fs), fs[0].msg)
	}
}
