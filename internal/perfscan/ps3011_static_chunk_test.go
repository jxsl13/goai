package main

import "testing"

func staticChunkFindings(t *testing.T, src string) []finding {
	t.Helper()
	var out []finding
	for _, f := range scanSrc(t, src) {
		if f.category == "static-chunk-barrier" {
			out = append(out, f)
		}
	}
	return out
}

const staticChunkSrc = `package p

func run(d int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	chunk := (d + nw - 1) / nw
	var wg sync.WaitGroup
	for lo := 0; lo < d; lo += chunk {
		hi := lo + chunk
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}`

// TestDetectPS3011_StaticChunkBarrier is the measured shape: the autograd WKV VJP before its
// channels were claimed rather than dealt. Converting it went -28.73% and -29.58%, bit-identical.
func TestDetectPS3011_StaticChunkBarrier(t *testing.T) {
	fs := staticChunkFindings(t, staticChunkSrc)
	if len(fs) != 1 {
		t.Fatalf("%d findings, want 1", len(fs))
	}
	// The diagnostic is the actionable part: this site was nearly missed because a LINE profile
	// ranks the kernel and hides the waiting.
	if !containsAll(fs[0].msg, "GOMAXPROCS", "FUNCTION profile") {
		t.Fatalf("message omits the diagnostic:\n%s", fs[0].msg)
	}
}

// TestDetectPS3011_SilentOnAtomicClaim pins the applied form. A function that reaches for an atomic
// cursor has already had this done to it, and reporting it would flag the fix as the defect.
func TestDetectPS3011_SilentOnAtomicClaim(t *testing.T) {
	src := `package p

func run(d int, body func(c int)) {
	nw := runtime.GOMAXPROCS(0)
	chunk := (d + nw - 1) / nw
	_ = chunk
	var next atomic.Int64
	var wg sync.WaitGroup
	for range nw {
		go func() {
			defer wg.Done()
			for {
				c := int(next.Add(1)) - 1
				if c >= d {
					return
				}
				body(c)
			}
		}()
	}
	wg.Wait()
}`
	if fs := staticChunkFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — an atomic cursor is the applied form:\n%s", len(fs), fs[0].msg)
	}
}

// TestDetectPS3011_SilentWithoutSpawn keeps the check to work that is actually dispatched. A
// ceil-division on its own is just arithmetic — batching, tiling, pagination all use it.
func TestDetectPS3011_SilentWithoutSpawn(t *testing.T) {
	src := `package p

func tile(n, w int, body func(lo, hi int)) {
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		body(lo, lo+chunk)
	}
	wg.Wait()
}`
	if fs := staticChunkFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing is dispatched, so there is no barrier to unbalance", len(fs))
	}
}

// TestDetectPS3011_SilentOnSingleSpawn is the IN-A-LOOP floor. One goroutine started once is not a
// partition; the imbalance this check is about needs many chunks racing to a shared barrier.
func TestDetectPS3011_SilentOnSingleSpawn(t *testing.T) {
	src := `package p

func once(n, w int, body func(lo, hi int)) {
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		body(0, chunk)
	}()
	wg.Wait()
}`
	if fs := staticChunkFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a single spawn is not a static partition", len(fs))
	}
}

// TestDetectPS3011_SilentWithoutBarrier keeps the check to work that is JOINED. Fire-and-forget
// goroutines have no barrier, so no worker's slowness is on anyone else's critical path.
func TestDetectPS3011_SilentWithoutBarrier(t *testing.T) {
	src := `package p

func detach(d, nw int, body func(lo, hi int)) {
	chunk := (d + nw - 1) / nw
	for lo := 0; lo < d; lo += chunk {
		go body(lo, lo+chunk)
	}
}`
	if fs := staticChunkFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — nothing joins, so there is no barrier", len(fs))
	}
}

// TestDetectPS3011_SilentOnPlainDivision pins the ceil idiom specifically. An even split carries no
// `+ w - 1`, and matching bare division would flag every ratio in the tree.
func TestDetectPS3011_SilentOnPlainDivision(t *testing.T) {
	src := `package p

func even(d, nw int, body func(lo, hi int)) {
	chunk := d / nw
	var wg sync.WaitGroup
	for lo := 0; lo < d; lo += chunk {
		wg.Add(1)
		go func(lo int) {
			defer wg.Done()
			body(lo, lo+chunk)
		}(lo)
	}
	wg.Wait()
}`
	if fs := staticChunkFindings(t, src); len(fs) != 0 {
		t.Fatalf("%d findings, want 0 — a bare division is not the ceil-chunk idiom", len(fs))
	}
}

func containsAll(s string, subs ...string) bool {
	for _, sub := range subs {
		found := false
		for i := 0; i+len(sub) <= len(s); i++ {
			if s[i:i+len(sub)] == sub {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}
