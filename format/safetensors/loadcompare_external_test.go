package safetensors_test

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/format/safetensors"
)

// TestSafetensorsLoadCompare times GoAI's full and partial safetensors load on the
// shared fixture the python companion (testdata/bench_safetensors_load.py) generates,
// so GoAI's pure-Go hostile-gated reader (V15/V29) can be compared head-to-head with
// the Rust-cored safetensors reference in BENCHMARKS.md. Gated on ST_BENCH_FILE; the
// value cross-check against the generator's pattern is the fairness anchor, not a
// correctness assertion. T885.
func TestSafetensorsLoadCompare(t *testing.T) {
	path := os.Getenv("ST_BENCH_FILE")
	if path == "" {
		t.Skip("set ST_BENCH_FILE to the shared fixture (python companion generates it)")
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	nbytes := fi.Size()

	// Full load, best of 7. The generator sets t0[0,0] = (0+0+0)%1000 = 0.
	fullBest := time.Hour
	var names int
	for range 7 {
		s := time.Now()
		m, _, err := safetensors.LoadFile(path)
		d := time.Since(s)
		if err != nil {
			t.Fatal(err)
		}
		names = len(m)
		if v := m["t0"].AtF64(0, 0); v != 0 {
			t.Fatalf("t0[0,0]=%v want 0 (fixture mismatch)", v)
		}
		if d < fullBest {
			fullBest = d
		}
	}

	// Partial load of the last tensor, best of 7 — GoAI seeks to that tensor's byte
	// range and reads nothing else. The generator sets t{k}[0,0] = k%1000.
	last := fmt.Sprintf("t%d", names-1)
	partBest := time.Hour
	var oneBytes int64
	for range 7 {
		s := time.Now()
		tn, err := safetensors.LoadTensor(path, last)
		d := time.Since(s)
		if err != nil {
			t.Fatal(err)
		}
		oneBytes = int64(tn.Numel()) * 4
		if v := tn.AtF64(0, 0); v != float64((names-1)%1000) {
			t.Fatalf("%s[0,0]=%v want %d", last, v, (names-1)%1000)
		}
		if d < partBest {
			partBest = d
		}
	}

	fmt.Printf("GOAI_LOAD full  best %.2f ms => %.1f MB/s (%d tensors, %d bytes)\n",
		float64(fullBest.Microseconds())/1000, float64(nbytes)/fullBest.Seconds()/1e6, names, nbytes)
	fmt.Printf("GOAI_LOAD one   best %.3f ms => %.1f MB/s (1 tensor from %d MB)\n",
		float64(partBest.Microseconds())/1000, float64(oneBytes)/partBest.Seconds()/1e6, nbytes/1_000_000)
}
