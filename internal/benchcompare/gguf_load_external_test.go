package benchcompare

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/format/gguf"
)

// TestGGUFLoadCompare times GoAI's full GGUF load on the shared fixture the python
// companion (testdata/bench_gguf_load.py) writes, so GoAI's pure-Go hostile-gated
// GGUF reader can be compared head-to-head with gguf-py's mmap-based GGUFReader in
// BENCHMARKS.md. Untagged (pure-Go read, no backend needed) and gated on
// GGUF_BENCH_FILE; the value cross-check against the generator pattern is the
// fairness anchor. The GGUF half of T885 (safetensors half: format/safetensors).
func TestGGUFLoadCompare(t *testing.T) {
	path := os.Getenv("GGUF_BENCH_FILE")
	if path == "" {
		t.Skip("set GGUF_BENCH_FILE to the shared fixture (python companion generates it)")
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	nbytes := fi.Size()

	// gguf.ReadFile parses the header and materializes every tensor to F32; the F32
	// fixture means no dequant on either side (a pure load comparison). best of 7.
	best := time.Hour
	var names int
	for range 7 {
		s := time.Now()
		f, err := gguf.ReadFile(path)
		d := time.Since(s)
		if err != nil {
			t.Fatal(err)
		}
		names = len(f.Tensors)
		if v := f.Tensors["t0"].AtF64(0, 0); v != 0 {
			t.Fatalf("t0[0,0]=%v want 0 (fixture mismatch)", v)
		}
		if d < best {
			best = d
		}
	}
	fmt.Printf("GOAI_GGUF_LOAD full best %.2f ms => %.1f MB/s (%d tensors, %d bytes)\n",
		float64(best.Microseconds())/1000, float64(nbytes)/best.Seconds()/1e6, names, nbytes)
}
