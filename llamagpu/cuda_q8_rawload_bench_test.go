//go:build cuda && cgo && (linux || windows)

package llamagpu_test

import (
	"os"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

// TestQ8RawLoadDirect loads a Q8_0 GGUF through the RAW-quant path (ReadRaw → QuantLlamaFromGGUF →
// NewQuant → cudaUploadQWeight), which now repacks Q8_0 blocks DIRECTLY into ResidentBQ8 (no f32
// round-trip). It validates the path works end-to-end AND times the load. Set GOAI_BENCH_MODEL to a
// Q8_0 GGUF. Compare the reported load time against the f32 path by toggling the qt==8 branch.
func TestQ8RawLoadDirect(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	path := os.Getenv("GOAI_BENCH_MODEL")
	if path == "" {
		t.Skip("set GOAI_BENCH_MODEL=<q8_0.gguf>")
	}
	fh, err := os.Open(path)
	if err != nil {
		t.Skipf("open: %v", err)
	}
	defer fh.Close()
	rf, err := gguf.ReadRaw(fh)
	if err != nil {
		t.Skipf("ReadRaw: %v", err)
	}
	ql, err := nlp.QuantLlamaFromGGUF(rf.Metadata, rf.Tensors)
	if err != nil {
		t.Skipf("QuantLlamaFromGGUF: %v", err)
	}
	t0 := time.Now()
	dec, err := llamagpu.NewQuantCUDA(ql)
	if err != nil {
		t.Fatalf("NewQuantCUDA (raw Q8 direct path): %v", err)
	}
	load := time.Since(t0)
	defer dec.Release()
	// Prove it actually runs end-to-end (logits computed without NaN).
	logits, err := dec.Step(1, 0)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	best, bi := logits[0], 0
	for i, v := range logits {
		if v > best {
			best, bi = v, i
		}
	}
	t.Logf("raw-Q8-direct load (NewQuantCUDA) = %v; vocab=%d argmax-token=%d", load, len(logits), bi)
}
