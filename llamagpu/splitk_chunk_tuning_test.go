//go:build darwin && cgo

package llamagpu_test

// TestSplitKChunkSizeIsTuned records that split-K's chunk size is already at its optimum, so the
// occupancy lever behind it is exhausted.
//
// Split-K wins by turning `heads` threadgroups into `heads x nchunk`, so more chunks should mean more
// parallelism and more latency hiding. It does not, past the shipped default. Decode-only GPU-busy
// per step (M2 Pro, TinyLlama-1.1B Q4_K_M), sweeping keys-per-chunk:
//
//	ctx= 512  perChunk 256 -> 157.2 tok/s | 128 -> 156.0 | 64 -> 153.8 | 32 -> 148.6
//	ctx=1536  perChunk 256 -> 118.3 tok/s | 128 -> 121.9 | 64 -> 118.2 | 32 -> 110.8
//
// 128 keys per chunk — what mtl_recorder_mha already uses — is the best or within noise of the best
// at both lengths, and halving it twice costs 5-9%. The second pass has to merge one (m,l,acc[64])
// partial per chunk, so merge work grows linearly in nchunk while the parallelism gain saturates
// once the GPU is filled.
//
// A first attempt at this sweep varied the chunk CAP (16/32/64/128) and measured no change at all,
// because nchunk = ceil(sk/128) is 4 at ctx=512 and 12 at ctx=1536 — both already under the cap, so
// the knob was never binding. The parameter that matters is the divisor, not the ceiling.
//
// Consequence: attention still sits ~7x off its bandwidth floor at ctx=1536 (2.8ms/token for ~69 MB
// of KV traffic, ~0.38ms at ~180 GB/s), and that residue is NOT recoverable by launching more
// threadgroups. Whatever is left is inside the per-chunk work.
//
// metal.SetSplitKPerChunk is exposed for geometries unlike this one; the default is what this sweep
// chose. Reported, not asserted on absolute timings.
import (
	"fmt"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/llamagpu"
	"github.com/jxsl13/goai/nlp"
)

func TestSplitKChunkSizeIsTuned(t *testing.T) {
	if !metal.Available() {
		t.Skip("no metal")
	}
	f, _ := os.Open(os.Getenv("GOAI_TINYLLAMA_GGUF"))
	if f == nil {
		t.Skip("no model")
	}
	defer f.Close()
	raw, _ := gguf.ReadRaw(f)
	qm, _ := nlp.QuantLlamaFromGGUF(raw.Metadata, raw.Tensors)
	defer qm.Close()
	dec, _ := llamagpu.NewQuant(qm)
	defer dec.Release()
	for _, plen := range []int{512, 1536} {
		p := make([]int, plen)
		for i := range p {
			p[i] = 1 + i%2000
		}
		dec.StepNLast(p, 0)
		for _, ch := range []int{256, 128, 64, 32} {
			metal.SetSplitKChunks(512)
			metal.SetSplitKPerChunk(ch)
			for i := 0; i < 3; i++ {
				dec.Step(5, plen+i)
			}
			var gpu float64
			const n = 12
			for i := 0; i < n; i++ {
				dec.Step(5, plen+3+i)
				gpu += metal.LastGPUSeconds() * 1e3
			}
			gpu /= n
			fmt.Printf("CHUNK ctx=%5d perChunk=%3d (nchunk=%3d)  gpuBusy=%6.2fms  %6.1f tok/s\n", plen, ch, (plen+ch)/ch, gpu, 1000/gpu)
		}
	}
	metal.SetSplitKChunks(16)
	metal.SetSplitKPerChunk(128)
}
