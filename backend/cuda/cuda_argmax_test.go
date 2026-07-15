//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// On-device argmax must equal the host argmax over the same logits.
func TestCUDAArgmaxMatchesHost(t *testing.T) {
	skipNoGPU(t)
	x := bench.RandF32(tensor.Shape{1, 32000}, 7)
	d, _ := cuda.UploadF32(x)
	defer d.Free()
	got := d.Argmax()
	want := argmaxRow(x, 0)
	if got != want {
		t.Fatalf("device argmax %d != host %d", got, want)
	}
	t.Logf("device argmax == host (%d)", got)
}

// Q8 graph decode using ON-DEVICE argmax (4-byte token download) vs the 128 KB
// full-logit download + host argmax (the current path, ~161.6 tok/s).
func BenchmarkTinyLlamaDecodeQ8GraphArgmax_p32(b *testing.B) {
	m := loadTinyLlama(b)
	gd := buildQ8GraphDecoder(b, m, 32+64+2)
	defer gd.free()
	for p := 0; p < 32; p++ {
		gd.step(b, int32((p*131+1)%m.Config.Vocab), p)
	}
	runtime.LockOSThread()
	mustTB(b, cuda.CaptureBegin())
	gd.forwardBody(b)
	g, err := cuda.CaptureEnd()
	runtime.UnlockOSThread()
	mustTB(b, err)
	gd.graph = g
	gd.pos.Set(32)
	gd.emb.EmbedInto([]int32{0}, gd.dx)
	gd.graph.Launch()
	cuda.GraphSync()
	tok := int32(gd.logits.Argmax())
	b.ResetTimer()
	for range b.N {
		for d := 0; d < 64; d++ {
			gd.pos.Set(32 + 1 + d)
			gd.emb.EmbedInto([]int32{tok}, gd.dx)
			gd.graph.Launch()
			// no explicit GraphSync: Argmax enqueues after the graph on the same
			// stream and its own sync waits for both — one sync/token, not two.
			tok = int32(gd.logits.Argmax())
		}
	}
	b.ReportMetric(float64(64)*float64(b.N)/b.Elapsed().Seconds(), "tok/s")
}
