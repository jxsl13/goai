//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// chainLen mimics the decode op count (~286 launches/token) with a fixed sequence
// of small in-place kernels on a resident buffer, to isolate host-launch overhead.
const chainLen = 200

// captureChain pins the OS thread, captures chainLen GELU ops on buf, and returns
// the instantiated graph. GELU is capture-safe (async, no host sync, no alloc).
func captureChain(t testing.TB, buf *cuda.DeviceF32) *cuda.CapturedGraph {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.CaptureBegin(); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < chainLen; i++ {
		if err := buf.GELU(); err != nil {
			t.Fatalf("capture op %d: %v", i, err)
		}
	}
	g, err := cuda.CaptureEnd()
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// A captured graph must replay the SAME computation as the direct op sequence:
// chainLen GELUs applied to identical inputs must land within f32 noise.
func TestCUDAGraphReplayMatchesDirect(t *testing.T) {
	skipNoGPU(t)
	x := bench.RandF32(tensor.Shape{1, 4096}, 3)

	// direct
	dd, _ := cuda.UploadF32(x)
	for i := 0; i < chainLen; i++ {
		dd.GELU()
	}
	cuda.GraphSync()
	direct, _ := dd.ToHost()
	dd.Free()

	// graph
	dg, _ := cuda.UploadF32(x)
	g := captureChain(t, dg)
	defer g.Free()
	if err := g.Launch(); err != nil {
		t.Fatal(err)
	}
	cuda.GraphSync()
	viaGraph, _ := dg.ToHost()
	dg.Free()

	var maxAbs float64
	for i := range direct.Numel() {
		idx := tensor.Unravel(i, direct.Shape())
		if d := abs(direct.AtF64(idx...) - viaGraph.AtF64(idx...)); d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > 1e-5 {
		t.Fatalf("graph replay diverges from direct: maxAbs %.3e", maxAbs)
	}
	t.Logf("graph replay == direct (%d-op chain), maxAbs %.2e", chainLen, maxAbs)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// The POC measurement: does replaying the captured graph cut host-launch overhead
// vs issuing the same chainLen ops directly (each op = cgo + mutex + cuCtxSetCurrent
// + driver launch)? Each iteration syncs once, mimicking a decode token's single
// end-of-step sync.
func BenchmarkGraphChainDirect(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	x := bench.RandF32(tensor.Shape{1, 4096}, 3)
	d, _ := cuda.UploadF32(x)
	defer d.Free()
	b.ResetTimer()
	for range b.N {
		for i := 0; i < chainLen; i++ {
			d.GELU()
		}
		cuda.GraphSync()
	}
}

func BenchmarkGraphChainReplay(b *testing.B) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	x := bench.RandF32(tensor.Shape{1, 4096}, 3)
	d, _ := cuda.UploadF32(x)
	defer d.Free()
	g := captureChain(b, d)
	defer g.Free()
	b.ResetTimer()
	for range b.N {
		g.Launch()
		cuda.GraphSync()
	}
}
