//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Isolate the graph-capture divergence: does a SINGLE cuBLAS matmul captured into
// a graph replay bit-identically to a direct call? If not, cuBLAS is the culprit.
func TestCUDAGraphCublasMatmulIsolate(t *testing.T) {
	skipNoGPU(t)
	a := bench.RandF32(tensor.Shape{1, 2048}, 1)
	w := bench.RandF32(tensor.Shape{2048, 2048}, 2)
	rw, _ := cuda.NewResidentB(w)
	defer rw.Free()

	da, _ := cuda.UploadF32(a)
	defer da.Free()
	outDirect, _ := cuda.NewDeviceF32(1, 2048)
	defer outDirect.Free()
	must(t, rw.MatMulInto(da, outDirect))
	cuda.GraphSync()
	hd, _ := outDirect.ToHost()

	// capture the same matmul, replay
	outGraph, _ := cuda.NewDeviceF32(1, 2048)
	defer outGraph.Free()
	runtime.LockOSThread()
	must(t, cuda.CaptureBegin())
	must(t, rw.MatMulInto(da, outGraph))
	g, err := cuda.CaptureEnd()
	runtime.UnlockOSThread()
	must(t, err)
	defer g.Free()
	must(t, g.Launch())
	must(t, cuda.GraphSync())
	hg, _ := outGraph.ToHost()

	var maxAbs float64
	for i := range hd.Numel() {
		idx := tensor.Unravel(i, hd.Shape())
		if d := abs(hd.AtF64(idx...) - hg.AtF64(idx...)); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("single cuBLAS matmul: graph vs direct maxAbs %.3e", maxAbs)
	if maxAbs > 1e-3 {
		t.Fatalf("cuBLAS matmul does NOT capture correctly (maxAbs %.3e) — cuBLAS is the culprit", maxAbs)
	}
	t.Log("cuBLAS matmul DOES capture correctly → the decode-graph bug is elsewhere (device-pos ops?)")
}

// The crux of the graph decode: a captured kernel that reads position from a
// DevicePos must use the value set BEFORE each replay (not the capture-time
// value). If it uses the capture-time pos, the device-pos-in-graph design is
// broken — which would exactly explain the decode divergence.
func TestCUDAGraphDevicePosReplay(t *testing.T) {
	skipNoGPU(t)
	const heads, hd = 4, 8
	pos, _ := cuda.NewDevicePos()
	defer pos.Free()
	inv, _ := cuda.BuildRoPEInv(hd, 10000)
	defer inv.Free()
	x := bench.RandF32(tensor.Shape{1, heads * hd}, 5)

	// capture RoPEDposInv reading *pos (pos happens to be 0 at capture)
	dg, _ := cuda.UploadF32(x)
	defer dg.Free()
	runtime.LockOSThread()
	must(t, cuda.CaptureBegin())
	must(t, dg.RoPEDposInv(heads, inv, pos, 0))
	g, err := cuda.CaptureEnd()
	runtime.UnlockOSThread()
	must(t, err)
	defer g.Free()

	for _, P := range []int{3, 9} {
		// direct reference at position P
		dref, _ := cuda.UploadF32(x)
		must(t, dref.RoPEDposInv(heads, inv, mustPos(t, P), 0))
		cuda.GraphSync()
		href, _ := dref.ToHost()
		dref.Free()

		// graph: reset dg to x, set pos=P, replay
		src, _ := cuda.UploadF32(x)
		must(t, dg.CopyFrom(src))
		src.Free()
		must(t, pos.Set(P))
		must(t, g.Launch())
		must(t, cuda.GraphSync())
		hg, _ := dg.ToHost()

		var maxAbs float64
		for i := range href.Numel() {
			idx := tensor.Unravel(i, href.Shape())
			if d := abs(href.AtF64(idx...) - hg.AtF64(idx...)); d > maxAbs {
				maxAbs = d
			}
		}
		t.Logf("pos=%d: graph-replay vs direct maxAbs %.3e", P, maxAbs)
		if maxAbs > 1e-5 {
			t.Fatalf("graph replay at pos=%d used the WRONG position (maxAbs %.3e) — device-pos-in-graph broken", P, maxAbs)
		}
	}
	t.Log("device-pos-in-graph works: replay uses the between-replay pos")
}

func mustPos(t *testing.T, p int) *cuda.DevicePos {
	dp, err := cuda.NewDevicePos()
	must(t, err)
	must(t, dp.Set(p))
	return dp
}
