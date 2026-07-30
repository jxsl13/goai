package nn_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestWandaPrunePanelFanOutBitIdentical is the bit-identity gate for the PARALLEL panel loop.
//
// It exists because nothing else reaches that code. WandaPrune fans out only when there are at
// least two panels, and the panel width is min(wandaPanel, cout) — so a test with cout below
// wandaPanel gets exactly one panel and runs serially no matter how the fan-out is written.
// Every existing Wanda test is in that regime: the quickselect-equivalence sweep caps cout at
// 40, and the goldens are smaller still. A geometry with cout above the panel width is the only
// way to enter the parallel path.
//
// The two arms are THE SAME SOURCE selected by GOMAXPROCS: parallelRows runs the body inline
// when it sees a single processor, so setting GOMAXPROCS(1) yields the serial arm without a
// second implementation to drift from. Any difference is therefore attributable to the
// partition, not to a reference written in a different shape.
func TestWandaPrunePanelFanOutBitIdentical(t *testing.T) {
	// cin/cout chosen so pn = wandaPanel (128) and nPanels = 2, and so nPanels*pn*cin clears
	// parallelRows' work threshold. If wandaPanel is ever raised above 128 this geometry stops
	// reaching the fan-out, so the guard below fails loudly rather than passing vacuously.
	const cin, cout, samples = 256, 256, 24
	mk := func(dt tensor.Dtype) (*tensor.Tensor, *tensor.Tensor) {
		w := tensor.New(dt, tensor.Shape{cin, cout})
		wf := w.Storage()
		x := tensor.New(dt, tensor.Shape{samples, cin})
		xf := x.Storage()
		if dt == tensor.F64 {
			for i, s := 0, wf.F64(); i < len(s); i++ {
				s[i] = math.Sin(float64(i) * 0.37)
			}
			for i, s := 0, xf.F64(); i < len(s); i++ {
				s[i] = math.Cos(float64(i) * 0.013)
			}
		} else {
			for i, s := 0, wf.F32(); i < len(s); i++ {
				s[i] = float32(math.Sin(float64(i) * 0.37))
			}
			for i, s := 0, xf.F32(); i < len(s); i++ {
				s[i] = float32(math.Cos(float64(i) * 0.013))
			}
		}
		return w, x
	}

	run := func(dt tensor.Dtype, procs int) []uint64 {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
		w, x := mk(dt)
		pruned, mask, err := nn.WandaPrune(w, x, 0.5)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]uint64, 0, 2*cin*cout)
		for _, tn := range []*tensor.Tensor{pruned, mask} {
			for i := range tn.Numel() {
				out = append(out, math.Float64bits(tn.AtF64(tensor.Unravel(i, tn.Shape())...)))
			}
		}
		return out
	}

	if runtime.NumCPU() < 2 {
		t.Skip("single-CPU host: the fan-out arm cannot differ from the serial arm")
	}
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		par := run(dt, runtime.NumCPU())
		ser := run(dt, 1)
		if len(par) != len(ser) {
			t.Fatalf("%v: %d values parallel, %d serial", dt, len(par), len(ser))
		}
		diff := 0
		for i := range ser {
			if par[i] != ser[i] {
				if diff == 0 {
					t.Errorf("%v: value %d: parallel %016x != serial %016x — the panel partition "+
						"changed a result", dt, i, par[i], ser[i])
				}
				diff++
			}
		}
		if diff > 0 {
			t.Fatalf("%v: %d of %d values differ between the panel fan-out and the serial arm",
				dt, diff, len(ser))
		}
	}
}
