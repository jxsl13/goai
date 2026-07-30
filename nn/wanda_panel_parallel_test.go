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

// TestWandaPruneNMFanOutBitIdentical is the same gate for the N:M path, which had none.
//
// The goldens do not reach it. WandaPruneNM fans its output-column loop out only when
// cout*cin clears parallelRows' work gate, and the largest golden case is cout=8, cin=16 — 128
// against a bound of 16384 — so every golden exercises the serial branch only. That matters
// specifically here because the per-worker grp/gsc/drop scratch and the block comparator live
// INSIDE the fan-out closure, so a scratch-sharing or comparator mistake there is invisible to
// every existing test.
//
// As with the unstructured gate above, the two arms are the same source selected by GOMAXPROCS,
// so a difference is attributable to the partition rather than to a separately written reference.
//
// WHAT THIS DOES AND DOES NOT CATCH, measured rather than assumed. It catches
// partition-DEPENDENT errors: skipping one column only when olo > 0 reddens this test and
// nothing else. It does NOT reliably catch SCRATCH SHARING — hoisting grp/gsc/drop out of the
// closure so every worker shares them left this test and all eleven other Wanda tests GREEN,
// because a racing write can still land on a value that matches the serial result. The race
// detector reported it immediately. So the two guards are complementary and neither substitutes
// for the other: this test for partitioning, `go test -race` (make race, and the CI cgo+race
// lane) for sharing. Note that the unstructured panel gate above DID catch its shared-scratch
// mutation, which is exactly why this had to be checked here separately instead of assumed.
func TestWandaPruneNMFanOutBitIdentical(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("single-CPU host: the fan-out arm cannot differ from the serial arm")
	}
	// cout*cin = 65536, comfortably above parallelRows' 1<<14 gate. cin is a multiple of m=4
	// so the N:M blocking divides evenly, and the guard below fails loudly if the gate is
	// ever raised past this geometry rather than letting the test go quietly serial.
	const cin, cout, samples = 256, 256, 24
	if cout*cin < 1<<14 {
		t.Fatalf("geometry no longer reaches the fan-out: cout*cin = %d", cout*cin)
	}
	run := func(dt tensor.Dtype, procs int) []uint64 {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
		w := tensor.New(dt, tensor.Shape{cin, cout})
		x := tensor.New(dt, tensor.Shape{samples, cin})
		if dt == tensor.F64 {
			for i, s := 0, w.Storage().F64(); i < len(s); i++ {
				s[i] = math.Sin(float64(i) * 0.37)
			}
			for i, s := 0, x.Storage().F64(); i < len(s); i++ {
				s[i] = math.Cos(float64(i) * 0.013)
			}
		} else {
			for i, s := 0, w.Storage().F32(); i < len(s); i++ {
				s[i] = float32(math.Sin(float64(i) * 0.37))
			}
			for i, s := 0, x.Storage().F32(); i < len(s); i++ {
				s[i] = float32(math.Cos(float64(i) * 0.013))
			}
		}
		pruned, mask, err := nn.WandaPruneNM(w, x, 2, 4)
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
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		par := run(dt, runtime.NumCPU())
		ser := run(dt, 1)
		if len(par) != len(ser) {
			t.Fatalf("%v: %d values parallel, %d serial", dt, len(par), len(ser))
		}
		for i := range ser {
			if par[i] != ser[i] {
				t.Fatalf("%v: value %d: parallel %016x != serial %016x — the N:M column partition "+
					"changed a result", dt, i, par[i], ser[i])
			}
		}
	}
}
