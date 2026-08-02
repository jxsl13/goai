package nn

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// memOracleBank builds a memory bank whose keys are all distinct in the head slice the
// test reads, so the top-M set is uniquely determined and a mis-assigned query row cannot
// coincidentally produce the right neighbours.
func memOracleBank(n, dim int) *MemMemory {
	m := &MemMemory{dim: dim, cap: n}
	m.keys = make([][]float64, n)
	m.vals = make([][]float64, n)
	for i := range m.keys {
		kr := make([]float64, dim)
		vr := make([]float64, dim)
		for d := range kr {
			kr[d] = math.Sin(float64(i*7+d)*0.31) + 0.001*float64(i)
			vr[d] = math.Cos(float64(i*13+d) * 0.17)
		}
		m.keys[i], m.vals[i] = kr, vr
	}
	return m
}

// TestMemGatherMatchesPerRowRetrieve is the oracle test for the TILED neighbour scan.
//
// gather no longer searches one query row at a time: it walks the key store once per tile
// of memGatherTile rows, keeping a heap per row. retrieveHead is deliberately left
// untouched by that change and is a second, independent implementation of the same
// search, so comparing gather's output against a per-row retrieveHead loop tests the
// tiling itself — the query-to-output assignment, the per-row heap isolation, and the
// scratch reuse between tiles — and not merely that the code agrees with itself.
//
// TestMemGatherParallelBitExact cannot do this job. It compares GOMAXPROCS=1 against N,
// and BOTH sides run the tiled scan, so every tiling bug that is not a data race is
// invisible to it.
//
// The row counts are chosen so partial tiles and reuse are both exercised: 37 is not a
// multiple of the tile width, and at GOMAXPROCS=1 one worker walks all of it, so the
// scratch heaps are reused across three tiles of which the last is short.
func TestMemGatherMatchesPerRowRetrieve(t *testing.T) {
	const n, dim, headDim, headOff, topM = 100, 96, 32, 32, 7
	m := memOracleBank(n, dim)

	for _, tc := range []struct {
		name string
		t    int
		proc int
	}{
		{"serial-partial-tile", 37, 1},   // 16+16+5 on one worker: reuse plus a short tail
		{"serial-under-a-tile", 9, 1},    // fewer rows than one tile
		{"parallel-many-tiles", 400, 12}, // every worker gets more than one full tile
	} {
		t.Run(tc.name, func(t *testing.T) {
			prev := runtime.GOMAXPROCS(tc.proc)
			defer runtime.GOMAXPROCS(prev)

			qh := tensor.New(tensor.F64, tensor.Shape{tc.t, headDim})
			qs := qh.Storage().F64()
			for i := range qs {
				qs[i] = math.Sin(float64(i)*0.37) * 1.5
			}
			kg, vg := m.gather(tensor.F64, qh, headOff, headDim, topM)
			kgs, vgs := kg.Storage().F64(), vg.Storage().F64()

			qrow := make([]float64, headDim)
			for ti := range tc.t {
				copy(qrow, qs[ti*headDim:ti*headDim+headDim])
				idx, _ := m.retrieveHead(qrow, headOff, headDim, topM)
				if len(idx) != topM {
					t.Fatalf("row %d: oracle returned %d neighbours, want %d", ti, len(idx), topM)
				}
				for r, id := range idx {
					base := ti*topM*headDim + r*headDim
					for d := range headDim {
						if got, want := kgs[base+d], m.keys[id][headOff+d]; got != want {
							t.Fatalf("row %d neighbour %d (key %d) dim %d: kg %v, oracle %v",
								ti, r, id, d, got, want)
						}
						if got, want := vgs[base+d], m.vals[id][headOff+d]; got != want {
							t.Fatalf("row %d neighbour %d (key %d) dim %d: vg %v, oracle %v",
								ti, r, id, d, got, want)
						}
					}
				}
			}
		})
	}
}

// TestMemGatherF32MatchesPerRowRetrieve holds the same oracle over the F32 output arm,
// which converts on the way out and is a separate code path through gatherRows.
func TestMemGatherF32MatchesPerRowRetrieve(t *testing.T) {
	const n, dim, headDim, headOff, topM, tSeg = 64, 64, 32, 0, 5, 37
	m := memOracleBank(n, dim)
	qh := tensor.New(tensor.F64, tensor.Shape{tSeg, headDim})
	qs := qh.Storage().F64()
	for i := range qs {
		qs[i] = math.Cos(float64(i) * 0.23)
	}
	kg, vg := m.gather(tensor.F32, qh, headOff, headDim, topM)
	kgs, vgs := kg.Storage().F32(), vg.Storage().F32()

	qrow := make([]float64, headDim)
	for ti := range tSeg {
		copy(qrow, qs[ti*headDim:ti*headDim+headDim])
		idx, _ := m.retrieveHead(qrow, headOff, headDim, topM)
		for r, id := range idx {
			base := ti*topM*headDim + r*headDim
			for d := range headDim {
				if got, want := kgs[base+d], float32(m.keys[id][headOff+d]); got != want {
					t.Fatalf("row %d neighbour %d dim %d: kg %v, oracle %v", ti, r, d, got, want)
				}
				if got, want := vgs[base+d], float32(m.vals[id][headOff+d]); got != want {
					t.Fatalf("row %d neighbour %d dim %d: vg %v, oracle %v", ti, r, d, got, want)
				}
			}
		}
	}
}

// TestMemGatherFewerKeysThanTopM pins the clamp the tiled scan has to reproduce: when the
// bank holds fewer pairs than topM, only that many neighbours are written and the rest of
// each output block stays zero. The block STRIDE is still the declared topM, so getting
// the clamp wrong would either write past a row or shift every later row.
func TestMemGatherFewerKeysThanTopM(t *testing.T) {
	const n, dim, headDim, topM, tSeg = 3, 32, 32, 8, 20
	m := memOracleBank(n, dim)
	qh := tensor.New(tensor.F64, tensor.Shape{tSeg, headDim})
	qs := qh.Storage().F64()
	for i := range qs {
		qs[i] = math.Sin(float64(i) * 0.11)
	}
	kg, _ := m.gather(tensor.F64, qh, 0, headDim, topM)
	kgs := kg.Storage().F64()
	if len(kgs) != tSeg*topM*headDim {
		t.Fatalf("output has %d cells, want %d", len(kgs), tSeg*topM*headDim)
	}
	qrow := make([]float64, headDim)
	for ti := range tSeg {
		copy(qrow, qs[ti*headDim:ti*headDim+headDim])
		idx, _ := m.retrieveHead(qrow, 0, headDim, topM)
		if len(idx) != n {
			t.Fatalf("row %d: oracle returned %d, want the whole bank %d", ti, len(idx), n)
		}
		for r, id := range idx {
			base := ti*topM*headDim + r*headDim
			for d := range headDim {
				if got, want := kgs[base+d], m.keys[id][d]; got != want {
					t.Fatalf("row %d neighbour %d dim %d: %v, oracle %v", ti, r, d, got, want)
				}
			}
		}
		for r := n; r < topM; r++ { // the unfilled tail of the block
			base := ti*topM*headDim + r*headDim
			for d := range headDim {
				if kgs[base+d] != 0 {
					t.Fatalf("row %d slot %d dim %d: %v, want an untouched zero", ti, r, d, kgs[base+d])
				}
			}
		}
	}
}
