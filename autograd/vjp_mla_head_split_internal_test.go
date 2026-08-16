package autograd

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

func mlaDigest(h uint64, xs []float64) uint64 {
	for _, x := range xs {
		b := math.Float64bits(x)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (b>>s)&0xff) * 1099511628211
		}
	}
	return h
}

// mlaSplitInputs builds an MLA backward with ENOUGH HEADS to split and enough work to clear the
// fan-out gate. Four heads is what the benchmark uses; six is used here so a band covers more
// than one head on this machine and the per-band loop is exercised rather than a one-head band.
func mlaSplitInputs(seq, heads, dh, dR int, causal bool) ([]*tensor.Tensor, *tensor.Tensor, backend.MLAAttrs) {
	hdh := heads * dh
	in := []*tensor.Tensor{
		moeF64(tensor.Shape{seq, hdh}, 11, false),
		moeF64(tensor.Shape{seq, hdh}, 22, false),
		moeF64(tensor.Shape{seq, hdh}, 33, false),
		moeF64(tensor.Shape{seq, heads * dR}, 44, false),
		moeF64(tensor.Shape{seq, dR}, 55, false),
	}
	g := moeF64(tensor.Shape{seq, hdh}, 66, false)
	return in, g, backend.MLAAttrs{Heads: heads, Causal: causal, RoPEBase: 10000}
}

func mlaGradsDigest(t *testing.T, outs []*tensor.Tensor) uint64 {
	t.Helper()
	h := uint64(14695981039346656037)
	for _, o := range outs {
		h = mlaDigest(h, o.Contiguous().Storage().F64())
	}
	return h
}

// TestMLAVJPHeadSplitIsBitExact is the golden for the head split.
//
// The split is safe for four of the five gradients because heads write disjoint windows. The
// fifth, the shared decoupled-key gradient, is accumulated by EVERY head at the same address, so
// its factors are recorded during the parallel pass and folded afterwards in the original
// (head, i, j) order. That fold is the whole reason this stays exact, and only a value produced
// by the previous implementation can show it does.
//
// Both causal settings are covered: the causal arm walks a triangular j range, so a fold that
// used the wrong bound would agree with the serial form on the rectangular arm and not on this one.
func TestMLAVJPHeadSplitIsBitExact(t *testing.T) {
	if raceBuild {
		// Not a tolerance concession: the digests below differ under -race on the PRE-change
		// implementation too, identically, so the race build is changing the arithmetic and not
		// this change. Pinning both values would pin a compiler mode, not a contract.
		t.Skip("the race build changes floating-point results; see racebuild_off.go")
	}
	for _, tc := range []struct {
		name   string
		causal bool
		want   uint64
	}{
		{"causal", true, archgold.Pick(3570864407073999628, 1489154577517272182)},
		{"full", false, archgold.Pick(11976000054950329798, 3782504584191206100)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			vjp := vjps[backend.OpMLA]
			in, g, attrs := mlaSplitInputs(48, 6, 16, 8, tc.causal)
			outs, err := vjp(nil, in, nil, attrs, g)
			if err != nil {
				t.Fatal(err)
			}
			if got := mlaGradsDigest(t, outs); got != tc.want {
				t.Fatalf("gradient digest %d, want %d", got, tc.want)
			}
		})
	}
}

// TestMLAVJPHeadSplitMatchesSerial gates the SPLIT itself. The band count is read from GOMAXPROCS,
// so the two arms genuinely take the one-band and the many-band path, and a band that overlapped
// or skipped a head would separate them. It says nothing about the FOLD — both arms record and
// fold — which is what the golden above is for.
func TestMLAVJPHeadSplitMatchesSerial(t *testing.T) {
	vjp := vjps[backend.OpMLA]
	in, g, attrs := mlaSplitInputs(48, 6, 16, 8, true)
	run := func() []float64 {
		outs, err := vjp(nil, in, nil, attrs, g)
		if err != nil {
			t.Fatal(err)
		}
		var flat []float64
		for _, o := range outs {
			flat = append(flat, o.Contiguous().Storage().F64()...)
		}
		return flat
	}
	prev := runtime.GOMAXPROCS(1)
	serial := run()
	runtime.GOMAXPROCS(prev)
	parallel := run()
	for i := range serial {
		if math.Float64bits(serial[i]) != math.Float64bits(parallel[i]) {
			t.Fatalf("gradient %d: serial %v, %d workers %v", i, serial[i], prev, parallel[i])
		}
	}
}

// TestMLAVJPHeadSplitF32IsBitExact is the same golden for the F32 arm, which is a separate copy of
// the loop and got the same record-and-fold.
//
// TestMLAF32Parity does not cover this. It compares the F32 arm against float32 of the F64 one,
// which is the right oracle for the arithmetic, but its shape is seq 5 with two heads — far below
// the fan-out gate — so both arms run unsplit there and the split is invisible to it. This shape
// splits.
//
// WHAT THIS GOLDEN CANNOT SEE, measured rather than assumed: reversing the head order in the fold
// leaves it green. The shared gradient accumulates in float64 and is stored back as float32, and
// that rounding swallows the last-bit difference a reordered sum produces. The F64 golden above,
// on identical code, DOES redden under the same mutation — so the order is pinned there and the
// values here, and neither test is asked to prove what it cannot.
func TestMLAVJPHeadSplitF32IsBitExact(t *testing.T) {
	if raceBuild {
		t.Skip("the race build changes floating-point results; see racebuild_off.go")
	}
	for _, tc := range []struct {
		name   string
		causal bool
		want   uint64
	}{
		{"causal", true, archgold.Pick(6899433073126379515, 6899433073126379515)},
		{"full", false, archgold.Pick(6359307992473138402, 6359307992473138402)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			vjp := vjps[backend.OpMLA]
			in64, g64, attrs := mlaSplitInputs(48, 6, 16, 8, tc.causal)
			in := make([]*tensor.Tensor, len(in64))
			for i := range in64 {
				in[i] = moeCastF32(in64[i])
			}
			outs, err := vjp(nil, in, nil, attrs, moeCastF32(g64))
			if err != nil {
				t.Fatal(err)
			}
			h := uint64(14695981039346656037)
			for _, o := range outs {
				if o.Dtype() == tensor.F32 {
					for _, v := range o.Contiguous().Storage().F32() {
						h = mlaDigest(h, []float64{float64(v)})
					}
					continue
				}
				h = mlaDigest(h, o.Contiguous().Storage().F64())
			}
			if h != tc.want {
				t.Fatalf("gradient digest %d, want %d", h, tc.want)
			}
		})
	}
}
