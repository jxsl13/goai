package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
)

// aqlmGoldenCodeHash is a rolling hash over EVERY assigned code, and aqlmGoldenCB0 the
// first codebook row, both captured before the ICM refinement was parallelized.
//
// TestAQLMDeterministic cannot serve this purpose: it encodes twice with the SAME code
// and compares, which proves the encode is reproducible, not that it still produces what
// it used to. ICM is a coordinate-descent search whose every step is decided by an
// argmin, so a change that reorders the search silently lands on different codes and
// passes a reproducibility test unchanged.
//
// The hash covers all 768 codes rather than a prefix: ICM refines groups independently,
// so a partitioning bug would corrupt a middle chunk that a prefix check never reads.
const aqlmGoldenCodeHash = 0x26377e33ca16f3ae

var aqlmGoldenCB0 = [4]uint64{
	0xbfb1691e3419a8ef,
	0x3fbd693351551919,
	0x3fd377d11b9a1ebb,
	0x3fdbe8c37a0f0f90,
}

// TestAQLMEncodeBitIdenticalToGolden holds AQLM encoding bit-for-bit against a frozen
// reference at tolerance 0 — every code and the codebook it refits to.
func TestAQLMEncodeBitIdenticalToGolden(t *testing.T) {
	w := randWeight(32, 48, 7)
	q, err := nn.EncodeAQLM(w, nn.WithAQLMCodebooks(2), nn.WithAQLMBits(4),
		nn.WithAQLMGroupSize(4), nn.WithAQLMIters(8), nn.WithAQLMSeed(11))
	if err != nil {
		t.Fatal(err)
	}
	var sum uint64
	for _, c := range q.Codes {
		sum = sum*1315423911 + uint64(c)
	}
	if sum != aqlmGoldenCodeHash {
		t.Fatalf("code hash %#x, want %#x — the ICM search no longer assigns the same codes",
			sum, aqlmGoldenCodeHash)
	}
	for i, want := range aqlmGoldenCB0 {
		if got := math.Float64bits(q.Codebooks[0][i]); got != want {
			t.Fatalf("codebook[0][%d] = %v (%#x), want %v (%#x)",
				i, q.Codebooks[0][i], got, math.Float64frombits(want), want)
		}
	}
}
