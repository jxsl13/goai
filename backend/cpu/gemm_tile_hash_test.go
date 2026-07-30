//go:build !(amd64 && goexperiment.simd) && !(arm64 && goexperiment.simd)

package cpu

import (
	"math"
	"math/rand"
	"testing"
)

// TestGemmF32PortableHash freezes the exact bits of the portable f32 GEMM across geometries that
// exercise BOTH the 4x4 register tile and every remainder it can leave: n not divisible by 4 takes
// the column tail, and m not divisible by 4 takes the single-row tail.
func TestGemmF32PortableHash(t *testing.T) {
	const wantHash uint64 = 0x9a9753f0e16903dc
	geoms := []struct{ m, k, n int }{
		{8, 8, 8}, {7, 5, 9}, {4, 3, 6}, {13, 11, 15}, {1, 1, 1}, {5, 4, 3},
		// Geometries that clear BOTH pack gates — the work gate needs k*n at or above
		// gemmPackMinWorkF32, and the band gate needs enough rows that a band gets two 4-row tile
		// blocks. One has n a multiple of 4 and one does not, so the packed path's own column
		// remainder runs too.
		//
		// These had to grow twice, and the second time exposed a real hole: while the guard below
		// checked only the ROW condition, this set satisfied it with k*n=72 and so never reached
		// the packed band at all once the work gate existed. The guard now asks the same question
		// the kernel asks, and immediately rejected the old set.
		{96, 64, 64}, {96, 64, 67}, {128, 40, 20},
	}
	var packed int
	for _, g := range geoms {
		if g.n >= 4 && g.k*g.n >= gemmPackMinWorkF32 && gemmPackBands(g.m, g.k, g.n, gemmPackTileBlocksF32) {
			packed++
		}
	}
	if packed == 0 {
		t.Fatalf("no geometry reaches the packed band (needs %d tile blocks per band); this golden "+
			"would cover only the unpacked kernel", gemmPackTileBlocksF32)
	}
	var h uint64 = 14695981039346656037
	for _, g := range geoms {
		rng := rand.New(rand.NewSource(int64(g.m*100 + g.n)))
		A := make([]float32, g.m*g.k)
		B := make([]float32, g.k*g.n)
		C := make([]float32, g.m*g.n)
		for i := range A {
			A[i] = rng.Float32()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float32()*2 - 1
		}
		gemmF32(A, B, C, g.m, g.k, g.n)
		for _, v := range C {
			h = (h ^ uint64(math.Float32bits(v))) * 1099511628211
		}
	}
	if wantHash == 0 {
		t.Fatalf("CAPTURE: %#x", h)
	}
	if h != wantHash {
		t.Fatalf("portable gemmF32 hash %#x, want %#x", h, wantHash)
	}
}

// TestGemmF64BandHash is the f64 twin, over the same remainder-covering geometries. The band is
// called directly with the full row range, which is how gemm.go drives a small matmul.
func TestGemmF64BandHash(t *testing.T) {
	const wantHash uint64 = 0x9a25beaffeacc1d5
	var h uint64 = 14695981039346656037
	for _, g := range []struct{ m, k, n int }{
		{8, 8, 8}, {7, 5, 9}, {4, 3, 6}, {13, 11, 15}, {1, 1, 1}, {5, 4, 3},
	} {
		rng := rand.New(rand.NewSource(int64(g.m*100 + g.n)))
		A := make([]float64, g.m*g.k)
		B := make([]float64, g.k*g.n)
		C := make([]float64, g.m*g.n)
		for i := range A {
			A[i] = rng.Float64()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float64()*2 - 1
		}
		// Seed C non-zero: this kernel ACCUMULATES, and a zeroed C would not catch a tile that
		// dropped the incoming value.
		for i := range C {
			C[i] = float64(i%7) * 0.25
		}
		gemmF64Band(A, B, C, 0, g.m, g.k, g.n)
		for _, v := range C {
			h = (h ^ math.Float64bits(v)) * 1099511628211
		}
	}
	if h != wantHash {
		t.Fatalf("gemmF64Band hash %#x, want %#x", h, wantHash)
	}
}

// TestGemmF32PackedMatchesUnpacked pins the packed band against the unpacked one directly, which a
// static golden alone cannot do: the pack is gated on B's size, so every geometry small enough to
// keep a golden fast is also small enough to skip packing entirely. Forcing the gate both ways and
// diffing is what actually covers the packed kernel.
func TestGemmF32PackedMatchesUnpacked(t *testing.T) {
	saved, savedB := gemmPackMinWorkF32, gemmPackTileBlocksF32
	defer func() { gemmPackMinWorkF32, gemmPackTileBlocksF32 = saved, savedB }()
	for _, g := range []struct{ m, k, n int }{
		{100, 9, 8}, {97, 7, 11}, {104, 5, 4}, {128, 16, 16}, {99, 3, 7},
	} {
		if g.n < 4 || !gemmPackBands(g.m, g.k, g.n, 1) {
			t.Fatalf("%+v never reaches the packed band whatever the work gate says", g)
		}
		rng := rand.New(rand.NewSource(int64(g.m*31 + g.n)))
		A := make([]float32, g.m*g.k)
		B := make([]float32, g.k*g.n)
		for i := range A {
			A[i] = rng.Float32()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float32()*2 - 1
		}
		run := func(gate int) []float32 {
			gemmPackMinWorkF32 = gate
			gemmPackTileBlocksF32 = 0
			C := make([]float32, g.m*g.n)
			gemmF32(A, B, C, g.m, g.k, g.n)
			return C
		}
		unpacked := run(1 << 30) // never packs
		packed := run(0)         // always packs
		for i := range unpacked {
			if math.Float32bits(unpacked[i]) != math.Float32bits(packed[i]) {
				t.Fatalf("%+v: C[%d] unpacked %v (%08x) != packed %v (%08x)", g, i,
					unpacked[i], math.Float32bits(unpacked[i]), packed[i], math.Float32bits(packed[i]))
			}
		}
	}
}

// TestGemmF64PackedMatchesUnpacked is the f64 twin of the packed/unpacked diff, and it is the only
// thing that covers gemmF64BandPacked: TestGemmF64BandHash calls the band directly, so it never
// routes through gemmF64Rows and never sees the pack at all.
func TestGemmF64PackedMatchesUnpacked(t *testing.T) {
	saved, savedB := gemmPackMinWorkF64, gemmPackTileBlocksF64
	defer func() { gemmPackMinWorkF64, gemmPackTileBlocksF64 = saved, savedB }()
	for _, g := range []struct{ m, k, n int }{
		{260, 9, 8}, {257, 7, 11}, {264, 5, 4}, {300, 16, 16}, {259, 3, 7},
	} {
		if g.n < 4 || !gemmPackBands(g.m, g.k, g.n, 1) {
			t.Fatalf("%+v never reaches the packed band whatever the work gate says", g)
		}
		rng := rand.New(rand.NewSource(int64(g.m*17 + g.k)))
		A := make([]float64, g.m*g.k)
		B := make([]float64, g.k*g.n)
		for i := range A {
			A[i] = rng.Float64()*2 - 1
		}
		for i := range B {
			B[i] = rng.Float64()*2 - 1
		}
		run := func(gate int) []float64 {
			gemmPackMinWorkF64 = gate
			gemmPackTileBlocksF64 = 0
			// Seed C non-zero: this path accumulates, and a zeroed C would not catch a packed
			// tile that dropped the incoming value.
			C := make([]float64, g.m*g.n)
			for i := range C {
				C[i] = float64(i%5) * 0.125
			}
			gemmF64Rows(A, B, C, g.m, g.k, g.n)
			return C
		}
		unpacked := run(1 << 30)
		packed := run(0)
		for i := range unpacked {
			if math.Float64bits(unpacked[i]) != math.Float64bits(packed[i]) {
				t.Fatalf("%+v: C[%d] unpacked %v != packed %v", g, i, unpacked[i], packed[i])
			}
		}
	}
}
