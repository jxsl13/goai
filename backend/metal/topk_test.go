//go:build darwin && cgo

package metal_test

import (
	"math/rand/v2"
	"sort"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
)

func topKReference(x []float32, n, k int) []int {
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(a, b int) bool {
		ai, bi := indices[a], indices[b]
		if x[ai] == x[bi] {
			return ai < bi
		}
		return x[ai] > x[bi]
	})
	return indices[:k]
}

func TestMetalResidentTopKN(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	for _, n := range []int{1, 31, 32, 33, 512, 32000, 128000} {
		rng := rand.New(rand.NewPCG(uint64(n)*131, 0x5eed))
		x := make([]float32, n+17) // exercise the first-n contract on an over-allocated buffer
		for i := range x {
			x[i] = float32(rng.NormFloat64())*7 + float32(i)*1e-6
		}
		buf, err := metal.NewDeviceBufferF32(x)
		if err != nil {
			t.Fatal(err)
		}
		for _, k := range []int{1, 8, 40, 56, 64, 256} {
			if k > n {
				continue
			}
			gotIdx, gotVal, err := buf.TopKN(n, k)
			if err != nil {
				buf.Release()
				t.Fatalf("n=%d k=%d: %v", n, k, err)
			}
			want := topKReference(x, n, k)
			for rank := range k {
				if int(gotIdx[rank]) != want[rank] || gotVal[rank] != x[want[rank]] {
					buf.Release()
					t.Fatalf("n=%d k=%d rank=%d: got (%d,%v), want (%d,%v)",
						n, k, rank, gotIdx[rank], gotVal[rank], want[rank], x[want[rank]])
				}
			}
		}
		buf.Release()
	}
}

func TestMetalResidentTopKNTiesAndGuards(t *testing.T) {
	if !metal.Available() {
		t.Skip("Metal unavailable")
	}
	buf, err := metal.NewDeviceBufferF32([]float32{5, 2, 5, 4, 5, 100})
	if err != nil {
		t.Fatal(err)
	}
	idx, val, err := buf.TopKN(5, 3)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int32{0, 2, 4}; !equalInt32(idx, want) {
		t.Fatalf("tie order indices=%v, want %v", idx, want)
	}
	for i, v := range val {
		if v != 5 {
			t.Fatalf("tie value[%d]=%v, want 5", i, v)
		}
	}
	for _, tc := range []struct{ n, k int }{{0, 1}, {7, 1}, {5, 0}, {5, 6}, {5, 257}} {
		if _, _, err := buf.TopKN(tc.n, tc.k); err == nil {
			t.Fatalf("TopKN(%d,%d) unexpectedly succeeded", tc.n, tc.k)
		}
	}
	buf.Release()
	if _, _, err := buf.TopKN(1, 1); err == nil {
		t.Fatal("TopKN on released buffer unexpectedly succeeded")
	}

	half, err := metal.NewDeviceBufferF16Zeros(8)
	if err != nil {
		t.Fatal(err)
	}
	defer half.Release()
	if _, _, err := half.TopKN(8, 1); err == nil {
		t.Fatal("TopKN on f16 storage unexpectedly succeeded")
	}
}

func equalInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func BenchmarkMetalResidentTopKN32K56(b *testing.B) {
	if !metal.Available() {
		b.Skip("Metal unavailable")
	}
	rng := rand.New(rand.NewPCG(32000, 0x5eed))
	x := make([]float32, 32000)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	buf, err := metal.NewDeviceBufferF32(x)
	if err != nil {
		b.Fatal(err)
	}
	defer buf.Release()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := buf.TopKN(32000, 56); err != nil {
			b.Fatal(err)
		}
	}
}
