package nlp

import (
	"math"
	"runtime"
	"testing"
)

// BenchmarkTurboQuantReconstructMany times the batched reconstruction (Keys/Values -> rows)
// over a realistic KV-cache length, where each row is an independent O(m log m) inverse
// rotation + QJL residual decode. The single-row BenchmarkTurboQuantReconstruct cannot show
// the batched path's parallelism.
func benchTQReconstructMany(b *testing.B, dim, rows int) {
	c, err := NewTurboQuantKVCache(dim, 2, 1)
	if err != nil {
		b.Fatal(err)
	}
	row := make([]float64, dim)
	for r := 0; r < rows; r++ {
		for i := range row {
			row[i] = math.Sin(float64(r*13+i*7)) + 0.1
		}
		if err := c.Append(row, row); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if out := c.Keys(); len(out) != rows {
			b.Fatalf("got %d rows, want %d", len(out), rows)
		}
	}
}

func BenchmarkTurboQuantReconstructMany_128x512(b *testing.B)  { benchTQReconstructMany(b, 128, 512) }
func BenchmarkTurboQuantReconstructMany_256x1024(b *testing.B) { benchTQReconstructMany(b, 256, 1024) }

// TestTurboQuantRowsParallelBitExact locks that fanning rows() over GOMAXPROCS is byte-for-byte
// identical to the serial reconstruction (each row's reconstruct is deterministic and writes a
// disjoint output row).
func TestTurboQuantRowsParallelBitExact(t *testing.T) {
	runtimeGOMAXPROCS := runtime.GOMAXPROCS
	prev := runtimeGOMAXPROCS(1)
	defer runtimeGOMAXPROCS(prev)

	for _, dim := range []int{96, 128, 256} {
		c, err := NewTurboQuantKVCache(dim, 2, 42)
		if err != nil {
			t.Fatal(err)
		}
		const rows = 500
		row := make([]float64, dim)
		for r := 0; r < rows; r++ {
			for i := range row {
				row[i] = math.Sin(float64(r*11+i*5)) - 0.2
			}
			if err := c.Append(row, row); err != nil {
				t.Fatal(err)
			}
		}
		runtimeGOMAXPROCS(1)
		serial := c.Keys()
		runtimeGOMAXPROCS(prev)
		par := c.Keys()
		for r := range serial {
			for i := range serial[r] {
				if serial[r][i] != par[r][i] {
					t.Fatalf("dim=%d row=%d col=%d: serial %v != parallel %v", dim, r, i, serial[r][i], par[r][i])
				}
			}
		}
	}
}
