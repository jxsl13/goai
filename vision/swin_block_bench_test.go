package vision_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// BenchmarkSwinBlockForward isolates ONE Swin block on the token grid it sees inside the
// benchmarked model, because BenchmarkSwinForwardBatched cannot answer a question about this block.
// The windowed-attention loop is roughly 95% of the package's backend dispatches and roughly 2% of
// its arithmetic, so a whole-model benchmark dilutes it behind the projection and MLP GEMMs by
// about 5x: a genuine 3x on the loop would read there as a ~1.15x overall and get rejected as
// noise. Measuring the block on its own is what makes the win, or the absence of one, visible.
//
// The geometry matches stage 0 of BenchmarkSwinForwardBatched: an 8x8 token grid at 96 channels,
// window 4, 3 heads, so 4 windows of 16 tokens per image and 32 windows across the batch of 8.
// Both blocks are run: index 0 is W-MSA and index 1 is SW-MSA, which additionally carries the
// cross-region mask add, one more dispatch per window and head.
func BenchmarkSwinBlockForward(b *testing.B) {
	const B, C, size, classes, embedC, grid = 8, 3, 32, 10, 96, 8
	m, err := vision.NewSwin(tensor.F32, size, 4, 4, embedC, []int{2, 2}, []int{3, 6}, classes, 7,
		vision.WithSwinRelativeBias(true), vision.WithSwinChannels(C))
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))
	x := tensor.New(tensor.F32, tensor.Shape{B * grid * grid, embedC})
	xs := x.Storage().F32()
	for i := range xs {
		xs[i] = float32(rng.NormFloat64())
	}
	ctx := backend.NewContext()

	// A batched call presents B images' grids stacked on axis 0, which is what the model's own
	// batched path does; the block treats them as B*grid*grid rows and partitions windows over the
	// whole stack.
	b.Run("wmsa", func(b *testing.B) {
		blk := m.Stages[0][0]
		for range b.N {
			if _, err := blk.Forward(ctx, x, B*grid, grid); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("swmsa", func(b *testing.B) {
		blk := m.Stages[0][1]
		for range b.N {
			if _, err := blk.Forward(ctx, x, B*grid, grid); err != nil {
				b.Fatal(err)
			}
		}
	})
}
