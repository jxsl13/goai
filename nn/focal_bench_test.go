package nn_test

import (
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
)

// BenchmarkFocalLoss measures the multi-class softmax focal loss at a DETECTION-like
// shape — many rows (anchors), few classes — which is focal loss's actual regime
// (RetinaNet: ~100k anchors × a handful of classes). That is exactly where the
// per-row one-hot construction (one Unravel alloc + two dispatches per row) is a
// real fraction of the wall time next to the small-classes softmax/exp/log ops.
func benchFocal(b *testing.B, batch, classes int) {
	logits := tensor.New(tensor.F64, tensor.Shape{batch, classes})
	ld := logits.Storage().F64()
	rng := rand.New(rand.NewPCG(0x9e3779b97f4a7c15, 0xd1b54a32d192ed03))
	for i := range ld {
		ld[i] = rng.NormFloat64()
	}
	targets := tensor.New(tensor.F64, tensor.Shape{batch})
	td := targets.Storage().F64()
	for i := range td {
		td[i] = float64(i % classes)
	}
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := nn.FocalLoss(ctx, logits, targets, 2, 0.25); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFocalLossDetection(b *testing.B) { benchFocal(b, 8192, 4) }
func BenchmarkFocalLossClassify(b *testing.B)  { benchFocal(b, 1024, 100) }

// BenchmarkFocalLossVocab exercises focal cross-entropy at vocab scale, where the old [batch,classes]
// one-hot alloc + mul/sum passes dominate — the gather path eliminates them.
func BenchmarkFocalLossVocab(b *testing.B) { benchFocal(b, 256, 32000) }
