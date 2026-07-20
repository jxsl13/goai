package vision_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
	"github.com/jxsl13/goai/vision"
)

// TestViTForwardBatchedParity: the batched ViT.Forward ([B,C,H,W] → one packed [B·(N+1),D] encoder
// pass) must equal looping the per-image path (forwardOne, via the [C,H,W] Ndim==3 branch) and
// concatenating — every batched op is row-wise, so the packing changes only GEMM shapes, not values.
// F64 for an exactness assertion (same per-row reduction order).
func TestViTForwardBatchedParity(t *testing.T) {
	const B, C, size, classes = 4, 3, 8, 3
	m, err := vision.NewViT(C, size, classes, 7,
		vision.WithViTDtype(tensor.F64), vision.WithViTDim(16), vision.WithViTDepth(2),
		vision.WithViTHeads(2), vision.WithViTPatch(4))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	data := make([]float64, B*C*size*size)
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	per := C * size * size

	ctx := backend.NewContext()
	xb := tensor.New(tensor.F64, tensor.Shape{B, C, size, size})
	copy(xb.Storage().F64(), data)
	got, err := m.Forward(ctx, xb) // [B, classes]
	if err != nil {
		t.Fatal(err)
	}

	// per-image reference (unchanged forwardOne path)
	refRows := make([]*tensor.Tensor, B)
	for b := range B {
		one := tensor.New(tensor.F64, tensor.Shape{C, size, size})
		copy(one.Storage().F64(), data[b*per:(b+1)*per])
		r, err := m.Forward(ctx, one) // [1, classes]
		if err != nil {
			t.Fatal(err)
		}
		refRows[b] = r
	}
	refT, err := backend.Execute(ctx, backend.OpConcat, refRows, backend.ConcatAttrs{Axis: 0})
	if err != nil {
		t.Fatal(err)
	}
	ref := refT[0]

	if got.Shape()[0] != B || got.Shape()[1] != classes {
		t.Fatalf("batched logits shape %v, want [%d,%d]", got.Shape(), B, classes)
	}
	ga, ra := got.Contiguous().Storage().F64(), ref.Contiguous().Storage().F64()
	var num, den float64
	for i := range ga {
		d := ga[i] - ra[i]
		num += d * d
		den += ra[i] * ra[i]
	}
	rel := math.Sqrt(num / (den + 1e-300))
	if rel > 1e-9 {
		t.Fatalf("batched vs per-image ViT logits rel-RMS %.3e too high", rel)
	}
	t.Logf("batched vs per-image ViT logits: rel-RMS %.3e — parity", rel)
}

// BenchmarkViTForwardBatched is the CURRENT-SYSTEM (Linux, no darwin tag) pre/post for the T908
// batching win: it times the OLD per-image loop (forwardOne over each [C,H,W]) against the NEW
// batched Forward on the SAME [B,C,H,W] input, on the pure-Go/CPU backend available here. img/s =
// batch / step. Run: `GOEXPERIMENT=simd go test -tags=simd -run=NONE -bench=ViTForwardBatched ./vision/`.
func BenchmarkViTForwardBatched(b *testing.B) {
	const B, C, size, classes = 8, 3, 32, 10
	m, err := vision.NewViT(C, size, classes, 7,
		vision.WithViTDtype(tensor.F32), vision.WithViTDim(128), vision.WithViTDepth(4),
		vision.WithViTHeads(4), vision.WithViTPatch(4))
	if err != nil {
		b.Fatal(err)
	}
	rng := rand.New(rand.NewSource(1))
	data := make([]float32, B*C*size*size)
	for i := range data {
		data[i] = float32(rng.NormFloat64())
	}
	per := C * size * size
	ctx := backend.NewContext()
	xb := tensor.New(tensor.F32, tensor.Shape{B, C, size, size})
	copy(xb.Storage().F32(), data)
	// per-image [C,H,W] inputs reused across iterations
	ones := make([]*tensor.Tensor, B)
	for i := range ones {
		o := tensor.New(tensor.F32, tensor.Shape{C, size, size})
		copy(o.Storage().F32(), data[i*per:(i+1)*per])
		ones[i] = o
	}

	b.Run("perimage", func(b *testing.B) {
		for range b.N {
			for i := range ones {
				if _, err := m.Forward(ctx, ones[i]); err != nil {
					b.Fatal(err)
				}
			}
		}
		b.ReportMetric(float64(B)*float64(b.N)/b.Elapsed().Seconds(), "img/s")
	})
	b.Run("batched", func(b *testing.B) {
		for range b.N {
			if _, err := m.Forward(ctx, xb); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(B)*float64(b.N)/b.Elapsed().Seconds(), "img/s")
	})
}
