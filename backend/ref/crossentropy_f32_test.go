package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestCrossEntropyF32Parity exercises the devirtualised F32 fast paths (§T646) of
// both OpCrossEntropy forward (per-row stable softmax cross-entropy, mean over the
// batch, + optional label-smoothing and PaLM z-loss) and OpCrossEntropyBackward
// (dz = g·(p − q' + 2·zl·lse·p)/b). The existing CE suite only covers F64, so this
// guards F32 ≡ float32(F64) (≤1e-3): the F32 kernels read logits as float64, keep the
// per-row max / softmax sum / lse / batch accumulator in float64, and only round the
// STORED result to float32, so they must match the F64 reference rounded to float32.
func TestCrossEntropyF32Parity(t *testing.T) {
	ref, ok := backend.Get(backend.Ref)
	if !ok {
		t.Fatal("ref backend not registered")
	}
	ctx := backend.NewContext().WithBackend(ref)

	const b, c = 4, 6
	zd := make([]float64, b*c)
	for i := range zd {
		// spread with a large-magnitude entry to exercise the max-shift stability.
		zd[i] = float64(i)*0.53 - 4.7
	}
	zd[2*c+1] = 22.0 // big logit in row 2 to stress exp range
	td := []float64{0, 3, 5, 1}

	zf := make([]float32, len(zd))
	for i := range zd {
		zf[i] = float32(zd[i])
	}
	tf := make([]float32, len(td))
	for i := range td {
		tf[i] = float32(td[i])
	}

	// attribute variants: plain, label-smoothing, z-loss, and both together.
	variants := []struct {
		name  string
		attrs backend.Attrs
	}{
		{"plain", nil},
		{"smooth", backend.CrossEntropyAttrs{LabelSmoothing: 0.1}},
		{"zloss", backend.CrossEntropyAttrs{ZLoss: 1e-3}},
		{"smooth+zloss", backend.CrossEntropyAttrs{LabelSmoothing: 0.1, ZLoss: 1e-3}},
	}

	for _, v := range variants {
		t.Run(v.name, func(t *testing.T) {
			z64 := tensor.FromFloat64(tensor.Shape{b, c}, zd)
			tg64 := tensor.FromFloat64(tensor.Shape{b}, td)
			z32 := tensor.FromFloat32(tensor.Shape{b, c}, zf)
			tg32 := tensor.FromFloat32(tensor.Shape{b}, tf)

			// ---- Forward ----
			l64, err := backend.Execute(ctx, backend.OpCrossEntropy, []*tensor.Tensor{z64, tg64}, v.attrs)
			if err != nil {
				t.Fatalf("crossentropy f64 forward: %v", err)
			}
			l32, err := backend.Execute(ctx, backend.OpCrossEntropy, []*tensor.Tensor{z32, tg32}, v.attrs)
			if err != nil {
				t.Fatalf("crossentropy f32 forward: %v", err)
			}
			if l32[0].Dtype() != tensor.F32 {
				t.Fatalf("forward: expected F32 output, got %v", l32[0].Dtype())
			}
			got := l32[0].AtF64()
			want := float64(float32(l64[0].AtF64()))
			if math.Abs(got-want) > 1e-3 {
				t.Errorf("forward loss: f32=%v want float32(f64)=%v", got, want)
			}

			// ---- Backward ----
			g64 := tensor.FromFloat64(tensor.Shape{}, []float64{1.7})
			g32 := tensor.FromFloat32(tensor.Shape{}, []float32{1.7})

			dz64, err := backend.Execute(ctx, backend.OpCrossEntropyBackward, []*tensor.Tensor{z64, tg64, g64}, v.attrs)
			if err != nil {
				t.Fatalf("crossentropy-backward f64: %v", err)
			}
			dz32, err := backend.Execute(ctx, backend.OpCrossEntropyBackward, []*tensor.Tensor{z32, tg32, g32}, v.attrs)
			if err != nil {
				t.Fatalf("crossentropy-backward f32: %v", err)
			}
			if dz32[0].Dtype() != tensor.F32 {
				t.Fatalf("backward: expected F32 output, got %v", dz32[0].Dtype())
			}
			for i := range b {
				for j := range c {
					gotG := dz32[0].AtF64(i, j)
					wantG := float64(float32(dz64[0].AtF64(i, j)))
					if math.Abs(gotG-wantG) > 1e-3 {
						t.Errorf("backward dz(%d,%d): f32=%v want float32(f64)=%v", i, j, gotG, wantG)
					}
				}
			}
		})
	}
}
