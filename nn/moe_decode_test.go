package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestSparseMoEForwardDecodeMatchesDense proves the sparse decode path (evaluate only the
// top-k routed experts) is numerically identical to the dense Forward (evaluate all E,
// mask, renormalized-combine) — the invariant that lets the KV-cached MoE decode use the
// cheaper path without changing outputs. Checked for single-token (the decode case) and a
// small multi-token batch, and for both the unbiased and load-free-balanced routers.
func TestSparseMoEForwardDecodeMatchesDense(t *testing.T) {
	const dim, hidden, experts, topK = 8, 16, 6, 2
	for _, tc := range []struct {
		name string
		opts []nn.SparseMoEOption
		toks int
	}{
		{"unbiased/1tok", nil, 1},
		{"unbiased/5tok", nil, 5},
		{"balanced/1tok", []nn.SparseMoEOption{nn.WithLossFreeBalance(0.02)}, 1},
		{"balanced/3tok", []nn.SparseMoEOption{nn.WithLossFreeBalance(0.02)}, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			moe := nn.NewSparseMoE(tensor.F64, dim, hidden, experts, topK, 7, tc.opts...)
			x := tensor.New(tensor.F64, tensor.Shape{tc.toks, dim})
			for i := range x.Storage().F64() {
				x.Storage().F64()[i] = math.Sin(float64(i)*0.7) * 1.3
			}
			dense, _, err := moe.Forward(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			sparse, _, err := moe.ForwardDecode(backend.NewContext(), x)
			if err != nil {
				t.Fatal(err)
			}
			if !sparse.Shape().Equal(dense.Shape()) {
				t.Fatalf("shape %v != dense %v", sparse.Shape(), dense.Shape())
			}
			var maxAbs float64
			for i := range tc.toks {
				for j := range dim {
					if d := math.Abs(sparse.AtF64(i, j) - dense.AtF64(i, j)); d > maxAbs {
						maxAbs = d
					}
				}
			}
			if maxAbs > 1e-12 {
				t.Fatalf("ForwardDecode diverges from dense Forward: %.3e", maxAbs)
			}
		})
	}
}

// TestSparseMoEForwardDecodeF32 guards a dtype footgun: ForwardDecode built its per-expert
// weight tensor at x.Dtype() but then took weight.Storage().F64(), which PANICS on F32/F16
// storage — so the standard inference dtype (F32) crashed the decode path. The write is now
// the dtype-agnostic SetF64 (matching the dense Forward). Assert F32 decode runs and equals
// the dense F32 Forward.
func TestSparseMoEForwardDecodeF32(t *testing.T) {
	const dim, hidden, experts, topK = 8, 16, 6, 2
	moe := nn.NewSparseMoE(tensor.F32, dim, hidden, experts, topK, 7)
	x := tensor.New(tensor.F32, tensor.Shape{3, dim})
	for i := range x.Storage().F32() {
		x.Storage().F32()[i] = float32(math.Sin(float64(i)*0.7) * 1.3)
	}
	dense, _, err := moe.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	sparse, _, err := moe.ForwardDecode(backend.NewContext(), x) // previously panicked on F32
	if err != nil {
		t.Fatal(err)
	}
	var maxAbs float64
	for i := range 3 {
		for j := range dim {
			if d := math.Abs(sparse.AtF64(i, j) - dense.AtF64(i, j)); d > maxAbs {
				maxAbs = d
			}
		}
	}
	if maxAbs > 1e-5 { // F32 tolerance
		t.Fatalf("F32 ForwardDecode diverges from dense Forward: %.3e", maxAbs)
	}
}

// BenchmarkSparseMoEDecode measures the dense-vs-sparse decode cost at one token for a
// Mixtral-like (8 experts, top-2) and a Qwen-MoE-like (64 experts, top-8) config — the
// §V22 evidence for the sparse-decode win (expert FFNs are the decode weight-BW floor).
func benchDecode(b *testing.B, experts, topK int, sparse bool) {
	const dim, hidden = 512, 1408
	moe := nn.NewSparseMoE(tensor.F64, dim, hidden, experts, topK, 7)
	x := tensor.New(tensor.F64, tensor.Shape{1, dim})
	for i := range x.Storage().F64() {
		x.Storage().F64()[i] = math.Sin(float64(i) * 0.1)
	}
	b.ResetTimer()
	for range b.N {
		if sparse {
			moe.ForwardDecode(backend.NewContext(), x)
		} else {
			moe.Forward(backend.NewContext(), x)
		}
	}
}
func BenchmarkMoEDecodeMixtralDense(b *testing.B)  { benchDecode(b, 8, 2, false) }
func BenchmarkMoEDecodeMixtralSparse(b *testing.B) { benchDecode(b, 8, 2, true) }
func BenchmarkMoEDecodeQwenDense(b *testing.B)     { benchDecode(b, 64, 8, false) }
func BenchmarkMoEDecodeQwenSparse(b *testing.B)    { benchDecode(b, 64, 8, true) }
