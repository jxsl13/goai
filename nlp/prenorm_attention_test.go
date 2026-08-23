package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type noPreNormAttentionBackend struct{ backend.Backend }

func (b noPreNormAttentionBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPreNormAttention || op == backend.OpPreNormAttentionBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

type preNormAttentionRecorder struct{ ops []backend.Op }

func (r *preNormAttentionRecorder) Record(op backend.Op, _, _ []*tensor.Tensor, _ backend.Attrs) {
	r.ops = append(r.ops, op)
}

func preNormAttentionTestFixture(t *testing.T) (*nlp.MHA, *nn.LayerNorm, *tensor.Tensor) {
	t.Helper()
	const dim = 8
	mha, err := nlp.NewMHA(2,
		bench.RandF32(tensor.Shape{dim, dim}, 1),
		bench.RandF32(tensor.Shape{dim, dim}, 2),
		bench.RandF32(tensor.Shape{dim, dim}, 3),
		bench.RandF32(tensor.Shape{dim, dim}, 4),
	)
	if err != nil {
		t.Fatal(err)
	}
	norm := &nn.LayerNorm{
		Gamma: bench.RandF32(tensor.Shape{dim}, 5),
		Beta:  bench.RandF32(tensor.Shape{dim}, 6),
		Eps:   1e-5,
	}
	return mha, norm, bench.RandF32(tensor.Shape{6, dim}, 7)
}

func TestForwardPreNormAttentionFallsBackToExactComposite(t *testing.T) {
	mha, norm, x := preNormAttentionTestFixture(t)
	r := &preNormAttentionRecorder{}
	ctx := (&backend.Context{Backend: noPreNormAttentionBackend{backend.Reference()}}).WithRecorder(r)
	if _, err := mha.ForwardPreNorm(ctx, x, norm, 2); err != nil {
		t.Fatal(err)
	}
	want := []backend.Op{
		backend.OpLayerNorm,
		backend.OpMatMul, backend.OpMatMul, backend.OpMatMul,
		backend.OpMHA, backend.OpMatMul, backend.OpAdd,
	}
	if len(r.ops) != len(want) {
		t.Fatalf("recorded ops=%v, want %v", r.ops, want)
	}
	for i := range want {
		if r.ops[i] != want[i] {
			t.Fatalf("recorded ops=%v, want %v", r.ops, want)
		}
	}
}

func TestForwardPreNormAttentionFeatureExclusions(t *testing.T) {
	for _, tc := range []struct {
		name    string
		exclude func(*nlp.MHA)
	}{
		{name: "causal", exclude: func(m *nlp.MHA) { m.Causal = true }},
		{name: "bias", exclude: func(m *nlp.MHA) {
			m.Bias = map[string]*tensor.Tensor{"q": bench.RandF32(tensor.Shape{8}, 20)}
		}},
		{name: "lora", exclude: func(m *nlp.MHA) { m.LoRA = map[string]*nn.LoRALinear{"q": nil} }},
		{name: "mask", exclude: func(m *nlp.MHA) {
			m.Mask = tensor.Zeros(tensor.F32, tensor.Shape{3, 3})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mha, norm, x := preNormAttentionTestFixture(t)
			tc.exclude(mha)
			r := &preNormAttentionRecorder{}
			ctx := (&backend.Context{Backend: backend.Reference()}).WithRecorder(r)
			if _, err := mha.ForwardPreNorm(ctx, x, norm, 2); err != nil {
				t.Fatal(err)
			}
			for _, op := range r.ops {
				if op == backend.OpPreNormAttention || op == backend.OpPreNormAttentionBackward {
					t.Fatalf("excluded feature recorded fused op: %v", r.ops)
				}
			}
		})
	}
}

func TestForwardPreNormAttentionRejectsNilInputs(t *testing.T) {
	mha, norm, x := preNormAttentionTestFixture(t)
	if _, err := mha.ForwardPreNorm(nil, nil, norm, 2); err == nil {
		t.Fatal("expected nil input error")
	}
	var nilMHA *nlp.MHA
	if _, err := nilMHA.ForwardPreNorm(nil, x, norm, 2); err == nil {
		t.Fatal("expected nil MHA error")
	}
	if _, err := mha.ForwardPreNorm(nil, x, nil, 2); err == nil {
		t.Fatal("expected nil LayerNorm error")
	}
}
