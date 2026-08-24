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

type noPreNormTransformerBlockBackend struct{ backend.Backend }

func (b noPreNormTransformerBlockBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPreNormTransformerBlock || op == backend.OpPreNormTransformerBlockBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

type noPreNormTransformerStackBackend struct{ backend.Backend }

func (b noPreNormTransformerStackBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == backend.OpPreNormTransformerStack || op == backend.OpPreNormTransformerStackBackward {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

type noPreNormTransformerBlockDirectionBackend struct {
	backend.Backend
	disabled backend.Op
}

func (b noPreNormTransformerBlockDirectionBackend) Kernel(op backend.Op, dt tensor.Dtype) (backend.Kernel, bool) {
	if op == b.disabled {
		return nil, false
	}
	return b.Backend.Kernel(op, dt)
}

func preNormTransformerBlockFixture(t *testing.T) (*nlp.MHA, *nn.LayerNorm, *nn.LayerNorm, *nn.Linear, *nn.Linear, *tensor.Tensor) {
	t.Helper()
	const dim, hidden = 8, 12
	mha, err := nlp.NewMHA(2,
		bench.RandF32(tensor.Shape{dim, dim}, 31),
		bench.RandF32(tensor.Shape{dim, dim}, 32),
		bench.RandF32(tensor.Shape{dim, dim}, 33),
		bench.RandF32(tensor.Shape{dim, dim}, 34),
	)
	if err != nil {
		t.Fatal(err)
	}
	norm1 := &nn.LayerNorm{Gamma: bench.RandF32(tensor.Shape{dim}, 35), Beta: bench.RandF32(tensor.Shape{dim}, 36), Eps: 2e-5}
	norm2 := &nn.LayerNorm{Gamma: bench.RandF32(tensor.Shape{dim}, 37), Beta: bench.RandF32(tensor.Shape{dim}, 38), Eps: 3e-5}
	up := &nn.Linear{W: bench.RandF32(tensor.Shape{dim, hidden}, 39), B: bench.RandF32(tensor.Shape{hidden}, 40)}
	down := &nn.Linear{W: bench.RandF32(tensor.Shape{hidden, dim}, 41), B: bench.RandF32(tensor.Shape{dim}, 42)}
	return mha, norm1, norm2, up, down, bench.RandF32(tensor.Shape{6, dim}, 43)
}

func TestForwardPreNormTransformerBlockUsesCompleteBoundary(t *testing.T) {
	mha, norm1, norm2, up, down, x := preNormTransformerBlockFixture(t)
	r := &preNormAttentionRecorder{}
	ctx := (&backend.Context{Backend: backend.Reference()}).WithRecorder(r)
	if _, err := nlp.ForwardPreNormTransformerBlock(ctx, x, mha, norm1, norm2, up, down, 2); err != nil {
		t.Fatal(err)
	}
	if len(r.ops) != 1 || r.ops[0] != backend.OpPreNormTransformerBlock {
		t.Fatalf("recorded ops=%v, want one complete-block op", r.ops)
	}
}

func TestForwardPreNormTransformerBlockFallsBackToMergedBoundaries(t *testing.T) {
	for _, tc := range []struct {
		name    string
		backend backend.Backend
	}{
		{name: "both-directions-missing", backend: noPreNormTransformerBlockBackend{backend.Reference()}},
		{name: "forward-missing", backend: noPreNormTransformerBlockDirectionBackend{
			Backend: backend.Reference(), disabled: backend.OpPreNormTransformerBlock,
		}},
		{name: "backward-missing", backend: noPreNormTransformerBlockDirectionBackend{
			Backend: backend.Reference(), disabled: backend.OpPreNormTransformerBlockBackward,
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mha, norm1, norm2, up, down, x := preNormTransformerBlockFixture(t)
			r := &preNormAttentionRecorder{}
			ctx := (&backend.Context{Backend: tc.backend}).WithRecorder(r)
			if _, err := nlp.ForwardPreNormTransformerBlock(ctx, x, mha, norm1, norm2, up, down, 2); err != nil {
				t.Fatal(err)
			}
			want := []backend.Op{backend.OpPreNormAttention, backend.OpPreNormFFN}
			if len(r.ops) != len(want) {
				t.Fatalf("recorded ops=%v, want %v", r.ops, want)
			}
			for i := range want {
				if r.ops[i] != want[i] {
					t.Fatalf("recorded ops=%v, want %v", r.ops, want)
				}
			}
			for _, op := range r.ops {
				if op == backend.OpPreNormTransformerBlock || op == backend.OpPreNormTransformerBlockBackward {
					t.Fatalf("fallback recorded complete-block op: %v", r.ops)
				}
			}
		})
	}
}

func TestForwardPreNormTransformerBlockFeatureExclusions(t *testing.T) {
	for _, tc := range []struct {
		name    string
		exclude func(*nlp.MHA)
	}{
		{name: "causal", exclude: func(m *nlp.MHA) { m.Causal = true }},
		{name: "bias", exclude: func(m *nlp.MHA) { m.Bias = map[string]*tensor.Tensor{"q": bench.RandF32(tensor.Shape{8}, 50)} }},
		{name: "lora", exclude: func(m *nlp.MHA) { m.LoRA = map[string]*nn.LoRALinear{"q": nil} }},
		{name: "mask", exclude: func(m *nlp.MHA) { m.Mask = tensor.Zeros(tensor.F32, tensor.Shape{3, 3}) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mha, norm1, norm2, up, down, x := preNormTransformerBlockFixture(t)
			tc.exclude(mha)
			r := &preNormAttentionRecorder{}
			ctx := (&backend.Context{Backend: backend.Reference()}).WithRecorder(r)
			if _, err := nlp.ForwardPreNormTransformerBlock(ctx, x, mha, norm1, norm2, up, down, 2); err != nil {
				t.Fatal(err)
			}
			for _, op := range r.ops {
				if op == backend.OpPreNormTransformerBlock || op == backend.OpPreNormTransformerBlockBackward {
					t.Fatalf("excluded feature recorded complete-block op: %v", r.ops)
				}
			}
		})
	}
}

func TestForwardPreNormTransformerBlockRejectsNilInputs(t *testing.T) {
	mha, norm1, norm2, up, down, x := preNormTransformerBlockFixture(t)
	if _, err := nlp.ForwardPreNormTransformerBlock(nil, nil, mha, norm1, norm2, up, down, 2); err == nil {
		t.Fatal("expected nil input error")
	}
	if _, err := nlp.ForwardPreNormTransformerBlock(nil, x, nil, norm1, norm2, up, down, 2); err == nil {
		t.Fatal("expected nil attention error")
	}
	if _, err := nlp.ForwardPreNormTransformerBlock(nil, x, mha, nil, norm2, up, down, 2); err == nil {
		t.Fatal("expected nil normalization error")
	}
}

func preNormTransformerStackFixture(t *testing.T) ([]nlp.PreNormTransformerBlock, *tensor.Tensor) {
	t.Helper()
	blocks := make([]nlp.PreNormTransformerBlock, 2)
	var x *tensor.Tensor
	for i := range blocks {
		mha, norm1, norm2, up, down, blockX := preNormTransformerBlockFixture(t)
		if x == nil {
			x = blockX
		}
		blocks[i] = nlp.PreNormTransformerBlock{
			Attention: mha, Norm1: norm1, Norm2: norm2, Up: up, Down: down,
		}
	}
	return blocks, x
}

func TestForwardPreNormTransformerStackUsesCompleteStackBoundary(t *testing.T) {
	blocks, x := preNormTransformerStackFixture(t)
	r := &preNormAttentionRecorder{}
	ctx := (&backend.Context{Backend: backend.Reference()}).WithRecorder(r)
	if _, err := nlp.ForwardPreNormTransformerStack(ctx, x, blocks, 2); err != nil {
		t.Fatal(err)
	}
	if len(r.ops) != 1 || r.ops[0] != backend.OpPreNormTransformerStack {
		t.Fatalf("recorded ops=%v, want one complete-stack op", r.ops)
	}
}

func TestForwardPreNormTransformerStackFallsBackToCompleteBlocks(t *testing.T) {
	blocks, x := preNormTransformerStackFixture(t)
	r := &preNormAttentionRecorder{}
	ctx := (&backend.Context{Backend: noPreNormTransformerStackBackend{backend.Reference()}}).WithRecorder(r)
	if _, err := nlp.ForwardPreNormTransformerStack(ctx, x, blocks, 2); err != nil {
		t.Fatal(err)
	}
	want := []backend.Op{backend.OpPreNormTransformerBlock, backend.OpPreNormTransformerBlock}
	if len(r.ops) != len(want) {
		t.Fatalf("recorded ops=%v, want %v", r.ops, want)
	}
	for i := range want {
		if r.ops[i] != want[i] {
			t.Fatalf("recorded ops=%v, want %v", r.ops, want)
		}
	}
}

func TestForwardPreNormTransformerStackRequiresUniformGeometry(t *testing.T) {
	blocks, x := preNormTransformerStackFixture(t)
	blocks[1].Attention.Heads = 1
	r := &preNormAttentionRecorder{}
	ctx := (&backend.Context{Backend: backend.Reference()}).WithRecorder(r)
	if _, err := nlp.ForwardPreNormTransformerStack(ctx, x, blocks, 2); err != nil {
		t.Fatal(err)
	}
	for _, op := range r.ops {
		if op == backend.OpPreNormTransformerStack || op == backend.OpPreNormTransformerStackBackward {
			t.Fatalf("nonuniform stack recorded fused stack op: %v", r.ops)
		}
	}
}
