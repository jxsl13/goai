package cpu_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Transformer-shaped benchmarks for the norm / RoPE / attention hot path
// (seq 128/512, dmodel 2048, heads 32, kv_heads 4, head_dim 64) — the shapes a
// small LLaMA-style block actually runs. All go through backend.Execute on the
// cpu backend so dispatch + output allocation are included (what training pays).

func benchOp(b *testing.B, op backend.Op, in []*tensor.Tensor, attrs backend.Attrs) {
	b.Helper()
	cpuB, _ := backend.Get(backend.CPU)
	ctx := backend.NewContext().WithBackend(cpuB)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := backend.Execute(ctx, op, in, attrs); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNormAttn(b *testing.B) {
	const dm = 2048
	for _, seq := range []int{128, 512} {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			x := tensor.Randn(dt, 1, tensor.Shape{seq, dm})
			gamma := tensor.Randn(dt, 2, tensor.Shape{dm})
			beta := tensor.Randn(dt, 3, tensor.Shape{dm})
			g := tensor.Randn(dt, 4, tensor.Shape{seq, dm})
			tag := fmt.Sprintf("%s/seq%d", dt, seq)
			b.Run("RMSNorm/"+tag, func(b *testing.B) {
				benchOp(b, backend.OpRMSNorm, []*tensor.Tensor{x, gamma}, nil)
			})
			b.Run("LayerNorm/"+tag, func(b *testing.B) {
				benchOp(b, backend.OpLayerNorm, []*tensor.Tensor{x, gamma, beta}, nil)
			})
			b.Run("RMSNormBwd/"+tag, func(b *testing.B) {
				benchOp(b, backend.OpRMSNormBackward, []*tensor.Tensor{x, gamma, g}, nil)
			})
			b.Run("LayerNormBwd/"+tag, func(b *testing.B) {
				_ = beta
				benchOp(b, backend.OpLayerNormBackward, []*tensor.Tensor{x, gamma, g}, nil)
			})
			b.Run("RoPE/"+tag+"/h32", func(b *testing.B) {
				benchOp(b, backend.OpRoPE, []*tensor.Tensor{x}, backend.RoPEAttrs{Heads: 32})
			})
			b.Run("RoPEBwd/"+tag+"/h32", func(b *testing.B) {
				benchOp(b, backend.OpRoPEBackward, []*tensor.Tensor{g}, backend.RoPEAttrs{Heads: 32})
			})
		}
	}
}

func BenchmarkAttention(b *testing.B) {
	const (
		dm      = 2048
		heads   = 32
		kvHeads = 4
	)
	dk := dm / heads
	kvDM := kvHeads * dk
	for _, seq := range []int{128, 512} {
		q := tensor.Randn(tensor.F32, 1, tensor.Shape{seq, dm})
		k := tensor.Randn(tensor.F32, 2, tensor.Shape{seq, kvDM})
		v := tensor.Randn(tensor.F32, 3, tensor.Shape{seq, kvDM})
		g := tensor.Randn(tensor.F32, 4, tensor.Shape{seq, dm})
		gqa := backend.AttnAttrs{Heads: heads, KVHeads: kvHeads, Causal: true}
		tag := fmt.Sprintf("f32/seq%d", seq)
		b.Run("MHAFwdGQA/"+tag, func(b *testing.B) {
			benchOp(b, backend.OpMHA, []*tensor.Tensor{q, k, v}, gqa)
		})
		b.Run("MHABwdGQA/"+tag, func(b *testing.B) {
			benchOp(b, backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, gqa)
		})
		b.Run("FlashAttnGQA/"+tag, func(b *testing.B) {
			benchOp(b, backend.OpFlashAttn, []*tensor.Tensor{q, k, v},
				backend.AttnAttrs{Heads: heads, KVHeads: kvHeads, Causal: true, Block: 64})
		})
		// Full MHA (kv_heads == heads) at the same geometry: the classic case.
		kf := tensor.Randn(tensor.F32, 5, tensor.Shape{seq, dm})
		vf := tensor.Randn(tensor.F32, 6, tensor.Shape{seq, dm})
		b.Run("MHAFwdFull/"+tag, func(b *testing.B) {
			benchOp(b, backend.OpMHA, []*tensor.Tensor{q, kf, vf},
				backend.AttnAttrs{Heads: heads, Causal: true})
		})
	}
	// Single-token decode against a 512-entry KV cache (inference hot loop).
	q1 := tensor.Randn(tensor.F32, 7, tensor.Shape{1, dm})
	k1 := tensor.Randn(tensor.F32, 8, tensor.Shape{512, kvDM})
	v1 := tensor.Randn(tensor.F32, 9, tensor.Shape{512, kvDM})
	b.Run("MHADecodeGQA/f32/kv512", func(b *testing.B) {
		benchOp(b, backend.OpMHA, []*tensor.Tensor{q1, k1, v1},
			backend.AttnAttrs{Heads: heads, KVHeads: kvHeads, Causal: true})
	})
}
