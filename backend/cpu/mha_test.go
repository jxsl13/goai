package cpu_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// §T599/§V9: the typed MHA forward/backward match ref within ulps across the
// full attribute surface (causal, GQA/MQA, ALiBi, sliding window, KV-cache
// offset) on serial and parallel shapes, both dtypes.
func TestMHAKernelsMatchRefWithinUlps(t *testing.T) {
	cpuB := cpuBackend(t)
	refB, _ := backend.Get(backend.Ref)
	type cse struct {
		name       string
		sq, sk, dm int
		attrs      backend.AttnAttrs
	}
	cases := []cse{
		{"plain", 16, 16, 32, backend.AttnAttrs{Heads: 4}},
		{"causal", 16, 16, 32, backend.AttnAttrs{Heads: 4, Causal: true}},
		{"gqa", 16, 16, 32, backend.AttnAttrs{Heads: 4, KVHeads: 2, Causal: true}},
		{"mqa", 16, 16, 32, backend.AttnAttrs{Heads: 4, KVHeads: 1}},
		{"alibi", 16, 16, 32, backend.AttnAttrs{Heads: 4, Causal: true, ALiBi: true}},
		{"window", 16, 16, 32, backend.AttnAttrs{Heads: 4, Causal: true, Window: 5}},
		{"scale", 16, 16, 32, backend.AttnAttrs{Heads: 4, Scale: 0.7}},
		{"parallel", 96, 96, 128, backend.AttnAttrs{Heads: 8, Causal: true}},
		{"kv-decode", 1, 24, 32, backend.AttnAttrs{Heads: 4, Causal: true}}, // forward only
	}
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, tc := range cases {
			kvHeads := tc.attrs.KVHeads
			if kvHeads == 0 {
				kvHeads = tc.attrs.Heads
			}
			kvDM := kvHeads * (tc.dm / tc.attrs.Heads)
			q := tensor.Randn(dt, 1, tensor.Shape{tc.sq, tc.dm})
			k := tensor.Randn(dt, 2, tensor.Shape{tc.sk, kvDM})
			v := tensor.Randn(dt, 3, tensor.Shape{tc.sk, kvDM})
			ctx := backend.NewContext()
			got, err := backend.Execute(ctx.WithBackend(cpuB), backend.OpMHA, []*tensor.Tensor{q, k, v}, tc.attrs)
			if err != nil {
				t.Fatalf("%s fwd cpu: %v", tc.name, err)
			}
			want, err := backend.Execute(ctx.WithBackend(refB), backend.OpMHA, []*tensor.Tensor{q, k, v}, tc.attrs)
			if err != nil {
				t.Fatalf("%s fwd ref: %v", tc.name, err)
			}
			assertCloseUlps(t, got[0], want[0], fmt.Sprintf("mha/%s/%s", tc.name, dt))

			if tc.sq != tc.sk {
				continue // backward requires sq==sk (training path)
			}
			g := tensor.Randn(dt, 4, tensor.Shape{tc.sq, tc.dm})
			gotB, err := backend.Execute(ctx.WithBackend(cpuB), backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, tc.attrs)
			if err != nil {
				t.Fatalf("%s bwd cpu: %v", tc.name, err)
			}
			wantB, err := backend.Execute(ctx.WithBackend(refB), backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, tc.attrs)
			if err != nil {
				t.Fatalf("%s bwd ref: %v", tc.name, err)
			}
			for o, lbl := range []string{"dQ", "dK", "dV"} {
				assertCloseUlps(t, gotB[o], wantB[o], fmt.Sprintf("mha-bwd/%s/%s/%s", tc.name, dt, lbl))
			}
		}
	}
}

// BenchmarkMHA512 compares cpu vs ref at the torch reference shape 512×8×64
// (python_compare's MHAForward/MHAFwdBwd rows use the same geometry).
func BenchmarkMHA512(b *testing.B) {
	cpuB, _ := backend.Get(backend.CPU)
	refB, _ := backend.Get(backend.Ref)
	q := tensor.Randn(tensor.F32, 1, tensor.Shape{512, 512})
	k := tensor.Randn(tensor.F32, 2, tensor.Shape{512, 512})
	v := tensor.Randn(tensor.F32, 3, tensor.Shape{512, 512})
	g := tensor.Randn(tensor.F32, 4, tensor.Shape{512, 512})
	at := backend.AttnAttrs{Heads: 8, Causal: true}
	for name, be := range map[backend.Name]backend.Backend{backend.CPU: cpuB, backend.Ref: refB} {
		b.Run("fwd/"+string(name), func(b *testing.B) {
			ctx := backend.NewContext().WithBackend(be)
			for b.Loop() {
				if _, err := backend.Execute(ctx, backend.OpMHA, []*tensor.Tensor{q, k, v}, at); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run("bwd/"+string(name), func(b *testing.B) {
			ctx := backend.NewContext().WithBackend(be)
			for b.Loop() {
				if _, err := backend.Execute(ctx, backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, at); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// §T610/§V9: FlashAttn and Retention fwd/bwd match ref within ulps across
// dtypes, causal/GQA/block variants and serial+parallel sizes.
func TestFlashAttnRetentionMatchRefWithinUlps(t *testing.T) {
	cpuB := cpuBackend(t)
	refB, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext()
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, tc := range []struct {
			name string
			at   backend.AttnAttrs
			seq  int
		}{
			{"flash-causal", backend.AttnAttrs{Heads: 4, Causal: true, Block: 8}, 24},
			{"flash-full", backend.AttnAttrs{Heads: 2, Block: 0}, 16},
			{"flash-gqa", backend.AttnAttrs{Heads: 4, KVHeads: 2, Causal: true, Block: 4}, 96},
		} {
			dm := 32
			if tc.seq > 32 {
				dm = 128
			}
			kvHeads := tc.at.KVHeads
			if kvHeads == 0 {
				kvHeads = tc.at.Heads
			}
			dkv := kvHeads * (dm / tc.at.Heads)
			q := tensor.Randn(dt, 1, tensor.Shape{tc.seq, dm})
			k := tensor.Randn(dt, 2, tensor.Shape{tc.seq, dkv})
			v := tensor.Randn(dt, 3, tensor.Shape{tc.seq, dkv})
			got, err := backend.Execute(ctx.WithBackend(cpuB), backend.OpFlashAttn, []*tensor.Tensor{q, k, v}, tc.at)
			if err != nil {
				t.Fatalf("%s cpu: %v", tc.name, err)
			}
			want, err := backend.Execute(ctx.WithBackend(refB), backend.OpFlashAttn, []*tensor.Tensor{q, k, v}, tc.at)
			if err != nil {
				t.Fatalf("%s ref: %v", tc.name, err)
			}
			assertCloseUlps(t, got[0], want[0], "flashattn/"+tc.name+"/"+dt.String())
		}
		for _, sz := range []struct{ l, dk, dv int }{{12, 8, 6}, {96, 32, 48}} {
			q := tensor.Randn(dt, 4, tensor.Shape{sz.l, sz.dk})
			k := tensor.Randn(dt, 5, tensor.Shape{sz.l, sz.dk})
			v := tensor.Randn(dt, 6, tensor.Shape{sz.l, sz.dv})
			g := tensor.Randn(dt, 7, tensor.Shape{sz.l, sz.dv})
			at := backend.RetentionAttrs{Gamma: 0.9375}
			got, err := backend.Execute(ctx.WithBackend(cpuB), backend.OpRetention, []*tensor.Tensor{q, k, v}, at)
			if err != nil {
				t.Fatal(err)
			}
			want, err := backend.Execute(ctx.WithBackend(refB), backend.OpRetention, []*tensor.Tensor{q, k, v}, at)
			if err != nil {
				t.Fatal(err)
			}
			assertCloseUlps(t, got[0], want[0], fmt.Sprintf("retention/%v/%s", sz, dt))
			gotB, err := backend.Execute(ctx.WithBackend(cpuB), backend.OpRetentionBackward, []*tensor.Tensor{q, k, v, g}, at)
			if err != nil {
				t.Fatal(err)
			}
			wantB, err := backend.Execute(ctx.WithBackend(refB), backend.OpRetentionBackward, []*tensor.Tensor{q, k, v, g}, at)
			if err != nil {
				t.Fatal(err)
			}
			for o, lbl := range []string{"dQ", "dK", "dV"} {
				assertCloseUlps(t, gotB[o], wantB[o], fmt.Sprintf("retention-bwd/%v/%s/%s", sz, dt, lbl))
			}
		}
	}
}
