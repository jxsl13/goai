package cpu_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// Edge-case cpu≡ref parity probes for corners the main matrices skip:
// sliding-window attention during KV-cache decode (off>0 shifts the window),
// ALiBi during decode (bias argument j−(off+i)), rank-3 norm inputs (rows =
// product of leading dims), and a GQA shape whose kv group count (4) is far
// below GOMAXPROCS — the shape that exposed the backward's parallelism gap.
func TestAttnNormEdgeCasesMatchRef(t *testing.T) {
	cpuB := cpuBackend(t)
	refB, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext()

	t.Run("mha-window-decode", func(t *testing.T) {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			q := tensor.Randn(dt, 31, tensor.Shape{3, 32}) // 3 new tokens
			for _, at := range []backend.AttnAttrs{
				{Heads: 4, Causal: true, Window: 7},
				{Heads: 4, Causal: true, Window: 7, ALiBi: true},
				{Heads: 4, KVHeads: 2, Causal: true, Window: 5},
			} {
				kvHeads := at.KVHeads
				if kvHeads == 0 {
					kvHeads = at.Heads
				}
				kvDM := kvHeads * (32 / at.Heads)
				k := tensor.Randn(dt, 32, tensor.Shape{24, kvDM})
				v := tensor.Randn(dt, 33, tensor.Shape{24, kvDM})
				got, err := backend.Execute(ctx.WithBackend(cpuB), backend.OpMHA, []*tensor.Tensor{q, k, v}, at)
				if err != nil {
					t.Fatal(err)
				}
				want, err := backend.Execute(ctx.WithBackend(refB), backend.OpMHA, []*tensor.Tensor{q, k, v}, at)
				if err != nil {
					t.Fatal(err)
				}
				assertCloseUlps(t, got[0], want[0], fmt.Sprintf("mha-window-decode/%s/w%d-alibi%v", dt, at.Window, at.ALiBi))
			}
		}
	})

	t.Run("norm-rank3", func(t *testing.T) {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			x := tensor.Randn(dt, 41, tensor.Shape{4, 9, 33})
			gamma := tensor.Randn(dt, 42, tensor.Shape{33})
			beta := tensor.Randn(dt, 43, tensor.Shape{33})
			g := tensor.Randn(dt, 44, tensor.Shape{4, 9, 33})
			for _, tc := range []struct {
				name string
				op   backend.Op
				in   []*tensor.Tensor
			}{
				{"rms", backend.OpRMSNorm, []*tensor.Tensor{x, gamma}},
				{"ln", backend.OpLayerNorm, []*tensor.Tensor{x, gamma, beta}},
				{"rms-bwd", backend.OpRMSNormBackward, []*tensor.Tensor{x, gamma, g}},
				{"ln-bwd", backend.OpLayerNormBackward, []*tensor.Tensor{x, gamma, g}},
			} {
				got, err := backend.Execute(ctx.WithBackend(cpuB), tc.op, tc.in, nil)
				if err != nil {
					t.Fatal(err)
				}
				want, err := backend.Execute(ctx.WithBackend(refB), tc.op, tc.in, nil)
				if err != nil {
					t.Fatal(err)
				}
				for o := range want {
					assertCloseUlps(t, got[o], want[o], fmt.Sprintf("norm-rank3/%s/%s/out%d", tc.name, dt, o))
				}
			}
		}
	})

	// Regression guard for the head-parallel backward rewrite: a GQA shape big
	// enough to actually take the parallel path (the 16×16 matrix cases stay
	// under the serial threshold), with kv groups ≪ heads and a window.
	t.Run("mha-bwd-gqa-parallel", func(t *testing.T) {
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			const seq, dm = 96, 256
			at := backend.AttnAttrs{Heads: 16, KVHeads: 4, Causal: true, Window: 33}
			q := tensor.Randn(dt, 51, tensor.Shape{seq, dm})
			k := tensor.Randn(dt, 52, tensor.Shape{seq, 4 * (dm / 16)})
			v := tensor.Randn(dt, 53, tensor.Shape{seq, 4 * (dm / 16)})
			g := tensor.Randn(dt, 54, tensor.Shape{seq, dm})
			got, err := backend.Execute(ctx.WithBackend(cpuB), backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, at)
			if err != nil {
				t.Fatal(err)
			}
			want, err := backend.Execute(ctx.WithBackend(refB), backend.OpMHABackward, []*tensor.Tensor{q, k, v, g}, at)
			if err != nil {
				t.Fatal(err)
			}
			for o, lbl := range []string{"dQ", "dK", "dV"} {
				assertCloseUlps(t, got[o], want[o], fmt.Sprintf("mha-bwd-gqa-parallel/%s/%s", dt, lbl))
			}
		}
	})
}
