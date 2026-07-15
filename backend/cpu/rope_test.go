package cpu_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The cpu RoPE forward/backward match ref within ulps across the full
// attribute surface: plain, multi-head, PosOffset (KV-cache decode), linear PI,
// YaRN, xPos query/key paths — serial and parallel sizes, both dtypes. The cpu
// kernel hoists the per-(row,pair) cos/sin out of the head loop; the values
// come from the SAME math.Cos/Sin/Pow calls ref makes, so parity is ~exact.
func TestRoPEKernelsMatchRefWithinUlps(t *testing.T) {
	cpuB := cpuBackend(t)
	refB, _ := backend.Get(backend.Ref)
	cases := []struct {
		name       string
		seq, width int
		attrs      backend.RoPEAttrs
	}{
		{"plain", 8, 16, backend.RoPEAttrs{}},
		{"heads", 16, 64, backend.RoPEAttrs{Heads: 4}},
		{"offset", 4, 32, backend.RoPEAttrs{Heads: 2, PosOffset: 100}},
		{"pi", 16, 64, backend.RoPEAttrs{Heads: 4, PosScale: 4}},
		{"yarn", 16, 64, backend.RoPEAttrs{Heads: 4, YaRNScale: 8, YaRNOrigCtx: 128}},
		{"xpos-q", 16, 64, backend.RoPEAttrs{Heads: 4, XPos: true}},
		{"xpos-k", 16, 64, backend.RoPEAttrs{Heads: 4, XPos: true, XPosDownscale: true, PosOffset: 7}},
		{"parallel", 256, 512, backend.RoPEAttrs{Heads: 8}},
	}
	ctx := backend.NewContext()
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, tc := range cases {
			q := tensor.Randn(dt, 11, tensor.Shape{tc.seq, tc.width})
			for _, side := range []struct {
				lbl string
				op  backend.Op
			}{{"fwd", backend.OpRoPE}, {"bwd", backend.OpRoPEBackward}} {
				got, err := backend.Execute(ctx.WithBackend(cpuB), side.op, []*tensor.Tensor{q}, tc.attrs)
				if err != nil {
					t.Fatalf("%s %s cpu: %v", tc.name, side.lbl, err)
				}
				want, err := backend.Execute(ctx.WithBackend(refB), side.op, []*tensor.Tensor{q}, tc.attrs)
				if err != nil {
					t.Fatalf("%s %s ref: %v", tc.name, side.lbl, err)
				}
				assertCloseUlps(t, got[0], want[0], fmt.Sprintf("rope-%s/%s/%s", side.lbl, tc.name, dt))
			}
		}
	}
}

// RoPE backward is the exact inverse rotation: fwd∘bwd must reproduce the
// input (up to roundoff). Catches pairing/rotate_half index errors and
// sign flips that a pure ref-parity test would miss if ref itself drifted.
func TestRoPEBackwardInvertsForward(t *testing.T) {
	cpuB := cpuBackend(t)
	ctx := backend.NewContext().WithBackend(cpuB)
	attrs := backend.RoPEAttrs{Heads: 4, PosOffset: 3}
	q := tensor.Randn(tensor.F64, 21, tensor.Shape{12, 32})
	fwd, err := backend.Execute(ctx, backend.OpRoPE, []*tensor.Tensor{q}, attrs)
	if err != nil {
		t.Fatal(err)
	}
	back, err := backend.Execute(ctx, backend.OpRoPEBackward, fwd, attrs)
	if err != nil {
		t.Fatal(err)
	}
	assertCloseUlps(t, back[0], q, "rope-roundtrip")
}
