package nlp

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// keepSinkRecentNaive is the per-element form the typed copy replaced, kept here
// as the reference the fast path is pinned against.
func keepSinkRecentNaive(t *tensor.Tensor, sinks, window int) *tensor.Tensor {
	rows, d := t.Shape()[0], t.Shape()[1]
	if rows <= sinks+window {
		return t
	}
	out := tensor.New(t.Dtype(), tensor.Shape{sinks + window, d})
	for i := range sinks {
		for j := range d {
			out.SetF64(t.AtF64(i, j), i, j)
		}
	}
	for i := range window {
		for j := range d {
			out.SetF64(t.AtF64(rows-window+i, j), sinks+i, j)
		}
	}
	return out
}

// TestKeepSinkRecentMatchesPerElement pins the typed row copy bit-for-bit against
// the per-element traversal. The op does no arithmetic — it is a same-dtype copy —
// so exact equality is the bar, not a tolerance.
func TestKeepSinkRecentMatchesPerElement(t *testing.T) {
	cases := []struct{ rows, d, sinks, window int }{
		{64, 16, 4, 8}, // past the bound: both blocks copied
		{12, 5, 4, 8},  // exactly at the bound: no-op
		{9, 3, 4, 8},   // below the bound: no-op
		{100, 1, 1, 1}, // degenerate width and blocks
		{33, 7, 0, 8},  // no sinks
		{33, 7, 4, 0},  // no window
	}
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		for _, c := range cases {
			t.Run(fmt.Sprintf("%v/r%d_d%d_s%d_w%d", dt, c.rows, c.d, c.sinks, c.window), func(t *testing.T) {
				x := tensor.New(dt, tensor.Shape{c.rows, c.d})
				for i := range x.Numel() {
					x.SetF64(float64(i)*0.25+1, tensor.Unravel(i, x.Shape())...)
				}
				want := keepSinkRecentNaive(x, c.sinks, c.window)
				got := keepSinkRecent(x, c.sinks, c.window)
				if !got.Shape().Equal(want.Shape()) {
					t.Fatalf("shape = %v, want %v", got.Shape(), want.Shape())
				}
				for i := range want.Numel() {
					idx := tensor.Unravel(i, want.Shape())
					if g, w := got.AtF64(idx...), want.AtF64(idx...); g != w {
						t.Fatalf("element %v = %v, want %v (not bit-identical)", idx, g, w)
					}
				}
			})
		}
	}
}

// BenchmarkKeepSinkRecent measures the bounded-cache eviction at a realistic
// StreamingLLM geometry — past the bound, so the early-return never fires.
func BenchmarkKeepSinkRecent(b *testing.B) {
	const rows, d, sinks, window = 2048, 2048, 4, 512
	x := tensor.New(tensor.F32, tensor.Shape{rows, d})
	b.ReportAllocs()
	b.SetBytes(int64((sinks + window) * d * 4))
	for b.Loop() {
		_ = keepSinkRecent(x, sinks, window)
	}
}
