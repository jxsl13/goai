//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// cu_rope_partial rotates only the first rotaryDim channels of each head (rotate_half),
// leaving the tail unchanged — the GPT-NeoX/Phi/StableLM "partial rotary". Two checks:
// (1) rotaryDim==hd must be BIT-IDENTICAL to the trusted full RoPE kernel; (2) rotaryDim<hd
// must match a direct host partial-rope (same backend.RoPEFreqs + rotate_half + passthrough).
func TestCUDARoPEPartial(t *testing.T) {
	skipNoGPU(t)

	// (1) Degenerate: partial with rotaryDim==hd == full RoPE, bit-exact.
	t.Run("degenerate_eq_full", func(t *testing.T) {
		for _, c := range []struct {
			seq, width int
			attrs      backend.RoPEAttrs
		}{
			{8, 64, backend.RoPEAttrs{Heads: 4}},
			{6, 32, backend.RoPEAttrs{Heads: 2, PosOffset: 100}},
			{8, 128, backend.RoPEAttrs{Heads: 8, PosScale: 2}},
		} {
			x := bench.RandF32(tensor.Shape{c.seq, c.width}, 5)
			heads := c.attrs.Heads
			hd := c.width / heads

			dp, _ := cuda.UploadF32(x)
			if err := dp.RoPEPartial(c.attrs, hd); err != nil {
				t.Fatalf("RoPEPartial: %v", err)
			}
			gotP, _ := dp.ToHost()
			dp.Free()

			df, _ := cuda.UploadF32(x)
			if err := df.RoPE(c.attrs); err != nil {
				t.Fatalf("RoPE: %v", err)
			}
			gotF, _ := df.ToHost()
			df.Free()

			for i := range gotP.Numel() {
				idx := tensor.Unravel(i, gotP.Shape())
				if p, f := gotP.AtF64(idx...), gotF.AtF64(idx...); p != f {
					t.Fatalf("rotaryDim==hd not bit-exact vs full RoPE @%d: partial %v full %v", i, p, f)
				}
			}
		}
	})

	// (2) Partial: rotaryDim < hd vs a direct host reference (rotated prefix + passthrough tail).
	t.Run("partial_vs_host", func(t *testing.T) {
		for _, c := range []struct {
			seq, heads, hd, rotaryDim int
			attrs                     backend.RoPEAttrs
		}{
			{8, 4, 64, 32, backend.RoPEAttrs{Heads: 4}},               // partial_rotary_factor 0.5 (Phi-style)
			{6, 2, 32, 24, backend.RoPEAttrs{Heads: 2, PosOffset: 7}}, // 0.75, decode offset
			{4, 8, 128, 16, backend.RoPEAttrs{Heads: 8, PosScale: 2}}, // 0.125 + PI scale
		} {
			width := c.heads * c.hd
			x := bench.RandF32(tensor.Shape{c.seq, width}, 9)
			xf := x.Storage().F32()

			dp, _ := cuda.UploadF32(x)
			if err := dp.RoPEPartial(c.attrs, c.rotaryDim); err != nil {
				t.Fatalf("RoPEPartial: %v", err)
			}
			got, _ := dp.ToHost()
			dp.Free()

			// Host reference: same RoPEFreqs over rotaryDim, rotate_half the prefix, pass the tail.
			inv, posDiv := backend.RoPEFreqs(c.rotaryDim, c.attrs)
			half := c.rotaryDim / 2
			want := append([]float32(nil), xf...)
			for s := 0; s < c.seq; s++ {
				pos := float64(c.attrs.PosOffset+s) / posDiv
				for h := 0; h < c.heads; h++ {
					base := s*width + h*c.hd
					for i := 0; i < half; i++ {
						ang := pos * inv[i]
						cs, sn := math.Cos(ang), math.Sin(ang)
						qi, qih := float64(xf[base+i]), float64(xf[base+i+half])
						want[base+i] = float32(qi*cs - qih*sn)
						want[base+i+half] = float32(qih*cs + qi*sn)
					}
				}
			}
			for i := range got.Numel() {
				idx := tensor.Unravel(i, got.Shape())
				g, w := got.AtF64(idx...), float64(want[i])
				if math.Abs(g-w) > 1e-3*math.Max(1, math.Abs(w)) {
					t.Fatalf("partial rope @%d (rotaryDim=%d hd=%d): device %v host %v", i, c.rotaryDim, c.hd, g, w)
				}
			}
		}
	})
}

// The device-position partial rope (graph-capturable) must be BIT-IDENTICAL to the
// host-posOffset partial rope at a matched position — same kernel, position from a
// device int vs a launch param.
func TestCUDARoPEPartialDpos(t *testing.T) {
	skipNoGPU(t)
	pos, err := cuda.NewDevicePos()
	if err != nil {
		t.Fatal(err)
	}
	defer pos.Free()
	for _, c := range []struct {
		seq, heads, hd, rotaryDim, at int
		attrs                         backend.RoPEAttrs
	}{
		{1, 4, 64, 32, 137, backend.RoPEAttrs{Heads: 4}},             // decode step at pos 137
		{1, 8, 128, 16, 0, backend.RoPEAttrs{Heads: 8}},              // pos 0
		{4, 2, 32, 24, 40, backend.RoPEAttrs{Heads: 2, PosScale: 2}}, // multi-row + PI
	} {
		width := c.heads * c.hd
		x := bench.RandF32(tensor.Shape{c.seq, width}, 11)

		dh, _ := cuda.UploadF32(x)
		attrsH := c.attrs
		attrsH.PosOffset = c.at
		if err := dh.RoPEPartial(attrsH, c.rotaryDim); err != nil {
			t.Fatalf("RoPEPartial: %v", err)
		}
		wantT, _ := dh.ToHost()
		dh.Free()

		dd, _ := cuda.UploadF32(x)
		if err := pos.Set(c.at); err != nil {
			t.Fatalf("pos.Set: %v", err)
		}
		if err := dd.RoPEPartialDpos(c.attrs, c.rotaryDim, pos); err != nil {
			t.Fatalf("RoPEPartialDpos: %v", err)
		}
		gotT, _ := dd.ToHost()
		dd.Free()

		for i := range gotT.Numel() {
			idx := tensor.Unravel(i, gotT.Shape())
			if g, w := gotT.AtF64(idx...), wantT.AtF64(idx...); g != w {
				t.Fatalf("dpos != host-offset @%d (rotaryDim=%d at=%d): dpos %v host %v", i, c.rotaryDim, c.at, g, w)
			}
		}
	}
}
