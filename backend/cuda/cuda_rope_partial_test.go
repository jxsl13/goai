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

// RoPEPartialBand rotates only the first rotaryDim channels of `heads` heads at a column
// offset inside a wider [seq,stride] buffer (the fused-QKV band), leaving everything else —
// the head tails AND the rest of the row — untouched. Checked vs a direct host oracle.
func TestCUDARoPEPartialBand(t *testing.T) {
	skipNoGPU(t)
	for _, c := range []struct {
		seq, stride, off, heads, hd, rotaryDim int
		attrs                                  backend.RoPEAttrs
	}{
		{4, 96, 0, 2, 32, 16, backend.RoPEAttrs{Heads: 2}},                // band at column 0, stride 96 > band 64
		{2, 160, 64, 4, 16, 8, backend.RoPEAttrs{Heads: 4, PosOffset: 5}}, // band at offset 64 (q/k band of a fused QKV)
	} {
		x := bench.RandF32(tensor.Shape{c.seq, c.stride}, 21)
		xf := append([]float32(nil), x.Storage().F32()...) // original, before the in-place rotation

		dx, _ := cuda.UploadF32(x)
		if err := dx.RoPEPartialBand(c.attrs, c.rotaryDim, c.off, c.heads, c.hd); err != nil {
			t.Fatalf("RoPEPartialBand: %v", err)
		}
		got, _ := dx.ToHost()
		dx.Free()

		// Host oracle: rotate_half the rotaryDim prefix of each head in the band; everything else = original.
		inv, posDiv := backend.RoPEFreqs(c.rotaryDim, c.attrs)
		half := c.rotaryDim / 2
		want := append([]float32(nil), xf...)
		for s := 0; s < c.seq; s++ {
			pos := float64(c.attrs.PosOffset+s) / posDiv
			for h := 0; h < c.heads; h++ {
				base := s*c.stride + c.off + h*c.hd
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
				t.Fatalf("band rope @%d (off=%d rotaryDim=%d): device %v host %v", i, c.off, c.rotaryDim, g, w)
			}
		}
	}
}

// Recorder.RoPEPartialPair rotates the rotaryDim prefix of the q band and the k band of a
// fused [seq,stride] QKV buffer in place — the fused-QKV partial-rope a graph-captured
// GPT-NeoX/Phi/StableLM decoder records. The v band and all head tails stay untouched.
func TestCUDARecRoPEPartialPair(t *testing.T) {
	skipNoGPU(t)
	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	const (
		seq, hd, rotaryDim = 2, 16, 8
		headsQ, headsK     = 4, 2
		offQ               = 0
		offK               = headsQ * hd // 64
		offV               = offK + headsK*hd
		stride             = offV + headsK*hd // + a v band → 128
		pos                = 9
		base               = 10000.0
		posDiv             = 1.0
	)
	x := bench.RandF32(tensor.Shape{seq, stride}, 31)
	xf := append([]float32(nil), x.Storage().F32()...)

	inv, err := cuda.BuildRoPEInv(rotaryDim, base)
	if err != nil {
		t.Fatal(err)
	}
	defer inv.Free()
	d, err := cuda.UploadF32(x)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Free()
	if err := rec.RoPEPartialPair(d, inv, seq, stride, headsQ, offQ, headsK, offK, hd, rotaryDim, pos, posDiv); err != nil {
		t.Fatalf("RoPEPartialPair: %v", err)
	}
	if err := cuda.GraphSync(); err != nil {
		t.Fatalf("GraphSync: %v", err)
	}
	got, err := d.ToHost()
	if err != nil {
		t.Fatal(err)
	}

	// Host oracle: rotate_half the rotaryDim prefix of each head in the q and k bands only.
	half := rotaryDim / 2
	freq := make([]float64, half)
	for i := range freq {
		freq[i] = 1.0 / math.Pow(base, float64(2*i)/float64(rotaryDim))
	}
	want := append([]float32(nil), xf...)
	rotate := func(off, heads int) {
		for s := 0; s < seq; s++ {
			p := float64(pos+s) / posDiv
			for h := 0; h < heads; h++ {
				b := s*stride + off + h*hd
				for i := 0; i < half; i++ {
					ang := p * freq[i]
					cs, sn := math.Cos(ang), math.Sin(ang)
					qi, qih := float64(xf[b+i]), float64(xf[b+i+half])
					want[b+i] = float32(qi*cs - qih*sn)
					want[b+i+half] = float32(qih*cs + qi*sn)
				}
			}
		}
	}
	rotate(offQ, headsQ)
	rotate(offK, headsK)
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), float64(want[i])
		if math.Abs(g-w) > 1e-3*math.Max(1, math.Abs(w)) {
			t.Fatalf("rec RoPEPartialPair @%d: device %v host %v", i, g, w)
		}
	}
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
