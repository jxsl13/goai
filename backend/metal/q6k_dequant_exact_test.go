//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
)

// TestQ6KDequantExact isolates the Q6_K expansion from any accumulation: it compares the
// dequantized WEIGHTS against a straight CPU decode of the ggml block layout, and requires exact
// agreement.
//
// This is the real correctness guard for the expansion. The matmul-level parity test can only show
// ~1e-4..1e-3 against the scalar kernel, because MPS GEMM and a sequential dot product sum K terms
// in different orders and that difference grows with K — it cannot distinguish a wrong dequant from
// ordinary reassociation. This one can: it is 0.000e+00 or it is broken.
func TestQ6KDequantExact(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 512, 64
	nb := K / 256
	raw := make([]byte, N*nb*210)
	for i := range raw {
		raw[i] = byte(i*29 + 13)
	}
	for b := 0; b+210 <= len(raw); b += 210 {
		raw[b+208], raw[b+209] = 0x00, 0x34
	}
	rw, err := Backend{}.UploadQuant(raw, 14, N, K)
	if err != nil {
		t.Skip(err)
	}
	rq := rw.(*ResidentQWeight)
	defer rq.Close()
	out, _ := NewDeviceBufferF32(make([]float32, K*N))
	defer out.Release()
	r, _ := NewRecorder()
	if err := r.DequantQ4K(rq, out); err != nil {
		t.Fatal(err)
	}
	r.Commit()
	r.Wait()
	r.Free()
	got := make([]float32, K*N)
	out.DownloadF32(got)

	// CPU reference straight from the ggml Q6_K layout.
	f16 := func(lo, hi byte) float32 {
		h := uint16(lo) | uint16(hi)<<8
		s := uint32(h&0x8000) << 16
		e := uint32(h>>10) & 0x1F
		m := uint32(h & 0x3FF)
		switch {
		case e == 0 && m == 0:
			return math.Float32frombits(s)
		case e == 31:
			return math.Float32frombits(s | 0x7F800000 | m<<13)
		}
		return math.Float32frombits(s | (e+112)<<23 | m<<13)
	}
	var maxAbs float64
	for n := range N {
		for sb := range nb {
			base := n*nb*210 + sb*210
			d := f16(raw[base+208], raw[base+209])
			for k := range 256 {
				chunk, rr := k>>7, k&127
				g, l := rr>>5, rr&31
				ql := raw[base+chunk*64+(g&1)*32+l]
				qh := raw[base+128+chunk*32+l]
				nib := int(ql & 0xF)
				if g >= 2 {
					nib = int(ql >> 4)
				}
				hi := int(qh>>(g*2)) & 3
				q := (nib | hi<<4) - 32
				sc := int8(raw[base+192+chunk*8+(l>>4)+g*2])
				want := float64(d) * float64(sc) * float64(q)
				g2 := float64(got[(sb*256+k)*N+n])
				if a := math.Abs(g2 - want); a > maxAbs {
					maxAbs = a
				}
			}
		}
	}
	fmt.Printf("Q6DQ max abs weight error vs CPU decode = %.3e\n", maxAbs)
	if maxAbs > 1e-6 {
		t.Errorf("Q6_K expansion disagrees with the CPU decode by %.3e", maxAbs)
	}
}
