//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/metal"
)

// BenchmarkMetalNonMatMulIncremental measures the floor-free per-op cost of the
// non-matmul decode ops at TinyLlama-1.1B's shapes. Apportionment showed those ops
// account for 25.4 ms of a 40.28 ms token — 63 percent, and 4.4x llama.cpp's ENTIRE
// token — so this is where the remaining gap mostly lives.
//
// Same method as the quant version: time a submit carrying 1 op against one carrying
// 16 and take (t16-t1)/15, so the ~149us per-submit floor cancels.
func BenchmarkMetalNonMatMulIncremental(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const dim, heads, kvHeads, hd = 2048, 32, 4, 64
	const seqK = 64 // KV length during decode of a short sequence

	mk := func(n int) *metal.DeviceBuffer {
		buf, err := metal.NewDeviceBufferF32(make([]float32, n))
		if err != nil {
			b.Fatal(err)
		}
		return buf
	}
	x := mk(dim)
	defer x.Release()
	g := mk(dim)
	defer g.Release()
	o := mk(dim)
	defer o.Release()
	inv := mk(hd)
	defer inv.Release()
	kbuf := mk(seqK * kvHeads * hd)
	defer kbuf.Release()
	vbuf := mk(seqK * kvHeads * hd)
	defer vbuf.Release()
	qbuf := mk(heads * hd)
	defer qbuf.Release()
	obuf := mk(heads * hd)
	defer obuf.Release()

	each := func(name string, rec func(r *metal.Recorder) error) {
		b.Run(name, func(b *testing.B) {
			run := func(ops int) {
				r, err := metal.NewRecorder()
				if err != nil {
					b.Fatal(err)
				}
				for range ops {
					if err := rec(r); err != nil {
						r.Free()
						b.Fatal(err)
					}
				}
				if err := r.Finish(); err != nil {
					r.Free()
					b.Fatal(err)
				}
				r.Free()
			}
			for range 10 {
				run(1)
			}
			b.Run("1op", func(b *testing.B) {
				for range b.N {
					run(1)
				}
			})
			b.Run("16ops", func(b *testing.B) {
				for range b.N {
					run(16)
				}
			})
		})
	}

	each("RMSNorm", func(r *metal.Recorder) error { return r.RMSNorm(x, g, o, 1, dim, 1e-5) })
	each("BinaryAdd", func(r *metal.Recorder) error { return r.Binary(x, g, o, 0) })
	each("RoPE", func(r *metal.Recorder) error {
		return r.RoPE(qbuf, inv, obuf, 1, heads*hd, heads, hd, hd/2, 0, 1)
	})
	each("MHA", func(r *metal.Recorder) error {
		return r.MHA(qbuf, kbuf, vbuf, obuf, 1, seqK, heads*hd, heads, kvHeads, hd, 0, 0, 0.125)
	})
	each("Blit", func(r *metal.Recorder) error { return r.Blit(x, 0, o, 0, dim) })
}
