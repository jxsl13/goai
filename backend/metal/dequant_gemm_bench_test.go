//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
)

// TestDequantGemmCrossover measures the prompt-processing alternative: dequantize the Q4_K weight
// ONCE into a dense f32 [K][N] buffer, then run a dense MPS GEMM over the batch — against the
// cooperative quant kernel, which re-derives every weight for every query row.
//
// This is the direction the matrix-unit attempt ruled IN. A matrix-unit kernel that dequantizes
// inside its inner loop spends ~240 scalar instructions per matrix instruction and cannot win;
// moving the dequantization out of the loop entirely is what changes the ratio.
//
// Measured on an M2 Pro, K=2048 N=5632:
//
//	M=  8  coop  246.6us  dequant+gemm 1122.7us  0.22x
//	M= 32  coop  978.7us  dequant+gemm 1127.8us  0.87x
//	M= 64  coop 1954.4us  dequant+gemm 1360.2us  1.44x
//	M=128  coop 3907.3us  dequant+gemm 1622.2us  2.41x
//	M=256  coop 7809.4us  dequant+gemm 2138.9us  3.65x
//
// The crossover sits near M=48: below it the fixed dequantization dominates, above it the dense
// GEMM's efficiency (3207 GFLOP/s at M=64, 4486 at M=256 — 47% and 66% of peak, against the quant
// kernel's 757 GFLOP/s / 11%) pays for it several times over.
//
// The dequantization is currently the weak half: 814us to write 44 MB is only 57 GB/s, because the
// transposing store Out[(k0+l)*N+n] strides by N and does not coalesce. A staged-tile transpose
// should approach the ~180 GB/s this machine sustains elsewhere, which would cut ~560us from every
// row of the table and move the crossover well below M=32.
//
// Reported, not asserted beyond a loose crossover check; absolute timings are machine-dependent.
func TestDequantGemmCrossover(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 5632
	nb := K / 256
	raw := make([]byte, N*nb*144)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	rw, err := Backend{}.UploadQuant(raw, 12, N, K)
	if err != nil {
		t.Skip(err)
	}
	rq := rw.(*ResidentQWeight)
	defer rq.Close()
	wf, err := NewDeviceBufferF32(make([]float32, K*N))
	if err != nil {
		t.Skip(err)
	}
	defer wf.Release()

	for _, M := range []int{64, 256} {
		x := make([]float32, M*K)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.21)) * 0.4
		}
		a, _ := NewDeviceBufferF32(x)
		c, _ := NewDeviceBufferF32(make([]float32, M*N))
		cq, _ := NewDeviceBufferF32(make([]float32, M*N))
		meas := func(fn func(r *Recorder)) float64 {
			best := 1e18
			for range 12 {
				r, err := NewRecorder()
				if err != nil {
					t.Fatal(err)
				}
				for range 4 {
					fn(r)
				}
				r.Commit()
				r.Wait()
				if d := LastGPUSeconds() / 4; d < best {
					best = d
				}
				r.Free()
			}
			return best
		}
		// QMatMulResident now ROUTES through dequant+GEMM above the wired threshold, so the
		// baseline arm must switch it off explicitly — otherwise this compares the path against
		// itself. It did exactly that once the wiring landed (ratio 1.00x, maxRel 0.00e+00), which
		// is what the crossover assertion below caught.
		SetQ4KDequantGemm(false)
		coop := meas(func(r *Recorder) { r.QMatMulResident(a, rq, cq, M) })
		SetQ4KDequantGemm(true)
		fused := meas(func(r *Recorder) {
			r.DequantQ4K(rq, wf)
			r.MatMul(a, wf, c, M, K, N)
		})

		// The two paths must agree. They differ only by f32 reassociation: the cooperative kernel
		// factors the min term out as d*sc*sum(x*q) - dmin*m*sum(x), while dequant-then-GEMM keeps
		// the per-element form x*(d*sc*q - dmin*m).
		SetQ4KDequantGemm(false) // the reference arm must be the quant kernel itself
		r, _ := NewRecorder()
		r.DequantQ4K(rq, wf)
		r.MatMul(a, wf, c, M, K, N)
		r.QMatMulResident(a, rq, cq, M)
		r.Commit()
		r.Wait()
		r.Free()
		SetQ4KDequantGemm(true)
		g := make([]float32, M*N)
		w := make([]float32, M*N)
		if err := c.DownloadF32(g); err != nil {
			t.Fatal(err)
		}
		if err := cq.DownloadF32(w); err != nil {
			t.Fatal(err)
		}
		var maxRel float64
		for i := range g {
			d := math.Abs(float64(g[i] - w[i]))
			den := math.Max(1, math.Abs(float64(w[i])))
			if rr := d / den; rr > maxRel {
				maxRel = rr
			}
		}
		fmt.Printf("dequant+GEMM M=%3d: coop %.1fus vs %.1fus = %.2fx (maxRel %.2e)\n",
			M, coop*1e6, fused*1e6, coop/fused, maxRel)
		if maxRel > 1e-3 {
			t.Errorf("M=%d: dequant+GEMM disagrees with the quant kernel by %.2e", M, maxRel)
		}
		if M >= 128 && fused >= coop {
			t.Errorf("M=%d: dequant+GEMM (%.1fus) should beat the quant kernel (%.1fus) at this batch size", M, fused*1e6, coop*1e6)
		}
		a.Release()
		c.Release()
		cq.Release()
	}
}
