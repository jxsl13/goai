//go:build darwin && cgo

package metal

import (
	"fmt"
	"os"
	"testing"
)

// TestQKVFusionValue prices fusing TinyLlama's 10 mixed-quant-type layers, and corrects an estimate
// that was nearly 4x too high.
//
// TestQKVFusionCoverage established that 10 of 22 blocks fall back to three separate q/k/v matmuls
// because attn_v is Q6_K where q and k are Q4_K. TestPrefillLeaveOneOut compared an ALL-unfused
// chain against the real decoder and put the prize at ~8.9%. That comparison was invalid: the real
// decoder already fuses 12 of the 22 layers, so the chain was not modelling it.
//
// Measuring the projection stage directly, with 12 blocks fused and the other 10 in each candidate
// shape:
//
//	M= 64   current 6.311 ms   q|k for the 10  5.372 ms (-14.9%)   all 22 fused 4.861 ms (-23.0%)
//	M=512   current 27.066 ms  q|k for the 10 26.040 ms (- 3.8%)   all 22 fused 24.648 ms (- 8.9%)
//
// Those are large percentages OF THE PROJECTION STAGE, which is 16% of a 39.34 ms prefill at M=64
// and 11% at M=512. End to end:
//
//	q|k fusion for the mixed layers   ~2.4% at pp64,  ~0.4% at pp512
//	full fusion (needs requantising v) ~3.7% at pp64,  ~1.0% at pp512
//
// So the change is worth ~2.4% at the shape where it helps most, not ~8.9%. Against that: q|k
// fusion needs a third QKV code path, because the fused layout is one q||k||v buffer that RoPE and
// attention index through an element offset, and a partial fusion changes those offsets in both the
// decode and prefill record paths. This branch has already fixed one row-misattribution bug of
// exactly that kind. NOT worth it at 2.4%, and the measurement cost one test rather than a day of
// plumbing plus the bug.
//
// Full fusion would require requantising attn_v from Q6_K to Q4_K, which changes model numerics —
// out of scope for a performance change.
func TestQKVFusionValue(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const (
		layers = 22
		dim    = 2048
		hidden = 5632
		kvh    = 4
		dk     = 64
	)
	kvDim := kvh * dk
	M := 64
	if v := os.Getenv("GOAI_FUSE_M"); v != "" {
		fmt.Sscanf(v, "%d", &M)
	}
	up := func(k, n int) *ResidentQWeight {
		raw := make([]byte, n*(k/256)*144)
		for i := range raw {
			raw[i] = byte(i*31 + 7)
		}
		rw, err := Backend{}.UploadQuant(raw, 12, n, k)
		if err != nil {
			t.Skip(err)
		}
		return rw.(*ResidentQWeight)
	}
	wqkv := up(dim, dim+2*kvDim) // all three fused
	wqk := up(dim, dim+kvDim)    // q|k fused, v separate
	wq, wk, wv := up(dim, dim), up(dim, kvDim), up(dim, kvDim)
	defer func() {
		for _, w := range []*ResidentQWeight{wqkv, wqk, wq, wk, wv} {
			w.Close()
		}
	}()
	nb := func(n int) *DeviceBuffer { b, _ := NewDeviceBufferF32(make([]float32, n)); return b }
	xn := nb(M * dim)
	o1, o2, o3 := nb(M*(dim+2*kvDim)), nb(M*kvDim), nb(M*kvDim)
	defer func() {
		for _, b := range []*DeviceBuffer{xn, o1, o2, o3} {
			b.Release()
		}
	}()
	// nFused of `layers` blocks use full fusion; the rest use `mode`
	run := func(nFused int, mode string) float64 {
		best := 1e18
		for range 7 {
			r, _ := NewRecorder()
			for i := 0; i < layers; i++ {
				if i < nFused {
					_ = r.QMatMulResident(xn, wqkv, o1, M)
					continue
				}
				switch mode {
				case "unfused":
					_ = r.QMatMulResident(xn, wq, o1, M)
					_ = r.QMatMulResident(xn, wk, o2, M)
					_ = r.QMatMulResident(xn, wv, o3, M)
				case "qk":
					_ = r.QMatMulResident(xn, wqk, o1, M)
					_ = r.QMatMulResident(xn, wv, o3, M)
				}
			}
			r.Commit()
			r.Wait()
			if d := LastGPUSeconds(); d < best {
				best = d
			}
			r.Free()
		}
		return best * 1e3
	}
	cur := run(12, "unfused")
	qk := run(12, "qk")
	all := run(22, "")
	fmt.Printf("FUSE M=%d  current(12 fused,10 unfused)=%.3f ms  q|k for the 10=%.3f ms (%.1f%%)  all 22 fused=%.3f ms (%.1f%%)\n",
		M, cur, qk, 100*(cur-qk)/cur, all, 100*(cur-all)/cur)
}
