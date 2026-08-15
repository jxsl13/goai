//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
)

// The two-pass mha_f32 path — non-causal sq>1 (encoder attention) and windowed attention — had NO
// correctness coverage at dk=64. That was established by mutation, not assumed: replacing the whole
// output of the dk==64 specialization with zeros left `go test -run 'MHA|Attn|Bert'` GREEN, while a
// stderr probe showed the specialized pipeline built and taking 1620 of 1620 dispatches. The kernel
// was live in production and unguarded.
//
// This closes that gap by comparing against attention computed directly in float64 here, for the
// shapes that actually reach this kernel: causal==0 (mtl_recorder_mha routes causal!=0 to the
// decode kernel) and window>0.
func TestMHATwoPassMatchesReference(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	for _, c := range []struct {
		name                   string
		sq, heads, kvHeads, dk int
		causal, window         int
	}{
		// dk=64 hits the specialization; dk=32 and dk=48 must keep the generic kernel, which is
		// what the exact-equality gate is for — an earlier cut used dk<=64 with a hardcoded 64
		// trip count and silently read past the real head width.
		{"noncausal-dk64", 12, 2, 2, 64, 0, 0},
		{"noncausal-dk64-gqa", 9, 4, 2, 64, 0, 0},
		{"window-dk64", 16, 2, 2, 64, 1, 5},
		{"noncausal-dk32", 12, 2, 2, 32, 0, 0},
		{"noncausal-dk48", 7, 3, 3, 48, 0, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			dm := c.heads * c.dk
			dkv := c.kvHeads * c.dk
			scale := float32(1 / math.Sqrt(float64(c.dk)))
			qh := make([]float32, c.sq*dm)
			kh := make([]float32, c.sq*dkv)
			vh := make([]float32, c.sq*dkv)
			for i := range qh {
				qh[i] = float32(math.Sin(float64(i)*0.31)) * 0.5
			}
			for i := range kh {
				kh[i] = float32(math.Cos(float64(i)*0.17)) * 0.5
				vh[i] = float32(math.Sin(float64(i)*0.11)) * 0.5
			}
			q, _ := NewDeviceBufferF32(qh)
			k, _ := NewDeviceBufferF32(kh)
			v, _ := NewDeviceBufferF32(vh)
			o, _ := NewDeviceBufferF32(make([]float32, c.sq*dm))
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			if err := r.MHA(q, k, v, o, c.sq, c.sq, dm, c.heads, c.kvHeads, c.dk, c.causal, c.window, scale); err != nil {
				t.Fatal(err)
			}
			if err := r.Commit(); err != nil {
				t.Fatal(err)
			}
			if err := r.Wait(); err != nil {
				t.Fatal(err)
			}
			got := make([]float32, c.sq*dm)
			if err := o.DownloadF32(got); err != nil {
				t.Fatal(err)
			}
			r.Free()

			rep := c.heads / c.kvHeads
			for h := range c.heads {
				for i := range c.sq {
					jmax := c.sq
					if c.causal != 0 {
						jmax = i + 1
					}
					jmin := 0
					if c.window > 0 && i-c.window+1 > 0 {
						jmin = i - c.window + 1
					}
					m := math.Inf(-1)
					sc := make([]float64, c.sq)
					for j := jmin; j < jmax; j++ {
						var s float64
						for d := range c.dk {
							s += float64(qh[i*dm+h*c.dk+d]) * float64(kh[j*dkv+(h/rep)*c.dk+d])
						}
						s *= float64(scale)
						sc[j] = s
						if s > m {
							m = s
						}
					}
					var sum float64
					acc := make([]float64, c.dk)
					for j := jmin; j < jmax; j++ {
						e := math.Exp(sc[j] - m)
						sum += e
						for d := range c.dk {
							acc[d] += e * float64(vh[j*dkv+(h/rep)*c.dk+d])
						}
					}
					for d := range c.dk {
						want := acc[d] / sum
						g := float64(got[i*dm+h*c.dk+d])
						if math.Abs(g-want) > 2e-5*math.Max(1, math.Abs(want)) {
							t.Fatalf("h=%d i=%d d=%d: got %v want %v", h, i, d, g, want)
						}
					}
				}
			}
		})
	}
}
