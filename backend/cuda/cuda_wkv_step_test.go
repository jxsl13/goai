//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAWKVStep checks cu_wkv_step (Recorder.WKVStep) — one decode step of the RWKV-4 WKV
// recurrence — against the reference full-sequence OpWKV. Applying the step kernel L times (per-
// channel numerator/denominator/max-exponent state aa/bb/pp carried between calls, starting
// aa=bb=0 / pp=−1e38) must reproduce, for every channel and timestep, the reference's stabilized
// linear-attention output. The decode-time WKV primitive for a GPU RWKV block (no KV cache — RWKV's
// O(1) recurrent inference).
func TestCUDAWKVStep(t *testing.T) {
	skipNoGPU(t)
	const L, D = 8, 6 // seq, channels

	gen := func(n, seed int, scale float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32((((i+seed)*11)%23 - 11)) * float32(scale)
		}
		return x
	}
	kD := gen(L*D, 1, 0.2)
	vD := gen(L*D, 2, 0.3)
	// decay w must be positive (= exp(WLog)); u is the current-token bonus.
	wRaw := gen(D, 3, 0.15)
	wD := make([]float32, D)
	for c := range wD {
		wD[c] = float32(math.Exp(float64(wRaw[c])))
	}
	uD := gen(D, 4, 0.25)

	mk2 := func(data []float32, r, c int) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{r, c})
		copy(x.Storage().F32(), data)
		return x
	}
	mk1 := func(data []float32) *tensor.Tensor {
		x := tensor.New(tensor.F32, tensor.Shape{len(data)})
		copy(x.Storage().F32(), data)
		return x
	}
	refT, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpWKV,
		[]*tensor.Tensor{mk2(kD, L, D), mk2(vD, L, D), mk1(wD), mk1(uD)}, nil)
	if err != nil {
		t.Fatalf("ref OpWKV: %v", err)
	}

	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	dw, _ := cuda.NewDeviceBufferF32(wD)
	du, _ := cuda.NewDeviceBufferF32(uD)
	// running state: aa=0, bb=0, pp=−1e38 (the pre-first-token WKV state).
	ppInit := make([]float32, D)
	for c := range ppInit {
		ppInit[c] = -1e38
	}
	daa, _ := cuda.NewDeviceBufferF32(make([]float32, D))
	dbb, _ := cuda.NewDeviceBufferF32(make([]float32, D))
	dpp, _ := cuda.NewDeviceBufferF32(ppInit)
	dk, _ := cuda.NewDeviceF32(1, D)
	dv, _ := cuda.NewDeviceF32(1, D)
	dout, _ := cuda.NewDeviceF32(1, D)
	defer func() {
		dw.Free()
		du.Free()
		daa.Free()
		dbb.Free()
		dpp.Free()
		dk.Free()
		dv.Free()
		dout.Free()
	}()

	for tstep := 0; tstep < L; tstep++ {
		dk.UploadF32(kD[tstep*D : (tstep+1)*D])
		dv.UploadF32(vD[tstep*D : (tstep+1)*D])
		if err := rec.WKVStep(dk, dv, dw, du, daa, dbb, dpp, dout, D); err != nil {
			t.Fatalf("step %d WKVStep: %v", tstep, err)
		}
		if err := rec.Wait(); err != nil {
			t.Fatalf("step %d wait: %v", tstep, err)
		}
		out := make([]float32, D)
		if err := dout.DownloadF32(out); err != nil {
			t.Fatalf("step %d download: %v", tstep, err)
		}
		for c := 0; c < D; c++ {
			want := refT[0].AtF64(tstep, c)
			if math.IsNaN(float64(out[c])) || math.Abs(float64(out[c])-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Fatalf("step %d chan %d: gpu %v vs ref %v", tstep, c, out[c], want)
			}
		}
	}
	t.Logf("cu_wkv_step reproduces the reference OpWKV full-sequence recurrence across %d timesteps, %d channels (RWKV decode)", L, D)
}
