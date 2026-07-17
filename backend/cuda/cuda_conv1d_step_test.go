//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAConv1DStep checks cu_conv1d_step (Recorder.Conv1DStep) — one decode step of Mamba's causal
// depthwise conv — against the reference full-sequence OpConv1D: applying the step kernel L times
// (conv state carried between calls, starting zeroed for the causal left-pad) must reproduce the
// reference conv's per-token outputs. The decode-time conv primitive for a GPU Mamba block.
func TestCUDAConv1DStep(t *testing.T) {
	skipNoGPU(t)
	const L, D, K = 7, 5, 4

	gen := func(n, seed int, scale float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32((((i+seed)*7)%23 - 11)) * float32(scale)
		}
		return x
	}
	xD := gen(L*D, 1, 0.05)
	wD := gen(D*K, 2, 0.08)
	bD := gen(D, 3, 0.1)

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
	refT, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpConv1D,
		[]*tensor.Tensor{mk2(xD, L, D), mk2(wD, D, K), mk1(bD)}, nil)
	if err != nil {
		t.Fatalf("ref OpConv1D: %v", err)
	}

	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	dw, _ := cuda.NewDeviceBufferF32(wD)
	db, _ := cuda.NewDeviceBufferF32(bD)
	dstate, _ := cuda.NewDeviceBufferF32(make([]float32, D*(K-1))) // state, explicitly zeroed
	dx, _ := cuda.NewDeviceF32(1, D)
	dout, _ := cuda.NewDeviceF32(1, D)
	defer func() {
		dw.Free()
		db.Free()
		dstate.Free()
		dx.Free()
		dout.Free()
	}()

	for tstep := 0; tstep < L; tstep++ {
		dx.UploadF32(xD[tstep*D : (tstep+1)*D])
		if err := rec.Conv1DStep(dx, dw, db, dstate, dout, D, K); err != nil {
			t.Fatalf("step %d Conv1DStep: %v", tstep, err)
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
			if math.IsNaN(float64(out[c])) || math.Abs(float64(out[c])-want) > 1e-4 {
				t.Fatalf("step %d chan %d: gpu %v vs ref %v", tstep, c, out[c], want)
			}
		}
	}
	t.Logf("cu_conv1d_step reproduces the reference OpConv1D full-sequence conv across %d timesteps (Mamba decode conv)", L)
}
