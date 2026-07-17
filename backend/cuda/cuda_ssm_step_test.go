//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDASSMStep checks cu_ssm_step (Recorder.SSMStep) — one timestep of the Mamba selective-scan
// recurrence — against the reference full-sequence OpSSM: applying the step kernel L times (state
// carried between calls) must reproduce the reference scan's per-token outputs. This is the
// DECODE-time SSM primitive (nlp.Mamba only has a full-sequence Forward), validated end to end.
func TestCUDASSMStep(t *testing.T) {
	skipNoGPU(t)
	const L, D, N = 6, 5, 4

	gen := func(nElem, seed int, scale float64) []float32 {
		x := make([]float32, nElem)
		for i := range x {
			x[i] = float32((((i+seed)*7)%23 - 11)) * float32(scale)
		}
		return x
	}
	// delta > 0 (softplus-like), A < 0 (stability), plus u, B, C, D_skip.
	uD := gen(L*D, 1, 0.05)
	deltaD := make([]float32, L*D)
	for i := range deltaD {
		deltaD[i] = float32(0.1 + math.Abs(float64(gen(1, i, 0.03)[0])))
	}
	aD := make([]float32, D*N)
	for i := range aD {
		aD[i] = -float32(0.2 + math.Abs(float64(gen(1, i+3, 0.05)[0])))
	}
	bD := gen(L*N, 2, 0.06)
	cD := gen(L*N, 4, 0.06)
	dskip := gen(D, 5, 0.1)

	// Reference: full selective scan via backend OpSSM (f64).
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
	refT, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpSSM,
		[]*tensor.Tensor{mk2(uD, L, D), mk2(deltaD, L, D), mk2(aD, D, N), mk2(bD, L, N), mk2(cD, L, N), mk1(dskip)}, nil)
	if err != nil {
		t.Fatalf("ref OpSSM: %v", err)
	}

	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	da, _ := cuda.NewDeviceBufferF32(aD)
	ddk, _ := cuda.NewDeviceBufferF32(dskip)
	dh, _ := cuda.NewDeviceF32(D, N) // state, zeroed
	du, _ := cuda.NewDeviceF32(1, D)
	ddelta, _ := cuda.NewDeviceF32(1, D)
	db, _ := cuda.NewDeviceF32(1, N)
	dc, _ := cuda.NewDeviceF32(1, N)
	dy, _ := cuda.NewDeviceF32(1, D)
	defer func() {
		da.Free()
		ddk.Free()
		dh.Free()
		du.Free()
		ddelta.Free()
		db.Free()
		dc.Free()
		dy.Free()
	}()

	for tstep := 0; tstep < L; tstep++ {
		du.UploadF32(uD[tstep*D : (tstep+1)*D])
		ddelta.UploadF32(deltaD[tstep*D : (tstep+1)*D])
		db.UploadF32(bD[tstep*N : (tstep+1)*N])
		dc.UploadF32(cD[tstep*N : (tstep+1)*N])
		if err := rec.SSMStep(du, ddelta, da, db, dc, ddk, dh, dy, D, N); err != nil {
			t.Fatalf("step %d SSMStep: %v", tstep, err)
		}
		if err := rec.Wait(); err != nil {
			t.Fatalf("step %d wait: %v", tstep, err)
		}
		y := make([]float32, D)
		if err := dy.DownloadF32(y); err != nil {
			t.Fatalf("step %d download: %v", tstep, err)
		}
		for d := 0; d < D; d++ {
			want := refT[0].AtF64(tstep, d)
			if math.IsNaN(float64(y[d])) || math.Abs(float64(y[d])-want) > 1e-4 {
				t.Fatalf("step %d chan %d: gpu %v vs ref %v", tstep, d, y[d], want)
			}
		}
	}
	t.Logf("cu_ssm_step reproduces the reference OpSSM full-scan across %d timesteps (Mamba decode primitive)", L)
}
