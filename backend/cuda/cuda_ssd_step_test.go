//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDASSDStep checks cu_ssd_step (Recorder.SSDStep) — one decode step of the Mamba-2 SSD scan —
// against the reference nn.SSDRecurrent applied PER HEAD over a full sequence. Mamba-2 restricts the
// SSM decay to a scalar per head (a = exp(Δ·A[h])) and shares B/C across a group of heads; the value
// is Δ-scaled and a per-head D·x skip is added. Running the step kernel L times (per-head state
// carried between calls, starting zeroed) must reproduce, for every head and timestep, the reference
// SSDRecurrent output + skip — exactly the decode primitive a GPU Mamba-2 block records.
func TestCUDASSDStep(t *testing.T) {
	skipNoGPU(t)
	const L, H, P, G, N = 6, 4, 3, 2, 5 // seq, heads, head_dim, groups, state; intermediate = H*P
	const HP, gN = H * P, G * N
	headsPerGroup := H / G

	gen := func(n, seed int, scale float64) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32((((i+seed)*13)%29 - 14)) * float32(scale)
		}
		return x
	}
	xD := gen(L*HP, 1, 0.05) // conv-output value x [L, H*P]
	dtD := gen(L*H, 2, 0.15) // Δ per head (already softplus'd in this test — kernel takes delta directly)
	bD := gen(L*gN, 3, 0.07) // grouped B [L, G*N]
	cD := gen(L*gN, 4, 0.06) // grouped C [L, G*N]
	aLog := gen(H, 5, 0.1)   // A_log per head
	dSkip := gen(H, 6, 0.2)  // D skip per head

	// Δ must be strictly positive (softplus output); make the test delta positive.
	delta := make([]float32, L*H)
	for i := range delta {
		delta[i] = float32(math.Log1p(math.Exp(float64(dtD[i])))) // softplus, host-side
	}
	A := make([]float32, H)
	for h := range A {
		A[h] = float32(-math.Exp(float64(aLog[h]))) // A = −exp(A_log)
	}

	// Reference: per head h (group g), run nn.SSDRecurrent on the Δ-scaled value with scalar decay
	// exp(Δ·A) and the group's B/C, then add the D·x skip. yRef[t][h*P+j].
	yRef := make([][]float64, L)
	for t := range yRef {
		yRef[t] = make([]float64, HP)
	}
	for h := 0; h < H; h++ {
		g := h / headsPerGroup
		xs := tensor.New(tensor.F64, tensor.Shape{L, P})
		as := tensor.New(tensor.F64, tensor.Shape{L})
		bm := tensor.New(tensor.F64, tensor.Shape{L, N})
		cm := tensor.New(tensor.F64, tensor.Shape{L, N})
		for tt := 0; tt < L; tt++ {
			dl := float64(delta[tt*H+h])
			as.SetF64(math.Exp(dl*float64(A[h])), tt)
			for j := 0; j < P; j++ {
				xs.SetF64(float64(xD[tt*HP+h*P+j])*dl, tt, j)
			}
			for i := 0; i < N; i++ {
				bm.SetF64(float64(bD[tt*gN+g*N+i]), tt, i)
				cm.SetF64(float64(cD[tt*gN+g*N+i]), tt, i)
			}
		}
		yh, err := nn.SSDRecurrent(xs, as, bm, cm)
		if err != nil {
			t.Fatalf("SSDRecurrent head %d: %v", h, err)
		}
		for tt := 0; tt < L; tt++ {
			for j := 0; j < P; j++ {
				yRef[tt][h*P+j] = yh.AtF64(tt, j) + float64(dSkip[h])*float64(xD[tt*HP+h*P+j])
			}
		}
	}

	rec, err := cuda.NewRecorder()
	if err != nil {
		t.Fatalf("NewRecorder: %v", err)
	}
	defer rec.Free()

	dA, _ := cuda.NewDeviceBufferF32(A)
	dD, _ := cuda.NewDeviceBufferF32(dSkip)
	dstate, _ := cuda.NewDeviceBufferF32(make([]float32, H*N*P)) // per-head [N,P] state, explicitly zeroed
	dx, _ := cuda.NewDeviceF32(1, HP)
	ddelta, _ := cuda.NewDeviceF32(1, H)
	dB, _ := cuda.NewDeviceF32(1, gN)
	dC, _ := cuda.NewDeviceF32(1, gN)
	dy, _ := cuda.NewDeviceF32(1, HP)
	defer func() {
		dA.Free()
		dD.Free()
		dstate.Free()
		dx.Free()
		ddelta.Free()
		dB.Free()
		dC.Free()
		dy.Free()
	}()

	for tstep := 0; tstep < L; tstep++ {
		dx.UploadF32(xD[tstep*HP : (tstep+1)*HP])
		ddelta.UploadF32(delta[tstep*H : (tstep+1)*H])
		dB.UploadF32(bD[tstep*gN : (tstep+1)*gN])
		dC.UploadF32(cD[tstep*gN : (tstep+1)*gN])
		if err := rec.SSDStep(dx, ddelta, dA, dB, dC, dD, dstate, dy, H, P, G, N); err != nil {
			t.Fatalf("step %d SSDStep: %v", tstep, err)
		}
		if err := rec.Wait(); err != nil {
			t.Fatalf("step %d wait: %v", tstep, err)
		}
		out := make([]float32, HP)
		if err := dy.DownloadF32(out); err != nil {
			t.Fatalf("step %d download: %v", tstep, err)
		}
		for o := 0; o < HP; o++ {
			want := yRef[tstep][o]
			if math.IsNaN(float64(out[o])) || math.Abs(float64(out[o])-want) > 1e-4*math.Max(1, math.Abs(want)) {
				t.Fatalf("step %d out[%d] (head %d,j %d): gpu %v vs ref %v", tstep, o, o/P, o%P, out[o], want)
			}
		}
	}
	t.Logf("cu_ssd_step reproduces nn.SSDRecurrent per-head (+D skip) across %d timesteps, %d heads / %d groups (Mamba-2 decode)", L, H, G)
}
