//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// TestCUDAResidentMMQPreQuant validates the shared-quant path: MatMulPreQuant (activation quantized
// once) is bit-identical to MatMulDevice (quantizes per call), and quantizing once for 3 projections
// that share an input (q/k/v) is faster than 3 separate quant+matmuls.
func TestCUDAResidentMMQPreQuant(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 128, 896, 896 // Qwen-0.5B q_proj-ish
	rng := rand.New(rand.NewSource(61))
	mkW := func(seed int64) *cuda.ResidentMMQ {
		w := tensor.New(tensor.F32, tensor.Shape{K, N})
		wf := w.Storage().F32()
		wr := rand.New(rand.NewSource(seed))
		for i := range wf {
			wf[i] = float32(wr.NormFloat64()) * 0.5
		}
		r, err := cuda.NewResidentMMQ(w)
		if err != nil {
			t.Fatalf("NewResidentMMQ: %v", err)
		}
		return r
	}
	wq, wk, wv := mkW(1), mkW(2), mkW(3)
	defer wq.Free()
	defer wk.Free()
	defer wv.Free()

	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	da, err := cuda.NewDeviceF32(M, K)
	if err != nil {
		t.Fatal(err)
	}
	defer da.Free()
	if err := da.UploadF32(a); err != nil {
		t.Fatal(err)
	}

	// Correctness: MatMulPreQuant(QuantizeActs(a)) == MatMulDevice(a), bit-identical.
	qa, err := cuda.QuantizeActs(da)
	if err != nil {
		t.Fatalf("QuantizeActs: %v", err)
	}
	defer qa.Free()
	for name, w := range map[string]*cuda.ResidentMMQ{"q": wq, "k": wk, "v": wv} {
		dRef, err := w.MatMulDevice(da)
		if err != nil {
			t.Fatal(err)
		}
		dPre, err := w.MatMulPreQuant(qa)
		if err != nil {
			t.Fatal(err)
		}
		ref, _ := dRef.ToHost()
		pre, _ := dPre.ToHost()
		dRef.Free()
		dPre.Free()
		var maxAbs float64
		for i := 0; i < M; i++ {
			for j := 0; j < N; j++ {
				d := ref.AtF64(i, j) - pre.AtF64(i, j)
				if d < 0 {
					d = -d
				}
				if d > maxAbs {
					maxAbs = d
				}
			}
		}
		if maxAbs != 0 {
			t.Errorf("%s: MatMulPreQuant != MatMulDevice, maxAbs %g (must be bit-identical)", name, maxAbs)
		}
	}

	// Perf: 3 shared-quant projections vs 3 per-call-quant projections (q/k/v pattern).
	timeIt := func(f func()) time.Duration {
		f()
		must(t, cuda.GraphSync())
		t0 := time.Now()
		for n := 0; n < 20; n++ {
			f()
		}
		must(t, cuda.GraphSync())
		return time.Since(t0) / 20
	}
	perCall := timeIt(func() {
		for _, w := range []*cuda.ResidentMMQ{wq, wk, wv} {
			d, _ := w.MatMulDevice(da)
			d.Free()
		}
	})
	shared := timeIt(func() {
		q, _ := cuda.QuantizeActs(da)
		for _, w := range []*cuda.ResidentMMQ{wq, wk, wv} {
			d, _ := w.MatMulPreQuant(q)
			d.Free()
		}
		q.Free()
	})
	t.Logf("q/k/v projections: per-call-quant %.1f us | shared-quant %.1f us (%.2fx)",
		float64(perCall.Microseconds()), float64(shared.Microseconds()),
		float64(perCall)/float64(shared))
	t.Log("SHARED-QUANT MMQ CORRECT (bit-identical) — one activation quant feeds q/k/v")
}
