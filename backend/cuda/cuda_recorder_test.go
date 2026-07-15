//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// dev uploads a host [rows,cols] tensor to a fresh device buffer (test helper).
func dev(t *testing.T, rows, cols, seed int) (*cuda.DeviceF32, []float32) {
	t.Helper()
	x := bench.RandF32(tensor.Shape{rows, cols}, uint64(seed))
	d, err := cuda.UploadF32(x)
	must(t, err)
	return d, append([]float32{}, x.Contiguous().Storage().F32()...)
}

func closeVec(t *testing.T, tag string, got []float32, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", tag, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol+tol*math.Abs(float64(want[i])) {
			t.Fatalf("%s[%d]=%v want %v", tag, i, got[i], want[i])
		}
	}
}

// The recorder's resident-Q8 GEMV (QMatMulResident) must match a host f32 matmul within
// Q8 quantization tolerance — the quantized-decode path the llamagpu NewQuantCUDA uses.
func TestCUDARecorderQ8(t *testing.T) {
	skipNoGPU(t)
	rec, err := cuda.NewRecorder()
	must(t, err)
	defer rec.Free()

	const m, k, n = 2, 64, 96 // k,n multiples of 32
	dx, x := dev(t, m, k, 11)
	// Weight is [K,N] (in,out) — NewResidentBQ8's orientation; host ref uses the pre-quant values.
	wten := bench.RandF32(tensor.Shape{k, n}, 12)
	wf := wten.Contiguous().Storage().F32()
	w, err := cuda.NewResidentBQ8(wten)
	must(t, err)
	defer w.Close()
	do, err := cuda.NewDeviceF32(m, n)
	must(t, err)
	defer do.Free()

	must(t, rec.QMatMulResident(dx, w, do, m))
	must(t, rec.Wait())
	got, err := do.ToHost()
	must(t, err)

	gf := got.Contiguous().Storage().F32()
	var maxAbs, maxRel float64
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(x[i*k+p]) * float64(wf[p*n+j])
			}
			d := math.Abs(float64(gf[i*n+j]) - s)
			if d > maxAbs {
				maxAbs = d
			}
			// Relative error only where the output is significant (near-zero outputs
			// blow up the ratio without signalling real error).
			if math.Abs(s) > 1 {
				if rel := d / math.Abs(s); rel > maxRel {
					maxRel = rel
				}
			}
			if d > 0.05+0.03*math.Abs(s) {
				t.Fatalf("Q8[%d,%d]=%v want %v (Δ%g)", i, j, gf[i*n+j], s, d)
			}
		}
	}
	t.Logf("QMatMulResident (resident Q8 GEMV) == host f32 within Q8 tol (max abs err %.4f, max rel err %.3f%% over |out|>1)", maxAbs, maxRel*100)
}

// The CUDA Recorder (the llamagpu backend-agnostic decode interface on NVIDIA) must
// produce the same results as host references for the ops with real wiring risk:
// the 3-buffer MatMul/MatMulAcc, the op-code + copy-path Unary/Binary, and the
// GQA-causal MHA offset chain.
func TestCUDARecorder(t *testing.T) {
	skipNoGPU(t)
	rec, err := cuda.NewRecorder()
	must(t, err)
	defer rec.Free()

	// --- MatMul: c = a·b, then MatMulAcc: c += a·b ---
	{
		const m, k, n = 3, 5, 4
		da, a := dev(t, m, k, 1)
		db, b := dev(t, k, n, 2)
		dc, err := cuda.NewDeviceF32(m, n)
		must(t, err)
		must(t, rec.MatMul(da, db, dc, m, k, n))
		must(t, rec.Wait())
		got, err := dc.ToHost()
		must(t, err)
		want := make([]float32, m*n)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var s float64
				for p := 0; p < k; p++ {
					s += float64(a[i*k+p]) * float64(b[p*n+j])
				}
				want[i*n+j] = float32(s)
			}
		}
		closeVec(t, "MatMul", got.Contiguous().Storage().F32(), want, 1e-4)
		// MatMulAcc: dc already holds a·b; add another a·b → 2·(a·b).
		must(t, rec.MatMulAcc(da, db, dc, m, k, n))
		must(t, rec.Wait())
		got2, err := dc.ToHost()
		must(t, err)
		want2 := make([]float32, m*n)
		for i := range want {
			want2[i] = 2 * want[i]
		}
		closeVec(t, "MatMulAcc", got2.Contiguous().Storage().F32(), want2, 1e-4)
		da.Free()
		db.Free()
		dc.Free()
		t.Log("MatMul + MatMulAcc == host reference")
	}

	// --- Unary SiLU with o≠x (exercises the copy path) ---
	{
		const n = 16
		dx, x := dev(t, 1, n, 3)
		do, err := cuda.NewDeviceF32(1, n)
		must(t, err)
		must(t, rec.Unary(dx, do, 6)) // unarySiLU
		must(t, rec.Wait())
		got, err := do.ToHost()
		must(t, err)
		want := make([]float32, n)
		for i := range x {
			want[i] = float32(float64(x[i]) / (1 + math.Exp(-float64(x[i]))))
		}
		closeVec(t, "Unary/SiLU", got.Contiguous().Storage().F32(), want, 1e-4)
		dx.Free()
		do.Free()
		t.Log("Unary SiLU (o≠x copy path) == host reference")
	}

	// --- Binary: add (o≠a), mul (o≠a), swiglu (o==a) ---
	{
		const n = 12
		da, a := dev(t, 1, n, 4)
		db, b := dev(t, 1, n, 5)
		do, err := cuda.NewDeviceF32(1, n)
		must(t, err)
		must(t, rec.Binary(da, db, do, 0)) // add → o = a+b
		must(t, rec.Wait())
		got, err := do.ToHost()
		must(t, err)
		wantAdd := make([]float32, n)
		for i := range a {
			wantAdd[i] = a[i] + b[i]
		}
		closeVec(t, "Binary/add", got.Contiguous().Storage().F32(), wantAdd, 1e-5)
		must(t, rec.Binary(da, db, do, 2)) // mul → o = a*b
		must(t, rec.Wait())
		got, err = do.ToHost()
		must(t, err)
		wantMul := make([]float32, n)
		for i := range a {
			wantMul[i] = a[i] * b[i]
		}
		closeVec(t, "Binary/mul", got.Contiguous().Storage().F32(), wantMul, 1e-5)
		// swiglu in place: gate = silu(gate)*up, o==a.
		must(t, rec.Binary(da, db, da, 6))
		must(t, rec.Wait())
		got, err = da.ToHost()
		must(t, err)
		wantSw := make([]float32, n)
		for i := range a {
			s := float64(a[i]) / (1 + math.Exp(-float64(a[i])))
			wantSw[i] = float32(s * float64(b[i]))
		}
		closeVec(t, "Binary/swiglu", got.Contiguous().Storage().F32(), wantSw, 1e-4)
		da.Free()
		db.Free()
		do.Free()
		t.Log("Binary add/mul/swiglu == host reference")
	}

	// --- MHA: GQA (2 q-heads share 1 kv-head), causal, prefill sq==sk ---
	{
		const sq, sk, heads, kvHeads, dk = 3, 3, 2, 1, 4
		dmw := heads * dk
		kvw := kvHeads * dk
		dq, q := dev(t, sq, dmw, 6)
		dk_, k := dev(t, sk, kvw, 7)
		dv, v := dev(t, sk, kvw, 8)
		do, err := cuda.NewDeviceF32(sq, dmw)
		must(t, err)
		scale := float32(1 / math.Sqrt(float64(dk)))
		must(t, rec.MHA(dq, dk_, dv, do, sq, sk, dmw, heads, kvHeads, dk, 1, 0, scale))
		must(t, rec.Wait())
		got, err := do.ToHost()
		must(t, err)

		// Host GQA causal reference (offset = sk-sq = 0 ⇒ j ≤ i).
		want := make([]float32, sq*dmw)
		for h := 0; h < heads; h++ {
			kvh := h / (heads / kvHeads)
			for i := 0; i < sq; i++ {
				logits := make([]float64, i+1)
				maxL := math.Inf(-1)
				for j := 0; j <= i; j++ {
					var s float64
					for c := 0; c < dk; c++ {
						s += float64(q[i*dmw+h*dk+c]) * float64(k[j*kvw+kvh*dk+c])
					}
					s *= float64(scale)
					logits[j] = s
					if s > maxL {
						maxL = s
					}
				}
				var sum float64
				for j := 0; j <= i; j++ {
					logits[j] = math.Exp(logits[j] - maxL)
					sum += logits[j]
				}
				for c := 0; c < dk; c++ {
					var acc float64
					for j := 0; j <= i; j++ {
						acc += (logits[j] / sum) * float64(v[j*kvw+kvh*dk+c])
					}
					want[i*dmw+h*dk+c] = float32(acc)
				}
			}
		}
		closeVec(t, "MHA", got.Contiguous().Storage().F32(), want, 1e-4)
		dq.Free()
		dk_.Free()
		dv.Free()
		do.Free()
		t.Log("MHA (GQA, causal, prefill) == host reference")
	}
}
