//go:build darwin && cgo

package metal_test

import (
	"errors"
	"math"
	"runtime"
	"testing"
	"time"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// §T157: a resident weight dropped WITHOUT an explicit Close is freed by a runtime cleanup — the
// cgo free runs from the cleanup goroutine without crashing or corrupting the device (a fresh
// resident matmul still works afterward). Best-effort: forces GC and gives the cleanup a moment.
func TestMetalResidentAutoFree(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const k, n = 256, 4
	w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 9))
	wq, _ := gguf.Quantize(w, gguf.Q8_0)
	// upload several without Close, drop the references
	for range 8 {
		rw, err := metal.UploadQWeightQ8_0(wq, n, k)
		if err != nil {
			t.Fatal(err)
		}
		_ = rw // goes out of scope, no Close
	}
	for range 3 {
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}
	// the device is still healthy: a fresh resident matmul is correct
	x := tensor.New(tensor.F32, tensor.Shape{1, k})
	for i := range k {
		x.Storage().F32()[i] = float32(qmRand(k, 1)[i])
	}
	rw, err := metal.UploadQWeightQ8_0(wq, n, k)
	if err != nil {
		t.Fatal(err)
	}
	defer rw.Close()
	got, err := rw.QMatMul(x)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := gguf.QMatMul(x, wq, gguf.Q8_0, n, k)
	for ni := range n {
		if g, r := got.AtF64(0, ni), ref.AtF64(0, ni); math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
			t.Errorf("[0,%d] after auto-free: %g vs %g", ni, g, r)
		}
	}
}

// §T154: metal.Backend implements backend.ResidentQuantMatMuler — UploadQuant returns a working
// resident weight for Q8_0 and ErrQuantUnsupported for a type without a resident kernel (so
// QuantLinear falls back to the per-call path).
func TestMetalBackendResidentQuantMatMuler(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	var rb backend.ResidentQuantMatMuler = metal.Backend{}
	const k, n = 256, 5
	x := tensor.New(tensor.F32, tensor.Shape{2, k})
	for i := range 2 * k {
		x.Storage().F32()[i] = float32(qmRand(2*k, 1)[i])
	}
	// every resident-supported k-quant/Q8_0 matches its per-call reference.
	types := []struct {
		qt   gguf.QuantType
		code uint32
	}{{gguf.Q8_0, 8}, {gguf.Q4_0, 2}, {gguf.Q4_K, 12}, {gguf.Q5_K, 13}, {gguf.Q6_K, 14}, {gguf.Q2_K, 10}, {gguf.Q3_K, 11}}
	for _, tc := range types {
		w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 2))
		wq, _ := gguf.Quantize(w, tc.qt)
		rw, err := rb.UploadQuant(wq, tc.code, n, k)
		if err != nil {
			t.Fatalf("%v upload: %v", tc.qt, err)
		}
		got, err := rw.QMatMul(x)
		if err != nil {
			t.Fatalf("%v: %v", tc.qt, err)
		}
		ref, _ := gguf.QMatMul(x, wq, tc.qt, n, k)
		for mi := range 2 {
			for ni := range n {
				if g, r := got.AtF64(mi, ni), ref.AtF64(mi, ni); math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
					t.Errorf("%v [%d,%d] %g vs %g", tc.qt, mi, ni, g, r)
				}
			}
		}
		_ = rw.Close()
	}
	// a type without a resident kernel (Q4_1, code 3) → ErrQuantUnsupported → CPU fallback.
	if _, err := rb.UploadQuant(make([]byte, 100), 3, n, k); !errors.Is(err, backend.ErrQuantUnsupported) {
		t.Errorf("Q4_1 resident: got %v, want ErrQuantUnsupported", err)
	}
}

// §V3/§V11 (§T153): a device-resident Q8_0 weight gives the SAME result as the per-call
// QMatMulQ8_0 (same kernel, same bytes) — uploading the weight once and reusing it is a pure perf
// change, not a numerical one. Covers decode (M=1) and prefill; also checks Close.
func TestMetalResidentQWeight(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	cases := []struct{ m, k, n int }{
		{1, 32, 8}, {1, 256, 16}, {4, 64, 5}, {8, 128, 12},
	}
	for _, c := range cases {
		xf := tensor.New(tensor.F32, tensor.Shape{c.m, c.k})
		for i := range c.m * c.k {
			xf.Storage().F32()[i] = float32(qmRand(c.m*c.k, 1)[i])
		}
		w := tensor.FromFloat64(tensor.Shape{c.n, c.k}, qmRand(c.n*c.k, 2))
		wq, err := gguf.Quantize(w, gguf.Q8_0)
		if err != nil {
			t.Fatal(err)
		}
		perCall, err := metal.QMatMulQ8_0(xf, wq, c.n, c.k)
		if err != nil {
			t.Fatal(err)
		}
		res, err := metal.UploadQWeightQ8_0(wq, c.n, c.k)
		if err != nil {
			t.Fatalf("case %+v: upload: %v", c, err)
		}
		got, err := res.QMatMul(xf)
		if err != nil {
			t.Fatalf("case %+v: %v", c, err)
		}
		for mi := range c.m {
			for ni := range c.n {
				if g, r := got.AtF64(mi, ni), perCall.AtF64(mi, ni); math.Abs(g-r) > 1e-5*math.Max(1, math.Abs(r)) {
					t.Errorf("case %+v [%d,%d]: resident %g vs per-call %g", c, mi, ni, g, r)
				}
			}
		}
		if err := res.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
		_ = res.Close() // idempotent
		if _, err := res.QMatMul(xf); err == nil {
			t.Error("QMatMul after Close should error")
		}
	}
}

// §V7-adjacent: bad K / weight length is rejected at upload.
func TestMetalResidentValidation(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	if _, err := metal.UploadQWeightQ8_0(make([]byte, 34), 1, 40); err == nil {
		t.Error("K=40 (not a multiple of 32) must be rejected")
	}
	if _, err := metal.UploadQWeightQ8_0(make([]byte, 10), 1, 32); err == nil {
		t.Error("wrong weight length must be rejected")
	}
}

// BenchmarkResidentDecode compares a decode loop (M=1 over a large weight) with the weight
// device-resident (uploaded once) vs re-uploaded per call — quantifying the residency win (§T153,
// §B). On Apple UMA the "upload" is a shared-buffer memcpy; on a discrete GPU it is a bus transfer,
// so the win is larger there.
func BenchmarkResidentDecode(b *testing.B) {
	if !metal.Available() {
		b.Skip("metal device unavailable")
	}
	const k, n = 4096, 4096
	x := tensor.New(tensor.F32, tensor.Shape{1, k}) // M=1 decode
	for i := range k {
		x.Storage().F32()[i] = float32(qmRand(1, i)[0])
	}
	w := tensor.FromFloat64(tensor.Shape{n, k}, qmRand(n*k, 3))
	wq, _ := gguf.Quantize(w, gguf.Q8_0)
	b.Run("resident", func(b *testing.B) {
		res, err := metal.UploadQWeightQ8_0(wq, n, k)
		if err != nil {
			b.Fatal(err)
		}
		defer res.Close()
		b.ResetTimer()
		for range b.N {
			if _, err := res.QMatMul(x); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("percall", func(b *testing.B) {
		for range b.N {
			if _, err := metal.QMatMulQ8_0(x, wq, n, k); err != nil {
				b.Fatal(err)
			}
		}
	})
}
