package cpu

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/archgold"
	"github.com/jxsl13/goai/tensor"
)

// wkvDyadicInputs is the frozen phase-A provenance fixture. Every element is
// produced only by bounded integer arithmetic and an exact power-of-two scale:
//
//	k[i] = (((37*i+11) mod 97)-48) * 2^-4
//	v[i] = (((29*i+7)  mod 89)-44) * 2^-5
//	w[c] = (((5*c+3)   mod 11)+1)  * 2^-5
//	u[c] = (((7*c+5)   mod 19)-9)  * 2^-4
//
// The bounded numerators are exactly representable in both F32 and F64. The
// 37-channel shape reaches vector remainders; 96 reaches complete groups.
func wkvDyadicInputs(dt tensor.Dtype, seq, d int) []*tensor.Tensor {
	mk := func(shape tensor.Shape, value func(int) float64) *tensor.Tensor {
		x := tensor.New(dt, shape)
		if dt == tensor.F64 {
			for i := range x.Numel() {
				x.Storage().F64()[i] = value(i)
			}
		} else {
			for i := range x.Numel() {
				x.Storage().F32()[i] = float32(value(i))
			}
		}
		return x
	}
	k := mk(tensor.Shape{seq, d}, func(i int) float64 {
		return math.Ldexp(float64((37*i+11)%97-48), -4)
	})
	v := mk(tensor.Shape{seq, d}, func(i int) float64 {
		return math.Ldexp(float64((29*i+7)%89-44), -5)
	})
	w := mk(tensor.Shape{d}, func(i int) float64 {
		return math.Ldexp(float64((5*i+3)%11+1), -5)
	})
	u := mk(tensor.Shape{d}, func(i int) float64 {
		return math.Ldexp(float64((7*i+5)%19-9), -4)
	})
	return []*tensor.Tensor{k, v, w, u}
}

// wkvInputSHA256 binds dtype, tensor order, shapes, and exact input bits. It is
// deliberately independent of the output digest used for native harvesting.
func wkvInputSHA256(in []*tensor.Tensor) string {
	h := sha256.New()
	var word [8]byte
	for tensorIndex, x := range in {
		binary.LittleEndian.PutUint64(word[:], uint64(tensorIndex))
		h.Write(word[:])
		h.Write([]byte(x.Dtype().String()))
		binary.LittleEndian.PutUint64(word[:], uint64(x.Ndim()))
		h.Write(word[:])
		for _, dim := range x.Shape() {
			binary.LittleEndian.PutUint64(word[:], uint64(dim))
			h.Write(word[:])
		}
		if x.Dtype() == tensor.F64 {
			for _, value := range x.Storage().F64()[:x.Numel()] {
				binary.LittleEndian.PutUint64(word[:], math.Float64bits(value))
				h.Write(word[:])
			}
		} else {
			for _, value := range x.Storage().F32()[:x.Numel()] {
				binary.LittleEndian.PutUint32(word[:4], math.Float32bits(value))
				h.Write(word[:4])
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func wkvOutputDigest(x *tensor.Tensor) uint64 {
	h := uint64(14695981039346656037)
	mix := func(word uint64, bytes int) {
		for shift := 0; shift < bytes*8; shift += 8 {
			h = (h ^ ((word >> shift) & 0xff)) * 1099511628211
		}
	}
	if x.Dtype() == tensor.F64 {
		for _, value := range x.Storage().F64()[:x.Numel()] {
			mix(math.Float64bits(value), 8)
		}
	} else {
		for _, value := range x.Storage().F32()[:x.Numel()] {
			mix(uint64(math.Float32bits(value)), 4)
		}
	}
	return h
}

func wkvOutputBitSnapshot(x *tensor.Tensor) []uint64 {
	bits := make([]uint64, x.Numel())
	if x.Dtype() == tensor.F64 {
		for i, value := range x.Storage().F64()[:x.Numel()] {
			bits[i] = math.Float64bits(value)
		}
	} else {
		for i, value := range x.Storage().F32()[:x.Numel()] {
			bits[i] = uint64(math.Float32bits(value))
		}
	}
	return bits
}

func requireWKVSnapshot(t *testing.T, label string, got *tensor.Tensor, want []uint64) {
	t.Helper()
	if got.Numel() != len(want) {
		t.Fatalf("%s extent: got %d want %d", label, got.Numel(), len(want))
	}
	gotBits := wkvOutputBitSnapshot(got)
	for i := range want {
		if gotBits[i] != want[i] {
			t.Fatalf("%s output[%d]: got %#016x want %#016x", label, i, gotBits[i], want[i])
		}
	}
}

func executeWKVDyadic(t *testing.T, be backend.Name, in []*tensor.Tensor) *tensor.Tensor {
	t.Helper()
	impl, ok := backend.Get(be)
	if !ok {
		t.Fatalf("backend %v not registered", be)
	}
	out, err := backend.Execute(backend.NewContext().WithBackend(impl), backend.OpWKV, in, nil)
	if err != nil {
		t.Fatalf("%v WKV: %v", be, err)
	}
	if len(out) != 1 {
		t.Fatalf("%v WKV returned %d tensors, want 1", be, len(out))
	}
	if out[0] == nil {
		t.Fatalf("%v WKV returned a nil output tensor", be)
	}
	want := in[0]
	if out[0].Dtype() != want.Dtype() || !out[0].Shape().Equal(want.Shape()) {
		t.Fatalf("%v WKV output metadata: got %s %v want %s %v", be,
			out[0].Dtype(), out[0].Shape(), want.Dtype(), want.Shape())
	}
	if out[0].Numel() == 0 || out[0].Numel() != want.Numel() {
		t.Fatalf("%v WKV output extent: got %d want nonzero %d", be, out[0].Numel(), want.Numel())
	}
	return out[0]
}

func requireWKVFinite(t *testing.T, be backend.Name, x *tensor.Tensor) {
	t.Helper()
	if x.Dtype() == tensor.F64 {
		for i, value := range x.Storage().F64()[:x.Numel()] {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				t.Fatalf("%v F64 output[%d] is non-finite: %v", be, i, value)
			}
		}
		return
	}
	for i, value := range x.Storage().F32()[:x.Numel()] {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("%v F32 output[%d] is non-finite: %v", be, i, value)
		}
	}
}

func requireWKVExact(t *testing.T, label string, got, want *tensor.Tensor) {
	t.Helper()
	if got.Dtype() != want.Dtype() || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("%s metadata differs: got %s %v want %s %v", label, got.Dtype(), got.Shape(), want.Dtype(), want.Shape())
	}
	if got.Dtype() == tensor.F64 {
		for i, gv := range got.Storage().F64()[:got.Numel()] {
			wv := want.Storage().F64()[i]
			if math.Float64bits(gv) != math.Float64bits(wv) {
				t.Fatalf("%s output[%d]: got %#016x want %#016x", label, i, math.Float64bits(gv), math.Float64bits(wv))
			}
		}
		return
	}
	for i, gv := range got.Storage().F32()[:got.Numel()] {
		wv := want.Storage().F32()[i]
		if math.Float32bits(gv) != math.Float32bits(wv) {
			t.Fatalf("%s output[%d]: got %#08x want %#08x", label, i, math.Float32bits(gv), math.Float32bits(wv))
		}
	}
}

func requireWKVF64Relative(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	for i, gv := range got.Storage().F64()[:got.Numel()] {
		wv := want.Storage().F64()[i]
		if math.IsNaN(gv) || math.IsNaN(wv) || math.IsInf(gv, 0) || math.IsInf(wv, 0) {
			t.Fatalf("SIMD F64 output[%d] has non-finite comparison: cpu=%v ref=%v", i, gv, wv)
		}
		denom := math.Max(1e-6, math.Abs(wv))
		rel := math.Abs(gv-wv) / denom
		if math.IsNaN(rel) || math.IsInf(rel, 0) || rel > 1e-10 {
			t.Fatalf("SIMD F64 output[%d]: cpu=%v ref=%v rel=%g exceeds 1e-10", i, gv, wv, rel)
		}
	}
}

func wkvSIMDExperimentEnabled() bool {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return false
	}
	for _, setting := range info.Settings {
		if setting.Key != "GOEXPERIMENT" {
			continue
		}
		for _, experiment := range strings.Split(setting.Value, ",") {
			if experiment == "simd" {
				return true
			}
		}
	}
	return false
}

func TestWKVDyadicNativeProvenance(t *testing.T) {
	if !archgold.Supported() {
		t.Skip(archgold.Reason)
	}
	buildMode := "default"
	if wkvSIMDExperimentEnabled() {
		buildMode = "simd"
	}
	shapes := [][2]int{{24, 37}, {64, 96}}
	for _, shape := range shapes {
		seq, d := shape[0], shape[1]
		for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
			name := fmt.Sprintf("%s/%dx%d", dt, seq, d)
			t.Run(name, func(t *testing.T) {
				type provenanceRow struct {
					be     backend.Name
					out    *tensor.Tensor
					digest uint64
				}
				var rows [2]provenanceRow
				var canonicalInputSHA string
				for rowIndex, be := range []backend.Name{backend.Ref, backend.CPU} {
					in := wkvDyadicInputs(dt, seq, d)
					beforeSHA := wkvInputSHA256(in)
					if canonicalInputSHA == "" {
						canonicalInputSHA = beforeSHA
					} else if beforeSHA != canonicalInputSHA {
						t.Fatalf("%v input SHA %s differs from %s", be, beforeSHA, canonicalInputSHA)
					}

					first := executeWKVDyadic(t, be, in)
					requireWKVFinite(t, be, first)
					firstBits := wkvOutputBitSnapshot(first)
					firstDigest := wkvOutputDigest(first)
					if afterSHA := wkvInputSHA256(in); afterSHA != beforeSHA {
						t.Fatalf("%v mutated inputs: before %s after %s", be, beforeSHA, afterSHA)
					}
					second := executeWKVDyadic(t, be, in)
					requireWKVFinite(t, be, second)
					requireWKVSnapshot(t, string(be)+" repeat", second, firstBits)
					if afterSHA := wkvInputSHA256(in); afterSHA != beforeSHA {
						t.Fatalf("%v repeat mutated inputs: before %s after %s", be, beforeSHA, afterSHA)
					}
					rows[rowIndex] = provenanceRow{be: be, out: second, digest: firstDigest}
				}
				if buildMode == "simd" && dt == tensor.F64 {
					requireWKVF64Relative(t, rows[1].out, rows[0].out)
				} else {
					requireWKVExact(t, "CPU vs Ref", rows[1].out, rows[0].out)
				}
				for _, row := range rows {
					t.Logf("WKV_PROVENANCE goos=%s goarch=%s buildmode=%s backend=%s dtype=%s shape=%dx%d input_sha256=%s output_digest=%d",
						runtime.GOOS, runtime.GOARCH, buildMode, row.be, dt, seq, d, canonicalInputSHA, row.digest)
				}
			})
		}
	}
}
