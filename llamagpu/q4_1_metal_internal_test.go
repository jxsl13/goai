//go:build darwin && cgo

package llamagpu

import (
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// The full-device decoder opts into native Q4_1 residency even though generic host-bound
// QuantLinear deliberately remains on the faster M2 ARM64 path.
func TestMetalUploadQ4_1UsesRecorderOnlyResidentPath(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 7, 64
	w := tensor.New(tensor.F32, tensor.Shape{n, k})
	for i := range w.Storage().F32() {
		w.Storage().F32()[i] = float32((i%31)-15) * 0.125
	}
	wq, err := gguf.Quantize(w, gguf.Q4_1)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := metalUploadQWeight(wq, uint32(gguf.Q4_1), n, k)
	if err != nil {
		t.Fatal(err)
	}
	rw, ok := raw.(*metal.ResidentQWeight)
	if !ok {
		t.Fatalf("Q4_1 upload type %T, want *metal.ResidentQWeight", raw)
	}
	if err := rw.Close(); err != nil {
		t.Fatal(err)
	}
}
