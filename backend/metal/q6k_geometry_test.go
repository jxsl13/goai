//go:build darwin && cgo

package metal_test

import (
	"encoding/binary"
	"hash/fnv"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/internal/archgold"
	"github.com/jxsl13/goai/tensor"
)

func q6KGeometryFixture(k, n int) (*tensor.Tensor, []byte) {
	x := tensor.New(tensor.F32, tensor.Shape{1, k})
	for i := range k {
		x.Storage().F32()[i] = float32(i%17-8) / 8
	}
	raw := make([]byte, n*(k/256)*210)
	for i := range raw {
		raw[i] = byte(i*29 + 13)
	}
	for block := 0; block < len(raw); block += 210 {
		// IEEE binary16 1.0, so arbitrary quant bytes cannot turn the fixture nonfinite.
		raw[block+208], raw[block+209] = 0x00, 0x3c
	}
	return x, raw
}

func f32Digest(values []float32) uint64 {
	h := fnv.New64a()
	var word [4]byte
	for _, value := range values {
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(value))
		_, _ = h.Write(word[:])
	}
	return h.Sum64()
}

// TestQ6KTwoRowBitExact freezes the shipped cooperative kernel itself. The scalar comparison in
// qmatmul_test.go permits reassociation within 2e-5, which is correct for cross-kernel parity but
// cannot guard a refactor that promises to preserve each row's exact arithmetic.
func TestQ6KTwoRowBitExact(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const k, n = 512, 7
	x, raw := q6KGeometryFixture(k, n)
	previous := metal.SetQ6KCooperative(true)
	defer metal.SetQ6KCooperative(previous)
	got, err := metal.QMatMulQ6_K(x, raw, n, k)
	if err != nil {
		t.Fatal(err)
	}
	digest := f32Digest(got.Storage().F32())
	want := archgold.Pick(2302814654717630432, 0)
	if want == 0 {
		t.Skip("Q6_K Metal bit-exact golden is not recorded for this architecture")
	}
	if digest != want {
		t.Fatalf("Q6_K two-row digest = %d, want %d", digest, want)
	}
}
