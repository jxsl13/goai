package vision

import (
	"slices"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func TestViTPatchifyBatchMatchesPerImagePacking(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F32, tensor.F64} {
		t.Run(dt.String(), func(t *testing.T) {
			m, err := NewViT(2, 8, 3, 7,
				WithViTDtype(dt), WithViTPatch(4), WithViTDim(16), WithViTHeads(2))
			if err != nil {
				t.Fatal(err)
			}
			x := tensor.New(dt, tensor.Shape{3, 2, 8, 8})
			var beforeF32 []float32
			var beforeF64 []float64
			if dt == tensor.F32 {
				for i := range x.Storage().F32() {
					x.Storage().F32()[i] = float32(i%37-18) / 19
				}
				beforeF32 = slices.Clone(x.Storage().F32())
			} else {
				for i := range x.Storage().F64() {
					x.Storage().F64()[i] = float64(i%37-18) / 19
				}
				beforeF64 = slices.Clone(x.Storage().F64())
			}

			got, err := m.patchifyBatch(x)
			if err != nil {
				t.Fatal(err)
			}
			imageValues := 2 * 8 * 8
			wantValues := got.Numel() / 3
			for b := range 3 {
				image := tensor.New(dt, tensor.Shape{2, 8, 8})
				if dt == tensor.F32 {
					copy(image.Storage().F32(), x.Storage().F32()[b*imageValues:(b+1)*imageValues])
				} else {
					copy(image.Storage().F64(), x.Storage().F64()[b*imageValues:(b+1)*imageValues])
				}
				want, err := m.patchify(image)
				if err != nil {
					t.Fatal(err)
				}
				for i := range wantValues {
					if gv, wv := got.AtF64(tensor.Unravel(b*wantValues+i, got.Shape())...), want.AtF64(tensor.Unravel(i, want.Shape())...); gv != wv {
						t.Fatalf("batch %d value %d: got %g want %g", b, i, gv, wv)
					}
				}
			}
			if dt == tensor.F32 && !slices.Equal(x.Storage().F32(), beforeF32) {
				t.Fatal("patchifyBatch mutated its input")
			}
			if dt == tensor.F64 && !slices.Equal(x.Storage().F64(), beforeF64) {
				t.Fatal("patchifyBatch mutated its input")
			}
		})
	}
}
