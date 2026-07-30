package vision

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// patchifyReference is an independent transcription of the flattening this package documents:
// row-major over the patch grid, channel-major within a patch. Written from the spec rather
// than from patchifyImage, so agreeing with it is evidence about the layout and not a
// restatement of the implementation.
func patchifyReference(img []float64, channels, size, p int) []float64 {
	grid := size / p
	out := make([]float64, 0, grid*grid*channels*p*p)
	for py := 0; py < grid; py++ {
		for px := 0; px < grid; px++ {
			for c := 0; c < channels; c++ {
				for dy := 0; dy < p; dy++ {
					for dx := 0; dx < p; dx++ {
						out = append(out, img[((c*size)+py*p+dy)*size+px*p+dx])
					}
				}
			}
		}
	}
	return out
}

// TestPatchifyImageBitIdentical is the §V22 gate for routing ViT, MLPMixer and MAE through one
// contiguous-run copy. Tolerance is ZERO: the change moves data and performs no arithmetic, so
// a relative-tolerance check would pass a genuine layout error on smooth input and is not the
// bar here.
//
// The geometries deliberately include channels > 1 and p > 1 together — with either equal to 1
// several index terms collapse and a transposed layout would still compare equal — plus a
// non-square patch count and the p == size degenerate case where the whole image is one patch.
func TestPatchifyImageBitIdentical(t *testing.T) {
	for _, g := range []struct{ channels, size, p int }{
		{3, 32, 4}, // the ViT and Mixer benchmark geometry
		{3, 8, 2},
		{1, 8, 4},
		{4, 6, 3},
		{2, 4, 4}, // one patch covering the whole image
	} {
		for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
			rng := rand.New(rand.NewPCG(9, 9))
			n := g.channels * g.size * g.size
			raw := make([]float64, n)
			for i := range raw {
				// Round F32 cases through float32 so the reference and the implementation
				// start from the same representable values; any difference then comes from
				// the layout, which is what this test is about.
				v := rng.NormFloat64()
				if dt == tensor.F32 {
					v = float64(float32(v))
				}
				raw[i] = v
			}
			img := tensor.FromFloat64(tensor.Shape{g.channels, g.size, g.size}, raw)
			if dt == tensor.F32 {
				img = img.Cast(tensor.F32)
			}
			got, err := patchifyImage(img, dt, g.channels, g.size, g.p, "test")
			if err != nil {
				t.Fatalf("%v %v: %v", g, dt, err)
			}
			want := patchifyReference(raw, g.channels, g.size, g.p)
			grid := g.size / g.p
			if sh := got.Shape(); sh[0] != grid*grid || sh[1] != g.channels*g.p*g.p {
				t.Fatalf("%v %v: shape %v", g, dt, sh)
			}
			var flat []float64
			if dt == tensor.F32 {
				f32 := got.Storage().F32()
				flat = make([]float64, len(f32))
				for i, v := range f32 {
					flat[i] = float64(v)
				}
			} else {
				flat = got.Storage().F64()
			}
			if len(flat) != len(want) {
				t.Fatalf("%v %v: %d elements, want %d", g, dt, len(flat), len(want))
			}
			for i := range want {
				if math.Float64bits(flat[i]) != math.Float64bits(want[i]) {
					t.Fatalf("%v %v: element %d = %v (%016x), want %v (%016x)",
						g, dt, i, flat[i], math.Float64bits(flat[i]), want[i], math.Float64bits(want[i]))
				}
			}
		}
	}
}

// TestPatchifyImageRejectsDtype pins the error path the three models share, including that the
// message still names the caller — the reason patchifyImage takes a model name at all.
func TestPatchifyImageRejectsDtype(t *testing.T) {
	img := tensor.New(tensor.F32, tensor.Shape{1, 4, 4})
	_, err := patchifyImage(img, tensor.BF16, 1, 4, 2, "ViT")
	if err == nil {
		t.Fatal("want an error for an unsupported dtype")
	}
	if want := "vision: ViT supports F32/F64"; err.Error()[:len(want)] != want {
		t.Fatalf("error %q does not name the caller", err)
	}
}
