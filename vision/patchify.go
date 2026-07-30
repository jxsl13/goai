package vision

import (
	"fmt"

	"github.com/jxsl13/goai/tensor"
)

// patchifyImage flattens one [C,H,W] image into [n, C·p·p] patch rows — row-major over the
// patch grid, channel-major within a patch (the ViT flattening). ViT, MLPMixer and MAE all
// need exactly this and each carried its own byte-identical copy; model names the caller so
// the unsupported-dtype error still says which model rejected the dtype.
//
// The image carries no gradient, so this runs outside the tape and does its own copy rather
// than dispatching an op.
//
// THE INNERMOST RUN IS CONTIGUOUS AT BOTH ENDS, which is the whole optimization. The source
// index is ((c·size)+py·p+dy)·size + px·p + dx, so consecutive dx are adjacent; the
// destination is filled in exactly that traversal order, so consecutive dx are adjacent
// there too. Each run of p elements is therefore one copy() instead of p per-element
// reads. What the previous form paid per pixel: an indirect call through a closure returned
// by makeReader, a widen to float64, a bounds-checked append into a staging buffer sized
// n·C·p·p (loop-invariant, allocated per image), and then a second full pass narrowing back
// out. The staging buffer is gone entirely.
//
// Bit-identical: there is no arithmetic here, the traversal order is unchanged, and the
// float32-to-float64-to-float32 round trip the old code performed is an exact identity.
//
// Mixed dtype pairs and any view the flat helpers reject fall back to the original
// per-element reader. Those are not hot — a model's parameters and its images normally share
// a dtype — and keeping one general path costs less than four specialized ones.
func patchifyImage(img *tensor.Tensor, dt tensor.Dtype, channels, size, p int, model string) (*tensor.Tensor, error) {
	if dt != tensor.F32 && dt != tensor.F64 {
		return nil, fmt.Errorf("vision: %s supports F32/F64, got %v", model, dt)
	}
	grid := size / p
	n := grid * grid
	out := tensor.New(dt, tensor.Shape{n, channels * p * p})
	src := img.Contiguous()
	if dt == tensor.F64 && src.Dtype() == tensor.F64 {
		if s := swinFlatF64(src); s != nil {
			patchifyRunsF64(out.Storage().F64(), s, channels, size, p, grid)
			return out, nil
		}
	}
	if dt == tensor.F32 && src.Dtype() == tensor.F32 {
		if s := swinFlatF32(src); s != nil {
			patchifyRunsF32(out.Storage().F32(), s, channels, size, p, grid)
			return out, nil
		}
	}
	read := makeReader(src)
	o := 0
	switch dt {
	case tensor.F64:
		d := out.Storage().F64()
		patchifyWalk(channels, size, p, grid, func(base int) {
			for dx := 0; dx < p; dx++ {
				d[o+dx] = read(base + dx)
			}
			o += p
		})
	default: // tensor.F32
		d := out.Storage().F32()
		patchifyWalk(channels, size, p, grid, func(base int) {
			for dx := 0; dx < p; dx++ {
				d[o+dx] = float32(read(base + dx))
			}
			o += p
		})
	}
	return out, nil
}

// patchifyWalk visits the start index of every contiguous source run of length p, in the
// order the destination rows are filled. Factored out so the three dtype paths cannot drift
// apart on the index arithmetic, which is the one thing here that could silently produce a
// wrong image.
func patchifyWalk(channels, size, p, grid int, run func(base int)) {
	for py := 0; py < grid; py++ {
		for px := 0; px < grid; px++ {
			for c := 0; c < channels; c++ {
				for dy := 0; dy < p; dy++ {
					run(((c*size)+py*p+dy)*size + px*p)
				}
			}
		}
	}
}

func patchifyRunsF64(dst, src []float64, channels, size, p, grid int) {
	o := 0
	patchifyWalk(channels, size, p, grid, func(base int) {
		copy(dst[o:o+p], src[base:base+p])
		o += p
	})
}

func patchifyRunsF32(dst, src []float32, channels, size, p, grid int) {
	o := 0
	patchifyWalk(channels, size, p, grid, func(base int) {
		copy(dst[o:o+p], src[base:base+p])
		o += p
	})
}
