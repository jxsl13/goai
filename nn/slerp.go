package nn

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/tensor"
)

// SLERPDotThreshold is the |cos Ω| above which SLERP falls back to linear interpolation:
// when the two tensors are nearly parallel sin Ω → 0 and the spherical formula is unstable.
const SLERPDotThreshold = 0.9995

// SLERP spherically interpolates two equal-shaped weight tensors (Shoemake 1985, "Animating
// Rotation with Quaternion Curves"; the default merge method of model-merging tools like
// mergekit). Flattening each tensor to a vector, with Ω the angle between them
// (cos Ω = (a·b)/(‖a‖‖b‖)),
//
//	SLERP(a,b,t) = sin((1−t)·Ω)/sin Ω · a + sin(t·Ω)/sin Ω · b
//
// interpolates along the great circle on the hypersphere at constant angular velocity, so
// (unlike linear interpolation, which sags toward the origin at the midpoint) it PRESERVES the
// norm when ‖a‖=‖b‖. t∈[0,1] goes from a to b (t=0→a, t=1→b). When the tensors are nearly
// parallel (|cos Ω| > SLERPDotThreshold) it falls back to linear interpolation (1−t)·a + t·b.
// A pure-f64 weight-merge utility (like TIESMerge), not differentiable; apply it per tensor.
func SLERP(a, b *tensor.Tensor, t float64) (*tensor.Tensor, error) {
	if !a.Shape().Equal(b.Shape()) {
		return nil, fmt.Errorf("nn: SLERP needs equal-shaped tensors, got %v and %v", a.Shape(), b.Shape())
	}
	n := a.Numel()
	var na, nb, dot float64
	for i := range n {
		c := tensor.Unravel(i, a.Shape())
		av, bv := a.AtF64(c...), b.AtF64(c...)
		na += av * av
		nb += bv * bv
		dot += av * bv
	}
	cos := 0.0
	if na > 0 && nb > 0 {
		cos = dot / (math.Sqrt(na) * math.Sqrt(nb))
	}
	cos = math.Max(-1, math.Min(1, cos))

	var s0, s1 float64
	if math.Abs(cos) > SLERPDotThreshold {
		s0, s1 = 1-t, t // near-parallel: linear interpolation
	} else {
		omega := math.Acos(cos)
		sinO := math.Sin(omega)
		s0 = math.Sin((1-t)*omega) / sinO
		s1 = math.Sin(t*omega) / sinO
	}
	out := tensor.New(a.Dtype(), a.Shape())
	for i := range n {
		c := tensor.Unravel(i, a.Shape())
		out.SetF64(s0*a.AtF64(c...)+s1*b.AtF64(c...), c...)
	}
	return out, nil
}
