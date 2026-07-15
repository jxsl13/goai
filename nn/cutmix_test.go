package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// img4 builds a distinctly-valued [B,C,H,W] tensor: value = 100·b + 10·c + h·W + w,
// so a pixel's origin sample/channel/location is readable from its value.
func img4(B, C, H, W int) *tensor.Tensor {
	x := tensor.New(tensor.F64, tensor.Shape{B, C, H, W})
	for b := range B {
		for c := range C {
			for h := range H {
				for w := range W {
					x.SetF64(float64(100*b+10*c+h*W+w), b, c, h, w)
				}
			}
		}
	}
	return x
}

// §V16 anchor 1 (COLLAPSE): CutMix λ=1 ⇒ zero-area box ⇒ x unchanged, corrected
// λ = 1 exactly.
func TestCutMixLambdaOneCollapse(t *testing.T) {
	c := nn.NewCutMix(1.0, 5, nn.WithCutMixLambda(1))
	x := img4(3, 2, 4, 4)
	out, _, lam, err := c.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	if lam != 1 {
		t.Fatalf("corrected lambda %g, want exactly 1 (zero-area box)", lam)
	}
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		if out.AtF64(idx...) != x.AtF64(idx...) {
			t.Fatalf("λ=1 not identity at %v", idx)
		}
	}
}

// §V16 anchor 2 (AREA→λ) + anchor 3 (REGION): with λ forced the box is
// deterministic; corrected λ = 1 − boxArea/(H·W) EXACT, x'==x outside the box and
// ==x[π] inside.
func TestCutMixRegionAndCorrectedLambda(t *testing.T) {
	const H, W = 8, 8
	c := nn.NewCutMix(1.0, 11, nn.WithCutMixLambda(0.5))
	x := img4(4, 3, H, W)
	out, perm, lam, err := c.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	// Recover the pasted box exactly: a pixel came from the partner iff it differs
	// from x, and the differing set must be a solid rectangle.
	y1, y2, x1, x2 := H, 0, W, 0
	pasted := 0
	for b := range 4 {
		for ch := range 3 {
			for h := range H {
				for w := range W {
					self := x.AtF64(b, ch, h, w)
					partner := x.AtF64(perm[b], ch, h, w)
					got := out.AtF64(b, ch, h, w)
					if got != self { // inside the box: must equal the partner pixel
						if got != partner {
							t.Fatalf("box pixel [%d,%d,%d,%d]=%v, want partner %v", b, ch, h, w, got, partner)
						}
						if b == 0 && ch == 0 {
							pasted++
							y1, y2 = min(y1, h), max(y2, h+1)
							x1, x2 = min(x1, w), max(x2, w+1)
						}
					} else if partner != self && got != self {
						t.Fatalf("outside-box pixel [%d,%d,%d,%d] changed", b, ch, h, w)
					}
				}
			}
		}
	}
	boxArea := pasted
	wantLam := 1 - float64(boxArea)/float64(H*W)
	if math.Abs(lam-wantLam) != 0 { // EXACT
		t.Errorf("corrected lambda %v, want %v (1 − boxArea/HW)", lam, wantLam)
	}
	// The changed set must be a filled rectangle: (y2−y1)·(x2−x1) == pasted count.
	if (y2-y1)*(x2-x1) != pasted {
		t.Errorf("pasted region not a solid rectangle: %d pixels vs %dx%d", pasted, y2-y1, x2-x1)
	}
}

// §V16 anchor 5: CutMix requires a rank-4 tensor.
func TestCutMixRankError(t *testing.T) {
	c := nn.NewCutMix(1.0, 1)
	if _, _, _, err := c.Forward(tensor.New(tensor.F64, tensor.Shape{4, 8})); err == nil {
		t.Error("rank-2 input: want error (needs [B,C,H,W])")
	}
	defer func() {
		if recover() == nil {
			t.Error("alpha≤0: want panic")
		}
	}()
	nn.NewCutMix(-1, 1)
}

// Determinism: same seed ⇒ identical box, permutation and corrected λ.
func TestCutMixDeterminism(t *testing.T) {
	x := img4(4, 2, 6, 6)
	a, pa, la, _ := nn.NewCutMix(1.0, 77).Forward(x)
	b, pb, lb, _ := nn.NewCutMix(1.0, 77).Forward(x)
	if la != lb {
		t.Fatalf("corrected lambda differs: %v vs %v", la, lb)
	}
	for i := range pa {
		if pa[i] != pb[i] {
			t.Fatalf("perm differs at %d", i)
		}
	}
	for i := range x.Numel() {
		idx := tensor.Unravel(i, x.Shape())
		if a.AtF64(idx...) != b.AtF64(idx...) {
			t.Fatalf("output differs at %v", idx)
		}
	}
}

// CutMix labels feed MixupLoss with the corrected λ exactly like Mixup.
func TestCutMixLabelPipeline(t *testing.T) {
	ctx := backend.NewContext()
	c := nn.NewCutMix(1.0, 9, nn.WithCutMixLambda(0.4))
	x := img4(5, 1, 4, 4)
	_, perm, lam, err := c.Forward(x)
	if err != nil {
		t.Fatal(err)
	}
	z := tensor.New(tensor.F64, tensor.Shape{5, 3})
	yA := tensor.New(tensor.F64, tensor.Shape{5})
	for i := range 5 {
		yA.SetF64(float64(i%3), i)
		for j := range 3 {
			z.SetF64(math.Sin(float64(i*3+j)), i, j)
		}
	}
	yB := nn.PermuteRows(yA, perm)
	loss, err := nn.MixupLoss(ctx, z, yA, yB, lam)
	if err != nil {
		t.Fatal(err)
	}
	ceA, _ := nn.CrossEntropy(ctx, z, yA)
	ceB, _ := nn.CrossEntropy(ctx, z, yB)
	want := lam*ceA.AtF64() + (1-lam)*ceB.AtF64()
	if math.Abs(loss.AtF64()-want) > 1e-10 {
		t.Errorf("CutMix loss=%v want %v", loss.AtF64(), want)
	}
}

func ExampleCutMix() {
	// α=1.0 is the paper default; the corrected λ reflects the actual box area.
	c := nn.NewCutMix(1.0, 3)
	x := img4(2, 3, 16, 16)
	_, perm, lambda, _ := c.Forward(x)
	frac := 1 - lambda // fraction of pixels taken from the partner image
	fmt.Printf("perm=%v boxfrac=%.4f in[0,1]=%v\n", perm, frac, frac >= 0 && frac <= 1)
	// Output:
	// perm=[1 0] boxfrac=0.3516 in[0,1]=true
}
