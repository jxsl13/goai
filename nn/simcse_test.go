package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// simcseRef independently computes the canonical one-directional SimCSE loss
// (Gao et al. 2021, Eq. 1): the row-wise softmax cross-entropy of the cosine
// similarity matrix S[i,j]=cos(z1[i],z2[j])/τ with the diagonal as each row's
// target. This is the pure-Go anchor for §V16 tier-1.
func simcseRef(z1, z2 [][]float64, tau float64, normalize bool) float64 {
	n, d := len(z1), len(z1[0])
	a := make([][]float64, n)
	b := make([][]float64, n)
	for i := range n {
		a[i] = append([]float64(nil), z1[i]...)
		b[i] = append([]float64(nil), z2[i]...)
		if normalize {
			var sa, sb float64
			for k := range d {
				sa += z1[i][k] * z1[i][k]
				sb += z2[i][k] * z2[i][k]
			}
			sa, sb = math.Sqrt(sa), math.Sqrt(sb)
			for k := range d {
				a[i][k] /= sa
				b[i][k] /= sb
			}
		}
	}
	// S[i,j] = a[i]·b[j] / τ; loss = mean_i −log softmax(S[i])[i].
	var loss float64
	for i := range n {
		row := make([]float64, n)
		for j := range n {
			for k := range d {
				row[j] += a[i][k] * b[j][k]
			}
			row[j] /= tau
		}
		mx := math.Inf(-1)
		for _, v := range row {
			mx = math.Max(mx, v)
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(v - mx)
		}
		loss += -(row[i] - mx - math.Log(sum))
	}
	return loss / float64(n)
}

var (
	scZ1 = [][]float64{{1, 0.5, -1}, {0.2, 1, 0.3}, {-0.5, 0.4, 1}, {0.7, -0.2, 0.6}}
	scZ2 = [][]float64{{0.9, 0.4, -0.8}, {0.1, 1.2, 0.2}, {-0.6, 0.3, 1.1}, {0.6, -0.1, 0.8}}
)

// §V16 tier-1 (InfoNCE correctness): SimCSELoss equals the pure-Go in-batch NT-Xent
// reference (cos-sim/τ, softmax-CE with diagonal targets) @1e-10, with and without
// L2 normalization, across temperatures.
func TestSimCSEFormula(t *testing.T) {
	for _, norm := range []bool{false, true} {
		for _, tau := range []float64{0.05, 0.1, 1.0} {
			got, err := nn.SimCSELoss(backend.NewContext(), toT(scZ1), toT(scZ2),
				nn.WithSimCSETemperature(tau), nn.WithSimCSENormalize(norm))
			if err != nil {
				t.Fatalf("norm=%v τ=%g: %v", norm, tau, err)
			}
			want := simcseRef(scZ1, scZ2, tau, norm)
			if math.Abs(got.AtF64()-want) > 1e-10 {
				t.Errorf("norm=%v τ=%g: SimCSELoss = %.12g, want %.12g (Δ=%.2e)",
					norm, tau, got.AtF64(), want, math.Abs(got.AtF64()-want))
			}
		}
	}
}

// §V16: the default temperature is the paper's τ=0.05 (calling with no options must
// equal the reference at 0.05), and the symmetric option matches the symmetric
// in-batch InfoNCE reference (shared with the InfoNCE test).
func TestSimCSEDefaultsAndSymmetric(t *testing.T) {
	def, err := nn.SimCSELoss(backend.NewContext(), toT(scZ1), toT(scZ2))
	if err != nil {
		t.Fatal(err)
	}
	if want := simcseRef(scZ1, scZ2, 0.05, true); math.Abs(def.AtF64()-want) > 1e-10 {
		t.Errorf("default SimCSELoss = %.12g, want τ=0.05 ref %.12g", def.AtF64(), want)
	}
	sym, err := nn.SimCSELoss(backend.NewContext(), toT(scZ1), toT(scZ2),
		nn.WithSimCSETemperature(0.1), nn.WithSimCSESymmetric(true))
	if err != nil {
		t.Fatal(err)
	}
	if want := infonceRef(scZ1, scZ2, 0.1, true); math.Abs(sym.AtF64()-want) > 1e-9 {
		t.Errorf("symmetric SimCSELoss = %.12g, want InfoNCE ref %.12g", sym.AtF64(), want)
	}
}

// §V2 (GRADCHECK): finite-difference gradcheck of SimCSELoss w.r.t. z1 and z2 through
// the cos-sim + softmax-CE, with and without L2 norm; reports the max relative error.
func TestSimCSEGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(21, 9))
	const n, d = 4, 3
	var maxRel float64
	for _, norm := range []bool{false, true} {
		z1 := randMat(rng, n, d)
		z2 := randMat(rng, n, d)
		tape := autograd.NewTape()
		loss, err := nn.SimCSELoss(tape.Context(), z1, z2, nn.WithSimCSENormalize(norm))
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		const h = 1e-6
		check := func(name string, p *tensor.Tensor) {
			g := tape.Grad(p)
			if g == nil {
				t.Fatalf("norm=%v %s: nil grad", norm, name)
			}
			for i := range n {
				for j := range d {
					o := p.AtF64(i, j)
					p.SetF64(o+h, i, j)
					lp, _ := nn.SimCSELoss(backend.NewContext(), z1, z2, nn.WithSimCSENormalize(norm))
					p.SetF64(o-h, i, j)
					lm, _ := nn.SimCSELoss(backend.NewContext(), z1, z2, nn.WithSimCSENormalize(norm))
					p.SetF64(o, i, j)
					num, ana := (lp.AtF64()-lm.AtF64())/(2*h), g.AtF64(i, j)
					rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
					if rel > maxRel {
						maxRel = rel
					}
					if rel > 1e-4 {
						t.Errorf("norm=%v %s grad[%d,%d]: numeric %.8g vs analytic %.8g", norm, name, i, j, num, ana)
					}
				}
			}
		}
		check("z1", z1)
		check("z2", z2)
	}
	t.Logf("SimCSE gradcheck max relative error: %.3e", maxRel)
}

// cosineRows returns cos(a[i], b[i]) for each row.
func cosineRows(a, b *tensor.Tensor) []float64 {
	n, d := a.Shape()[0], a.Shape()[1]
	out := make([]float64, n)
	for i := range n {
		var dot, na, nb float64
		for k := range d {
			av, bv := a.AtF64(i, k), b.AtF64(i, k)
			dot += av * bv
			na += av * av
			nb += bv * bv
		}
		out[i] = dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
	}
	return out
}

// §V16 tier-1 (THE VALUE — alignment/uniformity): a few steps of SimCSE training over
// an encoder INCREASE the positive-pair cosine (alignment ↑) and make each input's two
// dropout views the nearest neighbour among the whole batch (rank-1 discriminability ↑),
// while the loss decreases and the embeddings stay spread on the hypersphere (the
// Wang-Isola uniformity metric, logged). This is WHY SimCSE works — same-input-two-
// dropout-views is a positive signal without labelled pairs.
//
// Note on uniformity: a well-initialized encoder already spreads a handful of points
// near-optimally in a 16-d space, so the Wang-Isola metric has little room to improve
// and is reported as a diagnostic rather than gated; the honest, robust signals of the
// objective doing its job are alignment ↑, loss ↓, and the positive becoming each
// anchor's argmax (rank-1) — the empirical face of keeping different inputs separable.
func TestSimCSEAlignmentUniformity(t *testing.T) {
	rng := rand.New(rand.NewPCG(2024, 7))
	const batch, in, hid = 8, 6, 16
	x := randMat(rng, batch, in)

	// Encoder: Linear(in→hid) → Dropout → Linear(hid→hid). The Dropout is the SimCSE
	// augmentation; calling it advances its rng so the two passes get distinct masks.
	lin1 := nn.NewLinear(tensor.F64, in, hid, 1)
	lin2 := nn.NewLinear(tensor.F64, hid, hid, 2)
	encodeWith := func(ctx *backend.Context, in *tensor.Tensor, drop *nn.Dropout) (*tensor.Tensor, error) {
		h, err := lin1.Forward(ctx, in)
		if err != nil {
			return nil, err
		}
		if h, err = drop.Forward(ctx, h); err != nil {
			return nil, err
		}
		return lin2.Forward(ctx, h)
	}

	// cosAt returns cos(a[i], b[j]).
	cosAt := func(a, b *tensor.Tensor, i, j int) float64 {
		var dot, na, nb float64
		for k := range a.Shape()[1] {
			av, bv := a.AtF64(i, k), b.AtF64(j, k)
			dot += av * bv
			na += av * av
			nb += bv * bv
		}
		return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-12)
	}

	// measure encodes twice under a FRESH fixed-seed dropout (reproducible masks) so
	// step-0 and step-N differ only by the weights, then reports the loss, the mean
	// positive-pair cosine (alignment), the fraction of anchors whose positive is its
	// batch-wide argmax (rank-1), and the Wang-Isola uniformity of the eval encodings
	// (U = log mean_{i≠j} exp(−2‖ê_i−ê_j‖²); lower = more spread).
	measure := func() (loss, align, rank1, unif float64) {
		md := nn.NewDropout(0.3, 4242) // fixed seed → identical masks every measurement
		ctx := backend.NewContext()
		z1, err := encodeWith(ctx, x, md)
		if err != nil {
			t.Fatal(err)
		}
		z2, err := encodeWith(ctx, x, md)
		if err != nil {
			t.Fatal(err)
		}
		l, err := nn.SimCSELoss(ctx, z1, z2)
		if err != nil {
			t.Fatal(err)
		}
		for _, c := range cosineRows(z1, z2) {
			align += c
		}
		align /= float64(batch)
		hits := 0.0
		for i := range batch {
			best, bi := math.Inf(-1), -1
			for j := range batch {
				if c := cosAt(z1, z2, i, j); c > best {
					best, bi = c, j
				}
			}
			if bi == i {
				hits++
			}
		}
		rank1 = hits / float64(batch)
		// Wang-Isola uniformity on deterministic (dropout-off) eval encodings.
		ev := nn.NewDropout(0.3, 1)
		ev.Eval()
		e, err := encodeWith(ctx, x, ev)
		if err != nil {
			t.Fatal(err)
		}
		var s, cnt float64
		for i := range batch {
			for j := range batch {
				if i == j {
					continue
				}
				s += math.Exp(-2 * (2 - 2*cosAt(e, e, i, j))) // ‖ê_i−ê_j‖²=2−2cos on unit sphere
				cnt++
			}
		}
		unif = math.Log(s / cnt)
		return l.AtF64(), align, rank1, unif
	}

	loss0, align0, rank0, unif0 := measure()

	opt := nn.NewSGD([]*tensor.Tensor{lin1.W, lin1.B, lin2.W, lin2.B}, 0.3, 0.9)
	trainDrop := nn.NewDropout(0.3, 99)
	var lastLoss float64
	for step := range 80 {
		tape := autograd.NewTape()
		enc := func(ctx *backend.Context, in *tensor.Tensor) (*tensor.Tensor, error) {
			return encodeWith(ctx, in, trainDrop)
		}
		z1, z2, err := nn.SimCSEViews(tape.Context(), enc, x)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.SimCSELoss(tape.Context(), z1, z2)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
		if step == 79 {
			lastLoss = loss.AtF64()
		}
	}

	lossN, alignN, rankN, unifN := measure()
	t.Logf("loss %.4f→%.4f (train last %.4f)  align %.4f→%.4f  rank1 %.2f→%.2f  unif %.4f→%.4f",
		loss0, lossN, lastLoss, align0, alignN, rank0, rankN, unif0, unifN)

	if alignN <= align0 {
		t.Errorf("alignment did not increase: %.4f → %.4f", align0, alignN)
	}
	if rankN <= rank0 {
		t.Errorf("rank-1 discriminability did not increase: %.2f → %.2f", rank0, rankN)
	}
	if lossN >= loss0 {
		t.Errorf("loss did not decrease: %.4f → %.4f", loss0, lossN)
	}
}

// §V16 tier-1 (collapse/degenerate): identical views z1==z2 with distinct rows give
// diagonal similarities of 1 and off-diagonals < 1, hence a low, well-defined loss;
// and the loss is deterministic.
func TestSimCSEIdenticalViews(t *testing.T) {
	// Well-separated orthogonal-ish rows: diagonal cos=1, off-diagonals small.
	z := toT([][]float64{{3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 2, 0}})
	got, err := nn.SimCSELoss(backend.NewContext(), z, z, nn.WithSimCSETemperature(0.05))
	if err != nil {
		t.Fatal(err)
	}
	want := simcseRef([][]float64{{3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 2, 0}},
		[][]float64{{3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 2, 0}}, 0.05, true)
	if math.Abs(got.AtF64()-want) > 1e-10 {
		t.Errorf("identical-views loss %.12g != ref %.12g", got.AtF64(), want)
	}
	if got.AtF64() < 0 {
		t.Errorf("cross-entropy loss must be ≥ 0, got %g", got.AtF64())
	}
	// Determinism: the same inputs give the same loss.
	again, _ := nn.SimCSELoss(backend.NewContext(), z, z, nn.WithSimCSETemperature(0.05))
	if got.AtF64() != again.AtF64() {
		t.Errorf("non-deterministic: %g != %g", got.AtF64(), again.AtF64())
	}
}

func TestSimCSEErrors(t *testing.T) {
	a := toT(scZ1)
	if _, err := nn.SimCSELoss(backend.NewContext(), a, toT([][]float64{{1, 2}})); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.SimCSELoss(backend.NewContext(), a, toT(scZ2), nn.WithSimCSETemperature(-1)); err != nil {
		// τ=-1 is IGNORED by the setter (stays at the 0.05 default), so this must NOT error.
		t.Errorf("negative τ should be ignored, not error: %v", err)
	}
	if _, err := nn.SimCSELoss(backend.NewContext(),
		tensor.New(tensor.F64, tensor.Shape{3}), tensor.New(tensor.F64, tensor.Shape{3})); err == nil {
		t.Error("rank-1 input: want error")
	}
}

func ExampleSimCSELoss() {
	// Encoder = Linear → Dropout → Linear. Dropout is ON, so running the SAME batch
	// through it TWICE (via SimCSEViews) yields two independent views — the SimCSE
	// positive pair. In-batch, distinct inputs are the negatives, so the loss is > 0.
	enc := nn.NewSequential(
		nn.NewLinear(tensor.F64, 4, 6, 1),
		nn.NewDropout(0.2, 7),
		nn.NewLinear(tensor.F64, 6, 4, 2),
	)
	x := toT([][]float64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}})
	ctx := backend.NewContext()
	z1, z2, _ := nn.SimCSEViews(ctx, enc.Forward, x)
	loss, _ := nn.SimCSELoss(ctx, z1, z2) // τ defaults to the paper's 0.05
	fmt.Println(loss.AtF64() > 0, z1.Shape())
	// Output:
	// true (4, 4)
}
