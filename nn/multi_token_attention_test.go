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

// --- independent standard multi-head attention reference (delta-collapse oracle) ---

func mtaLinear(x [][]float64, w, b *tensor.Tensor) [][]float64 {
	in, out := w.Shape()[0], w.Shape()[1]
	y := make([][]float64, len(x))
	for r := range x {
		y[r] = make([]float64, out)
		for o := range out {
			s := 0.0
			if b != nil {
				s = b.AtF64(o)
			}
			for i := range in {
				s += x[r][i] * w.AtF64(i, o)
			}
			y[r][o] = s
		}
	}
	return y
}

// mtaStdAttnRef computes ordinary multi-head softmax attention from x using the
// layer's OWN projection weights — the oracle a delta-initialized MTA must equal.
func mtaStdAttnRef(x [][]float64, m *nn.MultiTokenAttention, causal bool) [][]float64 {
	q := mtaLinear(x, m.QProj.W, m.QProj.B)
	k := mtaLinear(x, m.KProj.W, m.KProj.B)
	v := mtaLinear(x, m.VProj.W, m.VProj.B)
	tq, hd := len(x), m.HeadDim
	scale := 1 / math.Sqrt(float64(hd))
	concat := make([][]float64, tq)
	for r := range concat {
		concat[r] = make([]float64, m.Dim)
	}
	for h := range m.Heads {
		off := h * hd
		for i := range tq {
			logit := make([]float64, tq)
			mx := math.Inf(-1)
			for j := range tq {
				if causal && j > i {
					logit[j] = math.Inf(-1)
					continue
				}
				var dp float64
				for c := range hd {
					dp += q[i][off+c] * k[j][off+c]
				}
				logit[j] = dp * scale
				if logit[j] > mx {
					mx = logit[j]
				}
			}
			var sum float64
			wgt := make([]float64, tq)
			for j := range tq {
				wgt[j] = math.Exp(logit[j] - mx)
				sum += wgt[j]
			}
			for c := range hd {
				var acc float64
				for j := range tq {
					acc += (wgt[j] / sum) * v[j][off+c]
				}
				concat[i][off+c] = acc
			}
		}
	}
	return mtaLinear(concat, m.OProj.W, m.OProj.B)
}

// §V16 anchor 1 (COLLAPSE): a freshly built MTA (delta key-query kernel, identity
// head kernel, no extra norm) is BIT-EXACT ordinary multi-head softmax attention,
// causal and bidirectional, at 1e-10. Pins "MTA adds only the convolution mixing".
func TestMultiTokenAttentionDeltaCollapse(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	const T, d, h = 6, 8, 2
	x := randMat(rng, T, d)
	xr := toRows(x)
	for _, causal := range []bool{false, true} {
		opts := []nn.MultiTokenAttentionOption{nn.WithMTAKeyQueryKernel(6, 11), nn.WithMTAHeadKernel(1)}
		if !causal {
			opts = append(opts, nn.WithMTABidirectional())
		}
		m, err := nn.NewMultiTokenAttention(tensor.F64, d, h, 7, opts...)
		if err != nil {
			t.Fatal(err)
		}
		got, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatalf("causal=%v: %v", causal, err)
		}
		want := mtaStdAttnRef(xr, m, causal)
		for i := range T {
			for j := range d {
				if g := got.AtF64(i, j); math.Abs(g-want[i][j]) > 1e-10 {
					t.Errorf("causal=%v out[%d,%d]=%.12g want %.12g", causal, i, j, g, want[i][j])
				}
			}
		}
	}
}

// §V16 anchor 2 (CAUSALITY): with causal masking and ACTIVE (non-delta) key-query
// and head kernels, the output at query t is independent of any key/value at
// position > t — perturbing the input token at position p leaves every output row
// i < p bit-identical (< 1e-12). Proves the convolutions never leak future keys.
func TestMultiTokenAttentionCausality(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 13))
	const T, d, h = 7, 8, 2
	m, err := nn.NewMultiTokenAttention(tensor.F64, d, h, 3,
		nn.WithMTAKeyQueryKernel(2, 3), nn.WithMTAHeadKernel(2))
	if err != nil {
		t.Fatal(err)
	}
	// Make both convolutions active (non-identity) so the test actually exercises
	// the kernel windows rather than the collapse special case.
	for idx := range m.KQKernel.Numel() {
		m.KQKernel.SetF64(rng.NormFloat64()*0.5, tensor.Unravel(idx, m.KQKernel.Shape())...)
	}
	for idx := range m.HeadKernel.Numel() {
		m.HeadKernel.SetF64(rng.NormFloat64()*0.5, tensor.Unravel(idx, m.HeadKernel.Shape())...)
	}
	x := randMat(rng, T, d)
	base, err := m.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	const p = 4 // perturb token p; rows 0..p-1 must be unaffected
	x2 := randMat(rng, T, d)
	// x2 equals x everywhere except row p (fully re-randomized there).
	for i := range T {
		if i == p {
			continue
		}
		for j := range d {
			x2.SetF64(x.AtF64(i, j), i, j)
		}
	}
	pert, err := m.Forward(backend.NewContext(), x2)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < p; i++ {
		for j := range d {
			if g, b := pert.AtF64(i, j), base.AtF64(i, j); math.Abs(g-b) > 1e-12 {
				t.Errorf("row %d changed by perturbing token %d: [%d,%d] %.12g vs %.12g", i, p, i, j, g, b)
			}
		}
	}
	// Sanity: row p ITSELF must change (else the perturbation was a no-op).
	changed := false
	for j := range d {
		if math.Abs(pert.AtF64(p, j)-base.AtF64(p, j)) > 1e-9 {
			changed = true
		}
	}
	if !changed {
		t.Error("perturbing token p left row p unchanged — test is vacuous")
	}
}

// §V16 anchor 4 (GRADCHECK): finite-difference gradient check through the whole
// layer INCLUDING both convolution kernels and the Q/K/V/O projections.
func TestMultiTokenAttentionGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 19))
	const T, d, h = 4, 4, 2
	m, err := nn.NewMultiTokenAttention(tensor.F64, d, h, 5,
		nn.WithMTAKeyQueryKernel(2, 3), nn.WithMTAHeadKernel(2))
	if err != nil {
		t.Fatal(err)
	}
	// Perturb the kernels off delta/identity so their gradients are non-degenerate.
	for idx := range m.KQKernel.Numel() {
		c := tensor.Unravel(idx, m.KQKernel.Shape())
		m.KQKernel.SetF64(m.KQKernel.AtF64(c...)+rng.NormFloat64()*0.2, c...)
	}
	for idx := range m.HeadKernel.Numel() {
		c := tensor.Unravel(idx, m.HeadKernel.Shape())
		m.HeadKernel.SetF64(m.HeadKernel.AtF64(c...)+rng.NormFloat64()*0.2, c...)
	}
	x := randMat(rng, T, d)
	tape := autograd.NewTape()
	out, err := m.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const step = 1e-6
	fwd := func() float64 {
		o, _ := m.Forward(backend.NewContext(), x)
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+step, coord...)
			lp := fwd()
			p.SetF64(o-step, coord...)
			lm := fwd()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*step), g.AtF64(coord...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
	check("Wq", m.QProj.W)
	check("bq", m.QProj.B)
	check("Wk", m.KProj.W)
	check("Wv", m.VProj.W)
	check("Wo", m.OProj.W)
	check("bo", m.OProj.B)
	check("KQKernel", m.KQKernel)
	check("HeadKernel", m.HeadKernel)
}

// mtaColors is the payload alphabet the retrieval toy must read back.
const mtaColors = 4

// mtaSample builds one adjacent-bigram-needle example (see TestMTATwoTokenRetrieval).
// Layout of the d=16 channels: 0=feature A, 1=feature B, 2..2+mtaColors−1 = a
// random per-position "color" one-hot (the PAYLOAD), channel 6 = probe marker,
// rest 0. Positions 0..4 are content; position 5 is a fixed, content-free PROBE
// token (the query source). Feature A appears at EXACTLY ONE content position t*−1
// and feature B at t* AND at ≥2 distractor positions — so "has B" alone cannot
// pick out t*; only "B whose predecessor is A" does. The target is the COLOR of
// t*: it is NOT recoverable by pooling the B positions (their colors are i.i.d.),
// so a model must concentrate attention on t* itself to read it. Returns x[T,d]
// and the target color.
func mtaSample(rng *rand.Rand) (*tensor.Tensor, int) {
	const T, d = 6, 16
	tstar := 1 + rng.IntN(4) // t* in [1,4]; A at t*−1, B at t*
	rest := []int{}          // content positions other than the A→B pair
	for j := range 5 {
		if j != tstar-1 && j != tstar {
			rest = append(rest, j)
		}
	}
	// Place ≥2 distractor B's among `rest` (never after the lone A at t*−1, so no
	// second adjacent A→B can form). A appears ONLY at t*−1.
	rng.Shuffle(len(rest), func(a, b int) { rest[a], rest[b] = rest[b], rest[a] })
	isB := map[int]bool{tstar: true}
	nB := 2 + rng.IntN(len(rest)-1) // 2..len(rest) distractor B's
	for i := 0; i < nB && i < len(rest); i++ {
		isB[rest[i]] = true
	}
	x := tensor.New(tensor.F64, tensor.Shape{T, d})
	var target int
	for j := range 5 {
		if j == tstar-1 {
			x.SetF64(1, j, 0) // feature A
		}
		if isB[j] {
			x.SetF64(1, j, 1) // feature B
		}
		color := rng.IntN(mtaColors)
		x.SetF64(1, j, 2+color) // payload color one-hot
		if j == tstar {
			target = color
		}
	}
	x.SetF64(1, T-1, 6) // probe token marker
	return x, target
}

// mtaEvalAcc returns argmax-color accuracy of readout(MTA(x)[probe]) over n fresh
// samples. readout maps the probe row (d) to mtaColors logits.
func mtaEvalAcc(m *nn.MultiTokenAttention, readout *nn.Linear, rng *rand.Rand, n int) float64 {
	const T = 6
	correct := 0
	for range n {
		x, target := mtaSample(rng)
		out, err := m.Forward(backend.NewContext(), x)
		if err != nil {
			return -1
		}
		probe, _ := out.Slice(0, T-1, T) // [1,d]
		probe = probe.Contiguous()
		logits, err := readout.Forward(backend.NewContext(), probe) // [1,mtaColors]
		if err != nil {
			return -1
		}
		best, bi := math.Inf(-1), 0
		for c := range mtaColors {
			if v := logits.AtF64(0, c); v > best {
				best, bi = v, c
			}
		}
		if bi == target {
			correct++
		}
	}
	return float64(correct) / float64(n)
}

// §V16 anchor 5 (THE VALUE — the paper's headline): the two-token retrieval toy.
// The model must localize the unique position where feature A and feature B occur
// in ADJACENT tokens — a conjunction across two positions. A single query·key dot
// product scores each key independently, so ordinary attention (delta kernel,
// frozen) provably cannot separate the true A→B position from distractor B-tokens;
// MTA's key-query convolution (shift A-evidence one key) plus head convolution
// (combine the two heads' evidence BEFORE the softmax) can. We train both and
// report: MTA solves it, standard attention stays near chance.
func TestMultiTokenAttentionTwoTokenRetrieval(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping trained two-token-retrieval toy in -short")
	}
	const T, d, h, batch, steps = 6, 16, 2, 32, 700
	train := func(mta bool) (float64, float64) {
		seed := uint64(101)
		var m *nn.MultiTokenAttention
		var err error
		if mta {
			m, err = nn.NewMultiTokenAttention(tensor.F64, d, h, seed,
				nn.WithMTAKeyQueryKernel(2, 3), nn.WithMTAHeadKernel(2), nn.WithMTABidirectional())
		} else { // standard attention: delta kernel, no head conv (cq=ck=ch=1)
			m, err = nn.NewMultiTokenAttention(tensor.F64, d, h, seed,
				nn.WithMTAKeyQueryKernel(1, 1), nn.WithMTAHeadKernel(1), nn.WithMTABidirectional())
		}
		if err != nil {
			t.Fatal(err)
		}
		readout := nn.NewLinear(tensor.F64, d, mtaColors, seed+50)
		params := append(m.Params(), readout.Params()...)
		opt := nn.NewAdam(params, 0.01)
		dataRng := rand.New(rand.NewPCG(7, 99))
		var lastLoss float64
		for s := range steps {
			tape := autograd.NewTape()
			// Build a batch: run each sample, collect probe logits, one CE loss.
			logitRows := make([]*tensor.Tensor, batch)
			targets := tensor.New(tensor.F64, tensor.Shape{batch})
			for b := range batch {
				x, tstar := mtaSample(dataRng)
				out, err := m.Forward(tape.Context(), x)
				if err != nil {
					t.Fatal(err)
				}
				probe, _ := backend.Execute(tape.Context(), backend.OpSlice,
					[]*tensor.Tensor{out}, backend.SliceAttrs{Axis: 0, Start: T - 1, End: T})
				lg, err := readout.Forward(tape.Context(), probe[0]) // [1,T]
				if err != nil {
					t.Fatal(err)
				}
				logitRows[b] = lg
				targets.SetF64(float64(tstar), b)
			}
			logits, err := backend.Execute(tape.Context(), backend.OpConcat, logitRows, backend.ConcatAttrs{Axis: 0})
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(tape.Context(), logits[0], targets)
			if err != nil {
				t.Fatal(err)
			}
			if err := tape.Backward(loss); err != nil {
				t.Fatal(err)
			}
			if err := opt.Step(tape.Grad); err != nil {
				t.Fatal(err)
			}
			lastLoss = loss.AtF64()
			_ = s
		}
		evalRng := rand.New(rand.NewPCG(555, 12))
		return mtaEvalAcc(m, readout, evalRng, 300), lastLoss
	}

	mtaAcc, mtaLoss := train(true)
	stdAcc, stdLoss := train(false)
	t.Logf("MTA:      acc=%.3f loss=%.4f", mtaAcc, mtaLoss)
	t.Logf("standard: acc=%.3f loss=%.4f", stdAcc, stdLoss)
	if mtaAcc < 0.85 {
		t.Errorf("MTA failed the two-token toy: acc=%.3f (< 0.85)", mtaAcc)
	}
	if mtaAcc < stdAcc+0.25 {
		t.Errorf("MTA (%.3f) did not clearly beat standard attention (%.3f)", mtaAcc, stdAcc)
	}
}

// §V16 anchor 6 (sanity): constructor validation, determinism, shape errors.
func TestMultiTokenAttentionErrors(t *testing.T) {
	if _, err := nn.NewMultiTokenAttention(tensor.F64, 0, 2, 1); err == nil {
		t.Error("dim 0: want error")
	}
	if _, err := nn.NewMultiTokenAttention(tensor.F64, 6, 4, 1); err == nil {
		t.Error("heads not dividing dim: want error")
	}
	if _, err := nn.NewMultiTokenAttention(tensor.F64, 8, 2, 1, nn.WithMTAKeyQueryKernel(0, 3)); err == nil {
		t.Error("cq<1: want error")
	}
	if _, err := nn.NewMultiTokenAttention(tensor.F64, 8, 2, 1, nn.WithMTAKeyQueryKernel(2, 0)); err == nil {
		t.Error("ck<1: want error")
	}
	if _, err := nn.NewMultiTokenAttention(tensor.F64, 8, 4, 1, nn.WithMTAHeadKernel(3)); err == nil {
		t.Error("ch not dividing heads: want error")
	}
	m, err := nn.NewMultiTokenAttention(tensor.F64, 8, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	bad := tensor.New(tensor.F64, tensor.Shape{4, 7}) // wrong width
	if _, err := m.Forward(backend.NewContext(), bad); err == nil {
		t.Error("wrong input width: want error")
	}
	r3 := tensor.New(tensor.F64, tensor.Shape{2, 8, 1})
	if _, err := m.Forward(backend.NewContext(), r3); err == nil {
		t.Error("rank-3 input: want error")
	}
	// Determinism: same seed ⇒ identical parameters and identical forward.
	x := tensor.New(tensor.F64, tensor.Shape{5, 8})
	for i := range x.Numel() {
		x.SetF64(math.Sin(float64(i)), tensor.Unravel(i, x.Shape())...)
	}
	m2, _ := nn.NewMultiTokenAttention(tensor.F64, 8, 2, 1)
	o1, _ := m.Forward(backend.NewContext(), x)
	o2, _ := m2.Forward(backend.NewContext(), x)
	for i := range o1.Numel() {
		c := tensor.Unravel(i, o1.Shape())
		if o1.AtF64(c...) != o2.AtF64(c...) {
			t.Errorf("non-deterministic at %v", c)
		}
	}
}

func ExampleMultiTokenAttention() {
	// A freshly built MTA is delta/identity-initialized, so it equals ordinary
	// multi-head softmax attention until trained (paper §4.6). Here one query row
	// aligned with key 0 reads value 0 back.
	x := toT([][]float64{{2, 0, 0, 0}, {0, 0, 1, 0}})
	m, _ := nn.NewMultiTokenAttention(tensor.F64, 4, 2, 1,
		nn.WithMTAKeyQueryKernel(3, 3), nn.WithMTABidirectional())
	out, _ := m.Forward(backend.NewContext(), x)
	fmt.Printf("%d params, out shape %v\n", len(m.Params()), out.Shape())
	// Output:
	// 10 params, out shape (2, 4)
}
