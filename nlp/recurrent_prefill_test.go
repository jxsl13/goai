package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Batched prefill for the RECURRENT families (Mamba, Mamba-2, Jamba, RWKV):
// Prefill(prompt) must leave the decode state BIT-EQUAL to stepping the prompt
// token-by-token through DecodeStep — the projections/conv/norms batch into
// full-sequence kernels, the recurrences replay the same host loops, so there
// is zero tolerance anywhere in these tests.

// bitsEqualF64 reports whether two f64 slices are bit-identical.
func bitsEqualF64(t *testing.T, name string, a, b []float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: length %d vs %d", name, len(a), len(b))
	}
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			t.Fatalf("%s[%d]: %v (bits %x) != %v (bits %x)", name, i, a[i], math.Float64bits(a[i]), b[i], math.Float64bits(b[i]))
		}
	}
}

// bitsEqualTensor reports whether two tensors are bit-identical (shape, dtype
// and every element's f64 reading).
func bitsEqualTensor(t *testing.T, name string, a, b *tensor.Tensor) {
	t.Helper()
	if (a == nil) != (b == nil) {
		t.Fatalf("%s: nil mismatch (%v vs %v)", name, a == nil, b == nil)
	}
	if a == nil {
		return
	}
	if !a.Shape().Equal(b.Shape()) {
		t.Fatalf("%s: shape %v vs %v", name, a.Shape(), b.Shape())
	}
	if a.Dtype() != b.Dtype() {
		t.Fatalf("%s: dtype %v vs %v", name, a.Dtype(), b.Dtype())
	}
	r, c := a.Shape()[0], a.Shape()[1]
	for i := range r {
		for j := range c {
			av, bv := a.AtF64(i, j), b.AtF64(i, j)
			if math.Float64bits(av) != math.Float64bits(bv) {
				t.Fatalf("%s[%d,%d]: %v != %v", name, i, j, av, bv)
			}
		}
	}
}

// prefillPrompt is shared with the attention-arch prefill parity tests
// (prefill_archs_test.go): []int{3, 7, 1, 9, 4, 2, 8}.

const prefillContinueTok = 5

// §V16 tier-1: Mamba batched prefill — state parity (conv window + SSM hidden
// state bit-equal to len(prompt) DecodeSteps), continuation parity (one
// DecodeStep from either state yields bit-identical logits), and last-row
// prefill logits bit-equal to the stepwise next-token logits.
func TestMambaPrefillStateParity(t *testing.T) {
	m := mambaHF(t)
	ctx := backend.NewContext()

	stStep := m.NewDecodeState()
	var stepLogits *tensor.Tensor
	var err error
	for _, tok := range prefillPrompt {
		if stepLogits, err = m.DecodeStep(ctx, stStep, tok); err != nil {
			t.Fatal(err)
		}
	}

	stPre := m.NewDecodeState()
	full, err := m.Prefill(ctx, stPre, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	last, err := full.Slice(0, full.Shape()[0]-1, full.Shape()[0])
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "prefill last-row logits", last.Contiguous(), stepLogits)

	for l := range stStep.Layers {
		bitsEqualF64(t, "layer ConvBuf", stPre.Layers[l].ConvBuf, stStep.Layers[l].ConvBuf)
		bitsEqualF64(t, "layer H", stPre.Layers[l].H, stStep.Layers[l].H)
	}

	// Continuation: one more DecodeStep from each state.
	l1, err := m.DecodeStep(ctx, stStep, prefillContinueTok)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := m.DecodeStep(ctx, stPre, prefillContinueTok)
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "continuation logits", l2, l1)

	// A used state must be rejected — the batched conv assumes sequence start.
	if _, err := m.Prefill(ctx, stPre, prefillPrompt); err == nil {
		t.Fatal("Prefill accepted a non-fresh decode state")
	}
}

// §V16 tier-1: Mamba-2 batched prefill — state, continuation and last-row
// logits parity, all bit-exact (the host-f64 mixer replays the same loops).
func TestMamba2PrefillStateParity(t *testing.T) {
	m := mamba2HF(t)
	ctx := backend.NewContext()

	stStep := m.NewDecodeState()
	var stepLogits *tensor.Tensor
	var err error
	for _, tok := range prefillPrompt {
		if stepLogits, err = m.DecodeStep(ctx, stStep, tok); err != nil {
			t.Fatal(err)
		}
	}

	stPre := m.NewDecodeState()
	full, err := m.Prefill(ctx, stPre, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	last, err := full.Slice(0, full.Shape()[0]-1, full.Shape()[0])
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "prefill last-row logits", last.Contiguous(), stepLogits)

	for l := range stStep.Layers {
		bitsEqualF64(t, "layer ConvBuf", stPre.Layers[l].ConvBuf, stStep.Layers[l].ConvBuf)
		bitsEqualF64(t, "layer H", stPre.Layers[l].H, stStep.Layers[l].H)
	}

	l1, err := m.DecodeStep(ctx, stStep, prefillContinueTok)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := m.DecodeStep(ctx, stPre, prefillContinueTok)
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "continuation logits", l2, l1)

	if _, err := m.Prefill(ctx, stPre, prefillPrompt); err == nil {
		t.Fatal("Prefill accepted a non-fresh decode state")
	}
}

// §V16 tier-1: Jamba hybrid batched prefill — the attention layers' KV rows,
// the Mamba layers' conv/SSM state, the continuation logits and the last-row
// prefill logits are all bit-equal to the stepwise path. The golden stack
// interleaves all four mixer/FFN families, so one parity check covers every
// combination.
func TestJambaPrefillStateParity(t *testing.T) {
	m := jambaGolden(t)
	ctx := backend.NewContext()

	stStep := m.NewDecodeState()
	var stepLogits *tensor.Tensor
	var err error
	for _, tok := range prefillPrompt {
		if stepLogits, err = m.DecodeStep(ctx, stStep, tok); err != nil {
			t.Fatal(err)
		}
	}

	stPre := m.NewDecodeState()
	full, err := m.Prefill(ctx, stPre, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	last, err := full.Slice(0, full.Shape()[0]-1, full.Shape()[0])
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "prefill last-row logits", last.Contiguous(), stepLogits)

	for l := range stStep.K {
		bitsEqualTensor(t, "layer K", stPre.K[l], stStep.K[l])
		bitsEqualTensor(t, "layer V", stPre.V[l], stStep.V[l])
		if (stPre.Mamba[l] == nil) != (stStep.Mamba[l] == nil) {
			t.Fatalf("layer %d: Mamba state nil mismatch", l)
		}
		if stStep.Mamba[l] != nil {
			bitsEqualF64(t, "layer ConvBuf", stPre.Mamba[l].ConvBuf, stStep.Mamba[l].ConvBuf)
			bitsEqualF64(t, "layer H", stPre.Mamba[l].H, stStep.Mamba[l].H)
		}
	}

	l1, err := m.DecodeStep(ctx, stStep, prefillContinueTok)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := m.DecodeStep(ctx, stPre, prefillContinueTok)
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "continuation logits", l2, l1)

	if _, err := m.Prefill(ctx, stPre, prefillPrompt); err == nil {
		t.Fatal("Prefill accepted a non-fresh decode state")
	}
}

// §V16 tier-1: RWKV batched prefill — the token-shift rows and the WKV
// numerator/denominator/max-exponent state, the continuation logits and the
// last-row prefill logits are all bit-equal to the stepwise path.
func TestRWKVPrefillStateParity(t *testing.T) {
	m := rwkvHF(t)
	ctx := backend.NewContext()

	stStep := m.NewDecodeState()
	var stepLogits *tensor.Tensor
	var err error
	for _, tok := range prefillPrompt {
		if stepLogits, err = m.DecodeStep(ctx, stStep, tok); err != nil {
			t.Fatal(err)
		}
	}

	stPre := m.NewDecodeState()
	full, err := m.Prefill(ctx, stPre, prefillPrompt)
	if err != nil {
		t.Fatal(err)
	}
	last, err := full.Slice(0, full.Shape()[0]-1, full.Shape()[0])
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "prefill last-row logits", last.Contiguous(), stepLogits)

	for l := range stStep.Layers {
		bitsEqualF64(t, "PrevTM", stPre.Layers[l].PrevTM, stStep.Layers[l].PrevTM)
		bitsEqualF64(t, "PrevCM", stPre.Layers[l].PrevCM, stStep.Layers[l].PrevCM)
		bitsEqualF64(t, "AA", stPre.Layers[l].AA, stStep.Layers[l].AA)
		bitsEqualF64(t, "BB", stPre.Layers[l].BB, stStep.Layers[l].BB)
		bitsEqualF64(t, "PP", stPre.Layers[l].PP, stStep.Layers[l].PP)
	}

	l1, err := m.DecodeStep(ctx, stStep, prefillContinueTok)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := m.DecodeStep(ctx, stPre, prefillContinueTok)
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "continuation logits", l2, l1)
}

// RWKV's prefill is state-general: prefilling a prompt in two chunks (the
// second continuing from the first's state) equals stepping every token.
func TestRWKVPrefillContinuesMidState(t *testing.T) {
	m := rwkvHF(t)
	ctx := backend.NewContext()

	stStep := m.NewDecodeState()
	var stepLogits *tensor.Tensor
	var err error
	for _, tok := range prefillPrompt {
		if stepLogits, err = m.DecodeStep(ctx, stStep, tok); err != nil {
			t.Fatal(err)
		}
	}

	st := m.NewDecodeState()
	if _, err := m.Prefill(ctx, st, prefillPrompt[:3]); err != nil {
		t.Fatal(err)
	}
	full, err := m.Prefill(ctx, st, prefillPrompt[3:])
	if err != nil {
		t.Fatal(err)
	}
	last, err := full.Slice(0, full.Shape()[0]-1, full.Shape()[0])
	if err != nil {
		t.Fatal(err)
	}
	bitsEqualTensor(t, "chunked prefill logits", last.Contiguous(), stepLogits)
	for l := range stStep.Layers {
		bitsEqualF64(t, "AA", st.Layers[l].AA, stStep.Layers[l].AA)
		bitsEqualF64(t, "BB", st.Layers[l].BB, stStep.Layers[l].BB)
		bitsEqualF64(t, "PP", st.Layers[l].PP, stStep.Layers[l].PP)
		bitsEqualF64(t, "PrevTM", st.Layers[l].PrevTM, stStep.Layers[l].PrevTM)
		bitsEqualF64(t, "PrevCM", st.Layers[l].PrevCM, stStep.Layers[l].PrevCM)
	}
}

// A prompt SHORTER than the conv window (T < d_conv−1) leaves the leading
// window slots zero — the batched tail capture must reproduce exactly what T
// stepwise window shifts leave behind, for both SSM families.
func TestMambaPrefillShortPromptStateParity(t *testing.T) {
	short := []int{3, 7} // golden d_conv = 4, so T=2 < d_conv−1
	t.Run("Mamba", func(t *testing.T) {
		m := mambaHF(t)
		ctx := backend.NewContext()
		stStep := m.NewDecodeState()
		for _, tok := range short {
			if _, err := m.DecodeStep(ctx, stStep, tok); err != nil {
				t.Fatal(err)
			}
		}
		stPre := m.NewDecodeState()
		if _, err := m.Prefill(ctx, stPre, short); err != nil {
			t.Fatal(err)
		}
		for l := range stStep.Layers {
			bitsEqualF64(t, "layer ConvBuf", stPre.Layers[l].ConvBuf, stStep.Layers[l].ConvBuf)
			bitsEqualF64(t, "layer H", stPre.Layers[l].H, stStep.Layers[l].H)
		}
	})
	t.Run("Mamba2", func(t *testing.T) {
		m := mamba2HF(t)
		ctx := backend.NewContext()
		stStep := m.NewDecodeState()
		for _, tok := range short {
			if _, err := m.DecodeStep(ctx, stStep, tok); err != nil {
				t.Fatal(err)
			}
		}
		stPre := m.NewDecodeState()
		if _, err := m.Prefill(ctx, stPre, short); err != nil {
			t.Fatal(err)
		}
		for l := range stStep.Layers {
			bitsEqualF64(t, "layer ConvBuf", stPre.Layers[l].ConvBuf, stStep.Layers[l].ConvBuf)
			bitsEqualF64(t, "layer H", stPre.Layers[l].H, stStep.Layers[l].H)
		}
	})
}

// stepwiseGreedy replays the PRE-prefill Generate loop (one DecodeStep per
// prompt token, then greedy sampling) — the reference the switched Generate
// must reproduce token-for-token.
func stepwiseGreedy(t *testing.T, step func(tok int) []float64, prompt []int, maxNew int) []int {
	t.Helper()
	out := append([]int(nil), prompt...)
	var logits []float64
	for _, tok := range prompt {
		logits = step(tok)
	}
	g := nlp.Greedy()
	for range maxNew {
		next := g.SampleWithHistory(logits, out)
		out = append(out, next)
		logits = step(next)
	}
	return out
}

// Generate (now prefill-backed) produces the identical greedy token sequence
// as the old stepwise prompt loop, for every recurrent family.
func TestRecurrentGenerateMatchesStepwise(t *testing.T) {
	prompt := []int{3, 7, 1, 9}
	const maxNew = 4

	rowF64 := func(l *tensor.Tensor) []float64 {
		v := l.Shape()[1]
		out := make([]float64, v)
		for j := range v {
			out[j] = l.AtF64(0, j)
		}
		return out
	}

	t.Run("Mamba", func(t *testing.T) {
		m := mambaHF(t)
		ctx := backend.NewContext()
		st := m.NewDecodeState()
		want := stepwiseGreedy(t, func(tok int) []float64 {
			l, err := m.DecodeStep(ctx, st, tok)
			if err != nil {
				t.Fatal(err)
			}
			return rowF64(l)
		}, prompt, maxNew)
		got, err := m.Generate(prompt, maxNew, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("token %d: prefill Generate %d != stepwise %d (%v vs %v)", i, got[i], want[i], got, want)
			}
		}
	})
	t.Run("Mamba2", func(t *testing.T) {
		m := mamba2HF(t)
		ctx := backend.NewContext()
		st := m.NewDecodeState()
		want := stepwiseGreedy(t, func(tok int) []float64 {
			l, err := m.DecodeStep(ctx, st, tok)
			if err != nil {
				t.Fatal(err)
			}
			return rowF64(l)
		}, prompt, maxNew)
		got, err := m.Generate(prompt, maxNew, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("token %d: prefill Generate %d != stepwise %d (%v vs %v)", i, got[i], want[i], got, want)
			}
		}
	})
	t.Run("Jamba", func(t *testing.T) {
		m := jambaGolden(t)
		ctx := backend.NewContext()
		st := m.NewDecodeState()
		want := stepwiseGreedy(t, func(tok int) []float64 {
			l, err := m.DecodeStep(ctx, st, tok)
			if err != nil {
				t.Fatal(err)
			}
			return rowF64(l)
		}, prompt, maxNew)
		got, err := m.Generate(prompt, maxNew, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("token %d: prefill Generate %d != stepwise %d (%v vs %v)", i, got[i], want[i], got, want)
			}
		}
	})
	t.Run("RWKV", func(t *testing.T) {
		m := rwkvHF(t)
		ctx := backend.NewContext()
		st := m.NewDecodeState()
		want := stepwiseGreedy(t, func(tok int) []float64 {
			l, err := m.DecodeStep(ctx, st, tok)
			if err != nil {
				t.Fatal(err)
			}
			return rowF64(l)
		}, prompt, maxNew)
		got, err := m.Generate(prompt, maxNew, nlp.Greedy())
		if err != nil {
			t.Fatal(err)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("token %d: prefill Generate %d != stepwise %d (%v vs %v)", i, got[i], want[i], got, want)
			}
		}
	})
}

// benchMamba builds a synthetic Mamba at realistic-ish dims for the §V22
// stepwise-vs-batched prompt-prefill benchmark: d_model 256, d_inner 512,
// N 16, d_conv 4, 2 layers, vocab 1024.
func benchMamba(tb testing.TB) *nlp.Mamba {
	tb.Helper()
	const (
		dModel = 256
		n      = 16
		dConv  = 4
		expand = 2
		layers = 2
		vocab  = 1024
	)
	emb := tensor.New(tensor.F64, tensor.Shape{vocab, dModel})
	nn.XavierUniform(emb, vocab, dModel, 1)
	head := tensor.New(tensor.F64, tensor.Shape{dModel, vocab})
	nn.XavierUniform(head, dModel, vocab, 2)
	m := &nlp.Mamba{
		Config: nlp.MambaConfig{
			DModel: dModel, N: n, DConv: dConv, Expand: expand,
			DtRank: (dModel + 15) / 16, Layers: layers, Vocab: vocab, Eps: 1e-5,
		},
		Embed: emb,
		Norm:  nn.NewRMSNorm(tensor.F64, dModel),
		Head:  head,
	}
	for l := range layers {
		mb, err := nn.NewMambaBlock(tensor.F64, dModel, n, dConv, expand, uint64(100*l+1))
		if err != nil {
			tb.Fatal(err)
		}
		m.Layers = append(m.Layers, nlp.MambaLayer{Norm: nn.NewRMSNorm(tensor.F64, dModel), Mixer: mb})
	}
	return m
}

var benchPrefillPrompt = func() []int {
	p := make([]int, 64)
	for i := range p {
		p[i] = (i*37 + 11) % 1024
	}
	return p
}()

// §V22: prompt absorption the OLD way — one DecodeStep per prompt token
// (per-token backend dispatches for every projection/conv).
func BenchmarkMambaPrefillStepwise(b *testing.B) {
	m := benchMamba(b)
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		st := m.NewDecodeState()
		for _, tok := range benchPrefillPrompt {
			if _, err := m.DecodeStep(ctx, st, tok); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// §V22: prompt absorption the NEW way — one batched Prefill (full-sequence
// projections/conv, host S6 scan).
func BenchmarkMambaPrefillBatched(b *testing.B) {
	m := benchMamba(b)
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		st := m.NewDecodeState()
		if _, err := m.Prefill(ctx, st, benchPrefillPrompt); err != nil {
			b.Fatal(err)
		}
	}
}
