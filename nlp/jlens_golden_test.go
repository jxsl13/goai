package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/format/npy"
	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// The §T812 tier-1 anchor (§V16, B67-lesson: independent fixtures): every
// fixture in testdata/jlens_* was produced by the ACTUAL Anthropic reference
// implementation (github.com/anthropics/jacobian-lens) running in the repo
// .venv against a tiny seeded transformers LlamaForCausalLM — regenerate with
// scripts/golden_jlens.py. Geometry: vocab 32, dim 16, 4 layers, heads 4,
// kv 2; reference fit with skip_first=2 over 16 sequences of 24 tokens.

// jlensGoldenModel loads the tiny HF Llama the reference fixtures were
// computed on.
func jlensGoldenModel(t *testing.T) *nlp.Llama {
	t.Helper()
	ts, _, err := safetensors.LoadFile("testdata/jlens_hf.safetensors")
	if err != nil {
		t.Fatalf("load weights (run scripts/golden_jlens.py): %v", err)
	}
	model, err := nlp.LlamaFromHF(ts, nlp.LlamaConfig{
		Heads: 4, KVHeads: 2, Eps: 1e-5, RopeBase: 10000, Ctx: 64,
	})
	if err != nil {
		t.Fatalf("LlamaFromHF: %v", err)
	}
	return model
}

// jlensGoldenRefJ loads the reference-fitted J matrices [K, dim, dim] (the
// f32 values the reference stores, written losslessly as f64).
func jlensGoldenRefJ(t *testing.T) *tensor.Tensor {
	t.Helper()
	refJ, err := npy.LoadFile("testdata/jlens_ref_J.npy")
	if err != nil {
		t.Fatalf("load reference J (run scripts/golden_jlens.py): %v", err)
	}
	if s := refJ.Shape(); len(s) != 3 || s[0] != 3 || s[1] != 16 || s[2] != 16 {
		t.Fatalf("reference J shape %v, want [3 16 16]", refJ.Shape())
	}
	return refJ
}

// jlensGoldenTokens loads a token-id fixture stored as f8.
func jlensGoldenTokens(t *testing.T, path string) [][]int {
	t.Helper()
	raw, err := npy.LoadFile(path)
	if err != nil {
		t.Fatalf("load %s (run scripts/golden_jlens.py): %v", path, err)
	}
	shape := raw.Shape()
	if len(shape) == 1 {
		row := make([]int, shape[0])
		for i := range row {
			row[i] = int(raw.AtF64(i))
		}
		return [][]int{row}
	}
	out := make([][]int, shape[0])
	for i := range out {
		out[i] = make([]int, shape[1])
		for j := range out[i] {
			out[i][j] = int(raw.AtF64(i, j))
		}
	}
	return out
}

// §T812 golden (a): JLensFromPT on the RAW reference artifact. This pins the
// actual .pt layout — {"J": {int layer: fp16 [d,d]}, "n_prompts",
// "source_layers", "d_model"} — and the corrected §T811 assumptions: int dict
// keys (not string "jacobians.N" paths), block-OUTPUT layer indexing
// (reference l = GoAI l+1), no post-embedding matrix (J[0] nil), and a
// synthesized identity at the target layer. Matrices must match the exported
// reference J to fp16 storage precision, verbatim orientation (no transpose).
func TestJLensFromPTReferenceArtifact(t *testing.T) {
	jl, err := nlp.JLensFromPT("testdata/jlens_ref.pt")
	if err != nil {
		t.Fatalf("JLensFromPT (run scripts/golden_jlens.py): %v", err)
	}
	if jl.Dim != 16 || jl.Layers != 4 || len(jl.J) != 5 {
		t.Fatalf("imported geometry {dim %d, layers %d, %d entries}, want {16, 4, 5}", jl.Dim, jl.Layers, len(jl.J))
	}
	if jl.J[0] != nil {
		t.Fatal("J[0] must be nil: the reference artifact has no post-embedding Jacobian")
	}
	for a := range jl.Dim {
		for b := range jl.Dim {
			want := 0.0
			if a == b {
				want = 1
			}
			if jl.J[4].AtF64(a, b) != want {
				t.Fatalf("J[Layers] must be the synthesized identity; [%d,%d] = %g", a, b, jl.J[4].AtF64(a, b))
			}
		}
	}
	refJ := jlensGoldenRefJ(t)
	var maxAbs float64
	for l := range 3 {
		for a := range 16 {
			for b := range 16 {
				d := math.Abs(jl.J[l+1].AtF64(a, b) - refJ.AtF64(l, a, b))
				if d > maxAbs {
					maxAbs = d
				}
			}
		}
	}
	t.Logf("imported J (fp16 artifact) vs reference f32 J: max abs diff %.3e", maxAbs)
	// The artifact stores fp16 (half-ULP ≈ 4.9e-4 · |v|, |J| entries O(1)).
	if maxAbs > 2e-3 {
		t.Fatalf("imported J diverges beyond fp16 storage error: %.3e (orientation/indexing wrong?)", maxAbs)
	}
	// fp16 error must be visibly nonzero — a 0 here would mean the goldens
	// were accidentally compared against themselves.
	if maxAbs == 0 {
		t.Fatal("fp16 artifact matched f32 export bit-exactly — fixture generation suspect")
	}
}

// §T812 golden (b), fit-parity: FitJLens on the SAME model (via LlamaFromHF)
// and the SAME corpus must reproduce the reference implementation's J
// matrices. Both estimators are exact VJP math over identical valid-position
// masks (skip_first=2, final position excluded — the §T812 semantic fix that
// aligned FitJLens with the reference); the only divergence left is the
// reference's f32 accumulation against GoAI's f64, so agreement is tight.
func TestFitJLensMatchesReference(t *testing.T) {
	model := jlensGoldenModel(t)
	corpus := jlensGoldenTokens(t, "testdata/jlens_corpus.npy")
	if len(corpus) != 16 || len(corpus[0]) != 24 {
		t.Fatalf("corpus %dx%d, want 16x24", len(corpus), len(corpus[0]))
	}
	jl, err := nlp.FitJLens(model, corpus, nlp.WithJLensSkipFirst(2))
	if err != nil {
		t.Fatalf("FitJLens: %v", err)
	}
	refJ := jlensGoldenRefJ(t)
	var maxAbs, maxVal float64
	for l := range 3 {
		for a := range 16 {
			for b := range 16 {
				d := math.Abs(jl.J[l+1].AtF64(a, b) - refJ.AtF64(l, a, b))
				if d > maxAbs {
					maxAbs = d
				}
				if v := math.Abs(refJ.AtF64(l, a, b)); v > maxVal {
					maxVal = v
				}
			}
		}
	}
	t.Logf("FitJLens vs reference fit: max abs diff %.3e (max |J| entry %.3f)", maxAbs, maxVal)
	if maxAbs > 1e-5 {
		t.Fatalf("FitJLens diverges from the reference estimator: max abs diff %.3e", maxAbs)
	}
}

// §T812 golden (c), readout-parity: JLens.Apply / ApplyAt against the
// reference lens.apply logits at fixed (prompt, layer, position) probes. The
// lens is built from the reference's own stored J values (f32, lossless in
// the .npy) so the comparison isolates the READOUT path: transport J·h, the
// model's final RMSNorm and unembedding. Rows 0..2 of the fixture are
// reference source layers 0..2 (GoAI layers 1..3); row 3 is the model's own
// logits, which the GoAI layer==Layers identity readout must reproduce.
func TestJLensApplyMatchesReference(t *testing.T) {
	model := jlensGoldenModel(t)
	refJ := jlensGoldenRefJ(t)
	probe := jlensGoldenTokens(t, "testdata/jlens_probe.npy")[0]
	golden, err := npy.LoadFile("testdata/jlens_ref_readout.npy")
	if err != nil {
		t.Fatalf("load readout golden (run scripts/golden_jlens.py): %v", err)
	}
	if s := golden.Shape(); len(s) != 3 || s[0] != 4 || s[1] != 2 || s[2] != 32 {
		t.Fatalf("readout golden shape %v, want [4 2 32]", golden.Shape())
	}
	positions := []int{3, 10} // PROBE_POSITIONS in scripts/golden_jlens.py

	// Assemble the lens exactly as JLensFromPT does, but from the lossless f32
	// export: J[0] nil, J[l+1] = reference J_l, J[4] = identity.
	jl := &nlp.JLens{Dim: 16, Layers: 4, Arch: "llama", Weight: 16}
	jl.J = make([]*tensor.Tensor, 5)
	for l := range 3 {
		m := tensor.New(tensor.F64, tensor.Shape{16, 16})
		for a := range 16 {
			for b := range 16 {
				m.SetF64(refJ.AtF64(l, a, b), a, b)
			}
		}
		jl.J[l+1] = m
	}
	eye := tensor.New(tensor.F64, tensor.Shape{16, 16})
	for i := range 16 {
		eye.SetF64(1, i, i)
	}
	jl.J[4] = eye

	var maxAbs float64
	for row := range 4 {
		layer := row + 1 // reference source layer row → GoAI layer; row 3 = layer 4 = model output
		for pi, pos := range positions {
			r, err := jl.ApplyAt(model, backend.NewContext(), probe, layer, pos)
			if err != nil {
				t.Fatalf("ApplyAt(layer %d, pos %d): %v", layer, pos, err)
			}
			for v := range 32 {
				d := math.Abs(r.Logits[v] - golden.AtF64(row, pi, v))
				if d > maxAbs {
					maxAbs = d
				}
			}
		}
	}
	t.Logf("Apply vs reference lens.apply: max abs logit diff %.3e", maxAbs)
	// Reference apply runs the transport in f32 (its stored J and residuals
	// are .float()'ed) and rounds the output to f32; GoAI computes f64 on the
	// same J values — the diff is f32 rounding noise.
	if maxAbs > 1e-4 {
		t.Fatalf("Apply diverges from the reference readout: max abs diff %.3e", maxAbs)
	}

	// Negative (Python-style) position indexing mirrors the reference: -2 on
	// the 12-token probe is position 10.
	rNeg, err := jl.ApplyAt(model, backend.NewContext(), probe, 2, -2)
	if err != nil {
		t.Fatal(err)
	}
	rPos, err := jl.ApplyAt(model, backend.NewContext(), probe, 2, 10)
	if err != nil {
		t.Fatal(err)
	}
	for v := range 32 {
		if rNeg.Logits[v] != rPos.Logits[v] {
			t.Fatalf("ApplyAt(-2) != ApplyAt(10) at token %d", v)
		}
	}

	// Slice consistency on the same fixtures: the bottom row is the model's
	// real output (Top == Output, Rank all 0), rows are ascending layers 1..4
	// (layer 0 unfitted in an imported lens), and every cell's Rank is
	// consistent with an independent ApplyAt RankOf.
	sl, err := jl.Slice(model, backend.NewContext(), probe)
	if err != nil {
		t.Fatalf("Slice: %v", err)
	}
	if len(sl.Rows) != 4 || sl.Rows[0].Layer != 1 || sl.Rows[3].Layer != 4 {
		t.Fatalf("Slice rows: got %d rows (first layer %d, last %d), want layers 1..4",
			len(sl.Rows), sl.Rows[0].Layer, sl.Rows[len(sl.Rows)-1].Layer)
	}
	bottom := sl.Rows[3]
	for p := range probe {
		if bottom.Top[p] != sl.Output[p] || bottom.Rank[p] != 0 {
			t.Fatalf("bottom row pos %d: top %d rank %d, want the model output %d at rank 0",
				p, bottom.Top[p], bottom.Rank[p], sl.Output[p])
		}
	}
	r2, err := jl.ApplyAt(model, backend.NewContext(), probe, 2, 5)
	if err != nil {
		t.Fatal(err)
	}
	if sl.Rows[1].Top[5] != r2.Ranked[0].Token || sl.Rows[1].Rank[5] != r2.RankOf(sl.Output[5]) {
		t.Fatalf("Slice cell (layer 2, pos 5) inconsistent with ApplyAt: top %d vs %d, rank %d vs %d",
			sl.Rows[1].Top[5], r2.Ranked[0].Token, sl.Rows[1].Rank[5], r2.RankOf(sl.Output[5]))
	}
}
