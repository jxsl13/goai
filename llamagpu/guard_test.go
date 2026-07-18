package llamagpu

import (
	"strconv"
	"strings"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// These are white-box tests for the host-side validation newDecoder and GPUBert.summedEmbedding do
// BEFORE they touch a device, so they run on every platform — no GPU, no build tag. A malformed
// model must produce a clean error, never a panic and never a confident wrong answer (§V29).

// TestNewDecoderRejectsBlocklessModel pins the guard that newDecoder was missing while its ten
// sibling constructors (newDeepSeekV2Decoder, newMambaDecoder, newJambaDecoder, newMamba2Decoder,
// newRWKVDecoder and the four MoE builders) all had it.
//
// Without it a model with no blocks constructed fine and every subsequent call succeeded: Step
// returned plausible logits, Generate returned a full sequence, err was nil throughout — with the
// entire transformer stack silently skipped, because the upload loop `for _, b := range m.Blocks`
// simply never ran and Step then went embedding → final norm → lm_head. newDecoder backs New,
// NewCUDA, NewVulkan, NewLlamaQ8CUDA, NewLlamaQ4KCUDA and the Qwen2/Qwen3/Phi3/Granite
// constructors, so it is the most-used constructor in the package and was the only unguarded one.
//
// Both guards run before any device work, so a stub backendOps is enough to reach them.
func TestNewDecoderRejectsBlocklessModel(t *testing.T) {
	base := nlp.LlamaConfig{
		Vocab: 16, Ctx: 8, Dim: 8, Heads: 2, KVHeads: 1, Layers: 4,
		Hidden: 16, Eps: 1e-5, RopeBase: 10000,
	}
	tests := []struct {
		name    string
		model   func(t *testing.T) *nlp.Llama
		wantMsg string
	}{
		{
			// nlp.NewLlama with Layers 0 appends no blocks and returns no error, so this is
			// reachable straight from the public API.
			name: "zero blocks",
			model: func(t *testing.T) *nlp.Llama {
				cfg := base
				cfg.Layers = 0
				m, err := nlp.NewLlama(cfg, 3)
				if err != nil {
					t.Fatalf("NewLlama: %v", err)
				}
				if len(m.Blocks) != 0 {
					t.Fatalf("setup: got %d blocks, want 0", len(m.Blocks))
				}
				return m
			},
			wantMsg: "has no blocks",
		},
		{
			// A config claiming more layers than the checkpoint carries is the same failure
			// wearing a disguise: the missing blocks are skipped just as silently.
			name: "config claims more layers than blocks",
			model: func(t *testing.T) *nlp.Llama {
				m, err := nlp.NewLlama(base, 3)
				if err != nil {
					t.Fatalf("NewLlama: %v", err)
				}
				m.Blocks = m.Blocks[:2] // cfg.Layers stays 4
				return m
			},
			wantMsg: "layers but",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := tc.model(t)
			d, err := newDecoder(m, backendOps{name: "test"})
			if err == nil {
				d.Release()
				t.Fatalf("newDecoder accepted a malformed model (Layers %d, %d blocks); want an error",
					m.Config.Layers, len(m.Blocks))
			}
			if !strings.Contains(err.Error(), tc.wantMsg) {
				t.Fatalf("newDecoder error = %q, want it to contain %q", err, tc.wantMsg)
			}
			t.Logf("newDecoder rejected %s: %v", tc.name, err)
		})
	}
}

// TestNewDecoderAcceptsUndeclaredLayerCount is the over-rejection control for the layer-count
// reconciliation: a config that leaves Layers unset still carries usable blocks, and nothing in
// newDecoder reads cfg.Layers, so it must NOT be rejected. Only a config that actively claims a
// different count is a mismatch. The stub backendOps means construction fails later, at the first
// buffer upload — the assertion is only that it fails somewhere OTHER than the two guards.
func TestNewDecoderAcceptsUndeclaredLayerCount(t *testing.T) {
	cfg := nlp.LlamaConfig{
		Vocab: 16, Ctx: 8, Dim: 8, Heads: 2, KVHeads: 1, Layers: 2,
		Hidden: 16, Eps: 1e-5, RopeBase: 10000,
	}
	m, err := nlp.NewLlama(cfg, 3)
	if err != nil {
		t.Fatalf("NewLlama: %v", err)
	}
	m.Config.Layers = 0 // config does not declare a count; the 2 blocks are real
	defer func() {
		// A nil newBuffer in the stub ops panics on the first upload. That is past both guards,
		// which is exactly what this test asserts.
		if r := recover(); r == nil {
			t.Log("construction reached the upload path without panicking")
		}
	}()
	if _, err := newDecoder(m, backendOps{name: "test"}); err != nil {
		if strings.Contains(err.Error(), "has no blocks") || strings.Contains(err.Error(), "layers but") {
			t.Fatalf("newDecoder rejected a model with real blocks and an undeclared layer count: %v", err)
		}
	}
}

// TestBertSegmentIDOutOfRangeErrors pins the bound check on BERT segment (token-type) ids.
// summedEmbedding validated the token against e.vocab but read segments[i] unchecked and passed it
// straight to e.segEmb.AtF64(seg, j), so an out-of-range id panicked with an index-out-of-range
// deep inside the tensor — while the CPU reference nlp.Bert.Embed returned a clean error on the
// identical input. cuda_bert_test.go compares exactly these two paths, so they must agree on what
// is an error.
func TestBertSegmentIDOutOfRangeErrors(t *testing.T) {
	const (
		vocab     = 6
		maxPos    = 8
		typeVocab = 2
		dim       = 4
	)
	fill := func(rows, cols int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		for i := range rows {
			for j := range cols {
				x.SetF64(float64(i*cols+j)*0.01, i, j)
			}
		}
		return x
	}
	tokEmb, posEmb, segEmb := fill(vocab, dim), fill(maxPos, dim), fill(typeVocab, dim)

	gpu := &GPUBert{
		ops: backendOps{name: "test"},
		dim: dim, vocab: vocab, maxPos: maxPos, typeVocab: typeVocab,
		tokEmb: tokEmb, posEmb: posEmb, segEmb: segEmb,
	}
	ones := func(n int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			x.SetF64(1, i)
		}
		return x
	}
	cpu := &nlp.Bert{
		Config: nlp.BertConfig{Vocab: vocab, MaxPos: maxPos, TypeVocab: typeVocab, Dim: dim, Heads: 2, Eps: 1e-12},
		TokEmb: tokEmb, PosEmb: posEmb, SegEmb: segEmb,
		EmbLN: &nn.LayerNorm{Gamma: ones(dim), Beta: tensor.New(tensor.F64, tensor.Shape{dim}), Eps: 1e-12},
	}
	ctx := backend.NewContext().WithBackend(backend.Reference())
	tokens := []int{1, 2, 3}

	for _, seg := range []int{typeVocab, typeVocab + 7, -1} {
		segments := []int{0, seg, 0}
		t.Run("segment id "+strconv.Itoa(seg), func(t *testing.T) {
			// The CPU reference is the contract: a clean error, no panic.
			_, cpuErr := cpu.Embed(ctx, tokens, segments)
			if cpuErr == nil {
				t.Fatalf("setup: CPU reference accepted segment id %d; it is supposed to reject it", seg)
			}
			gotErr := func() (err error) {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("summedEmbedding panicked on segment id %d (%v); want a clean error like the CPU path's %q", seg, r, cpuErr)
					}
				}()
				_, err = gpu.summedEmbedding(tokens, segments)
				return err
			}()
			if gotErr == nil {
				t.Fatalf("summedEmbedding accepted out-of-range segment id %d; CPU reference says %q", seg, cpuErr)
			}
			t.Logf("segment id %d — gpu: %v | cpu reference: %v", seg, gotErr, cpuErr)
		})
	}

	// Control: in-range ids (and the nil-segments single-sentence path) still work.
	if _, err := gpu.summedEmbedding(tokens, []int{0, 1, 1}); err != nil {
		t.Fatalf("summedEmbedding rejected valid segment ids: %v", err)
	}
	if _, err := gpu.summedEmbedding(tokens, nil); err != nil {
		t.Fatalf("summedEmbedding rejected nil segments: %v", err)
	}
	// DistilBERT carries no segment table (SegEmb nil, TypeVocab 0) — segments are unused there,
	// so the bound must not fire.
	noSeg := *gpu
	noSeg.segEmb, noSeg.typeVocab = nil, 0
	if _, err := noSeg.summedEmbedding(tokens, []int{0, 5, 0}); err != nil {
		t.Fatalf("summedEmbedding rejected segments on a model without a segment table: %v", err)
	}
}
