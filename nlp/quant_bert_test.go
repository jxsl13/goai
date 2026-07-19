package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/tensor"
)

// quantBertHFMap returns a block-aligned, Hugging-Face-shaped BERT checkpoint.
// The path under test is the public loaded-model flow: BertFromHF consumes this
// map before QuantizeBert replaces every large projection with persistent bytes.
func quantBertHFMap(layers int) map[string]*tensor.Tensor {
	const vocab, maxPos, types, dim, hidden = 64, 16, 2, 32, 64
	rng := rand.New(rand.NewPCG(739, 1))
	randTensor := func(shape tensor.Shape, scale, shift float64) *tensor.Tensor {
		x := tensor.New(tensor.F32, shape)
		for i := range x.Numel() {
			x.SetF64(shift+scale*rng.NormFloat64(), tensor.Unravel(i, shape)...)
		}
		return x
	}
	m := map[string]*tensor.Tensor{
		"embeddings.word_embeddings.weight":       randTensor(tensor.Shape{vocab, dim}, 0.08, 0),
		"embeddings.position_embeddings.weight":   randTensor(tensor.Shape{maxPos, dim}, 0.08, 0),
		"embeddings.token_type_embeddings.weight": randTensor(tensor.Shape{types, dim}, 0.08, 0),
		"embeddings.LayerNorm.weight":             randTensor(tensor.Shape{dim}, 0.02, 1),
		"embeddings.LayerNorm.bias":               randTensor(tensor.Shape{dim}, 0.01, 0),
	}
	for l := range layers {
		p := fmt.Sprintf("encoder.layer.%d.", l)
		for _, name := range []string{
			"attention.self.query", "attention.self.key", "attention.self.value",
			"attention.output.dense",
		} {
			m[p+name+".weight"] = randTensor(tensor.Shape{dim, dim}, 0.06, 0)
			m[p+name+".bias"] = randTensor(tensor.Shape{dim}, 0.01, 0)
		}
		m[p+"intermediate.dense.weight"] = randTensor(tensor.Shape{hidden, dim}, 0.06, 0)
		m[p+"intermediate.dense.bias"] = randTensor(tensor.Shape{hidden}, 0.01, 0)
		m[p+"output.dense.weight"] = randTensor(tensor.Shape{dim, hidden}, 0.06, 0)
		m[p+"output.dense.bias"] = randTensor(tensor.Shape{dim}, 0.01, 0)
		for _, name := range []string{"attention.output.LayerNorm", "output.LayerNorm"} {
			m[p+name+".weight"] = randTensor(tensor.Shape{dim}, 0.02, 1)
			m[p+name+".bias"] = randTensor(tensor.Shape{dim}, 0.01, 0)
		}
	}
	return m
}

func quantBertError(got, want *tensor.Tensor) (relRMS, cosine float64) {
	var diff2, want2, got2, dot float64
	for i := range got.Numel() {
		idx := tensor.Unravel(i, got.Shape())
		g, w := got.AtF64(idx...), want.AtF64(idx...)
		d := g - w
		diff2 += d * d
		want2 += w * w
		got2 += g * g
		dot += g * w
	}
	return math.Sqrt(diff2 / want2), dot / math.Sqrt(got2*want2)
}

// T739's decisive end-to-end gate: a Hugging-Face-shaped checkpoint is loaded
// through BertFromHF, converted to persistent Q8_0 storage, and the entire
// bidirectional two-layer encoder remains near-lossless against the f64 model.
func TestQuantBertLoadedHFForwardQ8(t *testing.T) {
	m, err := BertFromHF(quantBertHFMap(2), BertConfig{Heads: 4, Eps: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	q, err := QuantizeBert(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{1, 9, 4, 23, 7, 2}
	segments := []int{0, 0, 0, 1, 1, 1}
	want, err := m.Forward(backend.NewContext(), tokens, segments)
	if err != nil {
		t.Fatal(err)
	}
	got, err := q.Forward(backend.NewContext(), tokens, segments)
	if err != nil {
		t.Fatal(err)
	}
	rel, cos := quantBertError(got, want)
	t.Logf("Q8_0 loaded-HF BERT hidden rel-RMS %.6f%% cosine %.9f", 100*rel, cos)
	if rel > 0.02 || cos < 0.999 {
		t.Fatalf("quantized hidden states diverge: rel-RMS %.4g cosine %.6f", rel, cos)
	}
}

// The old T739 spike re-quantized an F64 weight on every call and retained that
// weight, so it saved no memory. QuantBert must instead retain exactly the Q8_0
// blocks for all six projections and no float projection tensors.
func TestQuantBertProjectionStorageIsActuallyQ8(t *testing.T) {
	const layers, dim, hidden = 2, 32, 64
	m, err := BertFromHF(quantBertHFMap(layers), BertConfig{Heads: 4, Eps: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	q, err := QuantizeBert(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	floatElems := layers * (4*dim*dim + dim*hidden + hidden*dim)
	wantBytes := floatElems / 32 * 34 // Q8_0: f16 scale + 32 int8 values per block
	if got := q.projectionBytes(); got != wantBytes {
		t.Fatalf("projection bytes %d, want exact Q8_0 payload %d", got, wantBytes)
	}
	floatBytes := floatElems * 8 // BertFromHF widens projections to F64
	if ratio := float64(floatBytes) / float64(wantBytes); ratio < 7.5 {
		t.Fatalf("projection storage reduction %.3fx, want >= 7.5x", ratio)
	} else {
		t.Logf("projection storage %d -> %d bytes (%.3fx smaller)", floatBytes, wantBytes, ratio)
	}
	for i, l := range q.layers {
		for _, item := range []struct {
			name   string
			weight []byte
		}{
			// A structural tripwire: every large slot is backed by byte storage. The
			// concrete QuantLinear fields themselves make a retained Tensor impossible.
			{"q", l.wq.Weight}, {"k", l.wk.Weight}, {"v", l.wv.Weight},
			{"o", l.wo.Weight}, {"1", l.w1.Weight}, {"2", l.w2.Weight},
		} {
			if len(item.weight) == 0 {
				t.Fatalf("layer %d projection %s has no quantized bytes", i, item.name)
			}
		}
	}
}

// QuantBert is intentionally the quantized twin of the shared Bert structure,
// so the two loader-level variants must survive conversion too: RoBERTa's
// position offset and DistilBERT's absent segment embedding.
func TestQuantBertLoadedHFVariants(t *testing.T) {
	for _, tc := range []struct {
		name      string
		posOffset int
		noSegment bool
	}{
		{name: "roberta-position-offset", posOffset: 2},
		{name: "distilbert-no-segment", noSegment: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, err := BertFromHF(quantBertHFMap(1), BertConfig{Heads: 4, Eps: 1e-12, PosOffset: tc.posOffset})
			if err != nil {
				t.Fatal(err)
			}
			if tc.noSegment {
				m.SegEmb = nil // DistilBertFromHF's structural distinction
			}
			q, err := QuantizeBert(m, gguf.Q8_0)
			if err != nil {
				t.Fatal(err)
			}
			tokens := []int{1, 9, 4, 23}
			want, err := m.Forward(backend.NewContext(), tokens, nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := q.Forward(backend.NewContext(), tokens, nil)
			if err != nil {
				t.Fatal(err)
			}
			rel, cos := quantBertError(got, want)
			if rel > 0.02 || cos < 0.999 {
				t.Fatalf("variant hidden states diverge: rel-RMS %.4g cosine %.6f", rel, cos)
			}
		})
	}
}

func TestQuantBertBoundaries(t *testing.T) {
	if _, err := QuantizeBert(nil, gguf.Q8_0); err == nil {
		t.Fatal("nil model: want error")
	}
	m, err := BertFromHF(quantBertHFMap(1), BertConfig{Heads: 4, Eps: 1e-12})
	if err != nil {
		t.Fatal(err)
	}
	q, err := QuantizeBert(m, gguf.Q8_0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Forward(backend.NewContext(), []int{1, 2}, []int{0}); err == nil {
		t.Fatal("segment length mismatch: want error")
	}
	if _, err := q.Forward(backend.NewContext(), []int{m.Config.Vocab}, nil); err == nil {
		t.Fatal("out-of-vocabulary token: want error")
	}
	if err := q.Close(); err != nil {
		t.Fatal(err)
	}
	if err := q.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func ExampleQuantBert() {
	m, _ := BertFromHF(quantBertHFMap(1), BertConfig{Heads: 4, Eps: 1e-12})
	q, _ := QuantizeBert(m, gguf.Q8_0)
	hidden, _ := q.Forward(backend.NewContext(), []int{1, 2, 3}, nil)
	fmt.Println(hidden.Shape(), "quantized projection bytes:", q.projectionBytes())
	_ = q.Close()
	// Output: (3, 32) quantized projection bytes: 8704
}
