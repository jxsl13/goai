package nlp

import (
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// BertClassifier is a sequence-classification head over a [Bert] encoder (also
// RoBERTa/DistilBERT, which share the type): it mean-pools the encoder output and
// applies a linear head to produce class logits. It is the ready-made shape for
// the common "fine-tune a loaded HF encoder for classification" task — build one
// over a [BertFromHF] model, optionally attach [ApplyLoRABert] adapters, and train
// [BertClassifier.Params] (or just the head + adapters) with the autograd toolkit.
type BertClassifier struct {
	Encoder *Bert
	Head    *nn.Linear // [dim, classes]
	Classes int
}

// NewBertClassifier attaches a fresh classification head (classes outputs) to enc.
func NewBertClassifier(enc *Bert, classes int, seed uint64) *BertClassifier {
	return &BertClassifier{Encoder: enc, Head: nn.NewLinear(tensor.F64, enc.Config.Dim, classes, seed), Classes: classes}
}

// Logits returns the class logits [1, classes] for one tokenized sequence
// (segments may be nil). It runs on ctx, so passing a recording tape context
// trains through the encoder + head; a plain context is inference.
func (c *BertClassifier) Logits(ctx *backend.Context, tokens, segments []int) (*tensor.Tensor, error) {
	hidden, err := c.Encoder.Forward(ctx, tokens, segments)
	if err != nil {
		return nil, err
	}
	seq := len(tokens)
	ones := tensor.New(tensor.F64, tensor.Shape{1, seq})
	for i := 0; i < seq; i++ {
		ones.SetF64(1.0/float64(seq), 0, i) // mean-pool weights
	}
	pooled, err := exec1(ctx, backend.OpMatMul, nil, ones, hidden) // [1, dim]
	if err != nil {
		return nil, err
	}
	return c.Head.Forward(ctx, pooled) // [1, classes]
}

// Params returns the trainable tensors (encoder + head) for full fine-tuning. For
// parameter-efficient tuning, freeze the encoder and train only Head.Params() plus
// the adapters returned by [ApplyLoRABert].
func (c *BertClassifier) Params() []*tensor.Tensor {
	return append(c.Encoder.Params(), c.Head.Params()...)
}
