// Package vision is the L4 computer-vision domain package (§G1/§T457): image
// models assembled from the verified nn building blocks.
//
// # For the AI professional
//
// CNN is a reference convolutional classifier over NCHW image tensors: a stack
// of Conv2D(k, pad k/2) → ReLU → MaxPool2D(2) blocks (each halving the spatial
// size and expanding channels), global average pooling, and a Linear head to
// class logits — the classic LeNet/VGG-style shape with a modern GAP head. All
// compute runs through the backend dispatch (fused conv backward, backend
// auto-selection), and Params feeds any optimizer in nn, so the standard
// tape → loss → Backward → Step loop trains it.
//
// # For the newcomer
//
// A convolutional network looks at an image through small sliding windows,
// learning filters that light up on local patterns (edges, corners); stacking
// such layers with downsampling lets later filters see larger structures. The
// final layers average what the filters found and score each class. Build one
// with NewCNN, train it with the loop shown in the examples, and call Forward
// to get a score per class for each image.
//
// Further reading: LeCun, Bottou, Bengio & Haffner 1998, "Gradient-Based Learning Applied to Document Recognition", the CNN blueprint; Goodfellow et al., "Deep Learning", ch. 9 for convolution foundations.
//
// In plain terms: computer vision — models that look at grids of pixels and answer "what is in this image", from the classic convolutional network to the transformer that reads images like sentences.
package vision

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// CNN is a small convolutional image classifier: per stage
// Conv2D(kernel, pad kernel/2) → ReLU → MaxPool2D(2), then global average
// pooling and a Linear head. Input x is NCHW [batch, in, H, W]; H and W must be
// divisible by 2^stages. Forward returns logits [batch, classes].
type CNN struct {
	Convs []*nn.Conv2D // the conv stages, in order
	Head  *nn.Linear   // the classification head over the pooled features
	relu  *nn.Activation
}

// CNNOption configures NewCNN (functional options, §C12).
type CNNOption func(*cnnCfg)

type cnnCfg struct {
	channels []int
	kernel   int
	dtype    tensor.Dtype
}

// WithChannels sets the per-stage output channel counts (default [8, 16]); the
// length is the number of conv/pool stages.
func WithChannels(ch ...int) CNNOption { return func(c *cnnCfg) { c.channels = ch } }

// WithKernel sets the square conv kernel size (default 3; must be odd so
// pad = kernel/2 preserves the spatial size).
func WithKernel(k int) CNNOption { return func(c *cnnCfg) { c.kernel = k } }

// WithDtype sets the parameter dtype (default F32).
func WithDtype(d tensor.Dtype) CNNOption { return func(c *cnnCfg) { c.dtype = d } }

// NewCNN builds the classifier for in-channel images and the given number of
// classes. Deterministic via seed.
func NewCNN(in, classes int, seed uint64, opts ...CNNOption) (*CNN, error) {
	if in <= 0 || classes <= 0 {
		return nil, fmt.Errorf("vision: NewCNN needs in, classes > 0 (got %d, %d)", in, classes)
	}
	cfg := cnnCfg{channels: []int{8, 16}, kernel: 3, dtype: tensor.F32}
	for _, o := range opts {
		o(&cfg)
	}
	if len(cfg.channels) == 0 {
		return nil, fmt.Errorf("vision: NewCNN needs ≥1 stage")
	}
	if cfg.kernel%2 == 0 || cfg.kernel <= 0 {
		return nil, fmt.Errorf("vision: kernel must be odd and > 0, got %d", cfg.kernel)
	}
	m := &CNN{relu: nn.ReLU()}
	prev := in
	for i, ch := range cfg.channels {
		if ch <= 0 {
			return nil, fmt.Errorf("vision: stage %d channels must be > 0, got %d", i, ch)
		}
		conv, err := nn.NewConv2D(cfg.dtype, prev, ch, cfg.kernel, 1, cfg.kernel/2, seed+uint64(i))
		if err != nil {
			return nil, err
		}
		m.Convs = append(m.Convs, conv)
		prev = ch
	}
	m.Head = nn.NewLinear(cfg.dtype, prev, classes, seed+uint64(len(cfg.channels)))
	return m, nil
}

// Forward computes class logits [batch, classes] for images x [batch, in, H, W].
func (m *CNN) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	if x.Ndim() != 4 {
		return nil, fmt.Errorf("vision: CNN expects x [batch,in,H,W], got %v", x.Shape())
	}
	pool := &nn.MaxPool2D{Kernel: 2}
	var err error
	for i, conv := range m.Convs {
		if x, err = conv.Forward(ctx, x); err != nil {
			return nil, err
		}
		if x, err = m.relu.Forward(ctx, x); err != nil {
			return nil, err
		}
		if x.Shape()[2] < 2 || x.Shape()[3] < 2 || x.Shape()[2]%2 != 0 || x.Shape()[3]%2 != 0 {
			return nil, fmt.Errorf("vision: stage %d spatial size %dx%d not divisible by 2 — H,W must be multiples of 2^stages", i, x.Shape()[2], x.Shape()[3])
		}
		if x, err = pool.Forward(ctx, x); err != nil {
			return nil, err
		}
	}
	// global average pool → [batch, channels] → head.
	h := x.Shape()[2]
	if x.Shape()[3] != h {
		// pool the remaining rectangle in one op per dim is not supported; require square.
		return nil, fmt.Errorf("vision: CNN needs square inputs, got %dx%d after the stages", x.Shape()[2], x.Shape()[3])
	}
	gap, err := backend.Execute(ctx, backend.OpAvgPool2D, []*tensor.Tensor{x}, backend.PoolAttrs{Kernel: h})
	if err != nil {
		return nil, err
	}
	flat, err := backend.Execute(ctx, backend.OpReshape, []*tensor.Tensor{gap[0]},
		backend.ReshapeAttrs{Shape: tensor.Shape{x.Shape()[0], x.Shape()[1]}})
	if err != nil {
		return nil, err
	}
	return m.Head.Forward(ctx, flat[0])
}

// Params returns every trainable tensor for optimizers.
func (m *CNN) Params() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, c := range m.Convs {
		ps = append(ps, c.Params()...)
	}
	return append(ps, m.Head.Params()...)
}

// Safetensors returns the model's parameters under a stable naming convention —
// the exact inverse of FromSafetensors — so a trained CNN checkpoints with
// safetensors.Save/SaveFile and reloads bit-identically (§T458, the §T435
// pattern). The map holds the LIVE tensors (no copy): serialize before mutating.
func (m *CNN) Safetensors() map[string]*tensor.Tensor {
	ts := map[string]*tensor.Tensor{
		"head.w": m.Head.W,
		"head.b": m.Head.B,
	}
	for i, c := range m.Convs {
		ts[fmt.Sprintf("convs.%d.w", i)] = c.W
		ts[fmt.Sprintf("convs.%d.b", i)] = c.B
	}
	return ts
}

// FromSafetensors rebuilds a CNN from a Safetensors export. The architecture
// (stage count, channels, kernel, stride/pad) is inferred from the tensor
// shapes; classes from the head.
func FromSafetensors(ts map[string]*tensor.Tensor) (*CNN, error) {
	m := &CNN{relu: nn.ReLU()}
	for i := 0; ; i++ {
		w := ts[fmt.Sprintf("convs.%d.w", i)]
		if w == nil {
			break
		}
		b := ts[fmt.Sprintf("convs.%d.b", i)]
		if b == nil {
			return nil, fmt.Errorf("vision: missing convs.%d.b", i)
		}
		if w.Ndim() != 4 {
			return nil, fmt.Errorf("vision: convs.%d.w must be rank-4, got %v", i, w.Shape())
		}
		k := w.Shape()[2]
		m.Convs = append(m.Convs, &nn.Conv2D{W: w, B: b, Stride: 1, Pad: k / 2})
	}
	if len(m.Convs) == 0 {
		return nil, fmt.Errorf("vision: no conv stages found")
	}
	hw, hb := ts["head.w"], ts["head.b"]
	if hw == nil || hb == nil {
		return nil, fmt.Errorf("vision: missing head.w / head.b")
	}
	m.Head = &nn.Linear{W: hw, B: hb}
	return m, nil
}
