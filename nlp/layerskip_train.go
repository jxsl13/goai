package nlp

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// LayerSkipTrainConfig configures the LayerSkip training recipe from Elhoushi
// et al. (ACL 2024, §4.1). Its zero value is deliberately conservative: no
// layer dropout, no intermediate-exit weight, all exits enabled, and therefore
// [Llama.LayerSkipLoss] collapses exactly to ordinary final-layer cross-entropy.
//
// MaxDropout is pmax and ExitScale is escale; both must be in [0,1]. Rotation R
// enables every R-th post-block exit and circularly advances the mask with Step;
// 0 means R=1 (all exits). FromScratch enables the paper's time curriculum and
// then requires TotalSteps > 0. Seed and Step make the one-sample Bernoulli mask
// reproducible, including after resuming training.
//
// GoAI's Llama forward API represents one sequence rather than a rank-3 batch,
// so the paper's per-sample dropout is applied to that whole sequence. Call the
// method once per sample (or accumulated microbatch) to obtain per-sample masks.
type LayerSkipTrainConfig struct {
	MaxDropout float64 // pmax, the final block's maximum dropout probability
	ExitScale  float64 // escale, the relative mass of intermediate-exit losses
	Rotation   int     // R, the spacing between enabled exits; 0 means every exit
	Step       int     // zero-based training step, used by both curricula
	TotalSteps int     // T for the from-scratch time curriculum

	FromScratch bool   // enable S(t)=2^(t/T)-1 instead of the finetuning constant 1
	Seed        uint64 // first word of the reproducible per-step PCG seed
}

// DropoutRates returns p(l,t)=S(t)D(l)pmax for every zero-based block. D(l)
// rises exponentially from 0 at block 0 to 1 at the final block. S(t)=1 for
// finetuning/continual pretraining; from-scratch training uses 2^(t/T)-1,
// clamped to the completed-training interval.
func (c LayerSkipTrainConfig) DropoutRates(layers int) ([]float64, error) {
	if err := c.validate(layers); err != nil {
		return nil, err
	}
	rates := make([]float64, layers)
	if layers == 1 || c.MaxDropout == 0 {
		return rates, nil
	}
	timeScale := 1.0
	if c.FromScratch {
		step := min(c.Step, c.TotalSteps)
		timeScale = math.Exp2(float64(step)/float64(c.TotalSteps)) - 1
	}
	for layer := range layers {
		depthScale := math.Exp2(float64(layer)/float64(layers-1)) - 1
		rates[layer] = timeScale * depthScale * c.MaxDropout
	}
	return rates, nil
}

// ExitWeights returns the normalized paper Eq.6 weights for the exits enabled
// by the rotational curriculum at Step. Intermediate block l has raw weight
// ExitScale*sum(i=0..l)i; the final block has the ordinary LM-loss term L-1
// plus ExitScale*sum(i=0..L-2)i. Disabled exits have weight zero.
func (c LayerSkipTrainConfig) ExitWeights(layers int) ([]float64, error) {
	if err := c.validate(layers); err != nil {
		return nil, err
	}
	weights := make([]float64, layers)
	if layers == 1 {
		weights[0] = 1
		return weights, nil
	}
	rotation := c.rotation()
	phase := c.Step % rotation
	var sum float64
	for layer := range layers {
		if layer%rotation != phase {
			continue
		}
		var raw float64
		if layer == layers-1 {
			raw = float64(layers-1) + c.ExitScale*triangular(layers-2)
		} else {
			raw = c.ExitScale * triangular(layer)
		}
		weights[layer] = raw
		sum += raw
	}
	if sum == 0 {
		return nil, fmt.Errorf("nlp: LayerSkip rotation %d at step %d enables no positive-weight exit across %d layers", rotation, c.Step, layers)
	}
	for layer := range weights {
		weights[layer] /= sum
	}
	return weights, nil
}

func (c LayerSkipTrainConfig) rotation() int {
	if c.Rotation == 0 {
		return 1
	}
	return c.Rotation
}

func (c LayerSkipTrainConfig) validate(layers int) error {
	if layers <= 0 {
		return fmt.Errorf("nlp: LayerSkip needs at least one transformer block")
	}
	if c.MaxDropout < 0 || c.MaxDropout > 1 {
		return fmt.Errorf("nlp: LayerSkip max dropout %.6g outside [0,1]", c.MaxDropout)
	}
	if c.ExitScale < 0 || c.ExitScale > 1 {
		return fmt.Errorf("nlp: LayerSkip exit scale %.6g outside [0,1]", c.ExitScale)
	}
	if c.Step < 0 {
		return fmt.Errorf("nlp: LayerSkip step %d must be non-negative", c.Step)
	}
	rotation := c.rotation()
	if rotation < 1 || rotation > layers {
		return fmt.Errorf("nlp: LayerSkip rotation %d outside [1,%d]", rotation, layers)
	}
	if c.FromScratch && c.TotalSteps <= 0 {
		return fmt.Errorf("nlp: LayerSkip from-scratch curriculum needs positive total steps")
	}
	return nil
}

func triangular(last int) float64 {
	if last <= 0 {
		return 0
	}
	return float64(last*(last+1)) / 2
}

// LayerSkipLoss runs one differentiable LayerSkip training forward and returns
// the normalized shared-head early-exit cross-entropy. Each block is either
// executed through Llama's ordinary block path or skipped as an identity for the
// whole input sequence according to cfg.DropoutRates. Enabled exits reuse the
// model's final RMSNorm and output projection; no auxiliary parameters are added.
//
// The target tensor follows [nn.CrossEntropy]: one class index per token row.
// A zero cfg executes every block and scores only the final exit, making the
// result and its gradients bit-identical to Forward followed by CrossEntropy.
func (m *Llama) LayerSkipLoss(ctx *backend.Context, tokens []int, targets *tensor.Tensor, cfg LayerSkipTrainConfig) (*tensor.Tensor, error) {
	if m == nil {
		return nil, fmt.Errorf("nlp: Llama.LayerSkipLoss needs a model")
	}
	layers := len(m.Blocks)
	if m.Config.Layers != layers {
		return nil, fmt.Errorf("nlp: Llama.LayerSkipLoss config has %d layers but model has %d blocks", m.Config.Layers, layers)
	}
	if targets == nil {
		return nil, fmt.Errorf("nlp: Llama.LayerSkipLoss needs targets")
	}
	if err := validateLayerBackends(m.LayerBackends, layers); err != nil {
		return nil, err
	}
	rates, err := cfg.DropoutRates(layers)
	if err != nil {
		return nil, err
	}
	weights, err := cfg.ExitWeights(layers)
	if err != nil {
		return nil, err
	}
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = scaleScalar(ctx, x, m.Config.EmbeddingMult); err != nil {
		return nil, err
	}

	rng := rand.New(rand.NewPCG(cfg.Seed, uint64(cfg.Step)+1))
	var total *tensor.Tensor
	for layer := range m.Blocks {
		if rng.Float64() >= rates[layer] {
			if x, err = m.forwardBlock(ctx, x, layer, nil); err != nil {
				return nil, err
			}
		}
		if weights[layer] == 0 {
			continue
		}
		logits, err := m.Unembed(ctx, x)
		if err != nil {
			return nil, err
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			return nil, err
		}
		loss, err = layerSkipScale(ctx, loss, weights[layer])
		if err != nil {
			return nil, err
		}
		if total == nil {
			total = loss
			continue
		}
		if total, err = exec1(ctx, backend.OpAdd, nil, total, loss); err != nil {
			return nil, err
		}
	}
	if total == nil {
		return nil, fmt.Errorf("nlp: LayerSkip produced no enabled exit loss")
	}
	return total, nil
}

func layerSkipScale(ctx *backend.Context, loss *tensor.Tensor, weight float64) (*tensor.Tensor, error) {
	if weight == 1 {
		return loss, nil
	}
	scale := tensor.New(loss.Dtype(), tensor.Shape{})
	scale.SetF64(weight)
	return exec1(ctx, backend.OpMul, nil, loss, scale)
}
