package llamagpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// StepHidden advances the decoder by one token like Step and ADDITIONALLY returns the
// final hidden state [dim] (the residual stream after the final LayerNorm — the row the
// LM head projects, still resident in the decoder's scratch after the step). This is
// the attachment point for auxiliary decoding heads (§T446, Medusa): the hidden row
// costs one extra small download, no extra GPU work.
func (d *GPTDecoder) StepHidden(token, pos int) (logits, hidden []float32, err error) {
	logits, err = d.Step(token, pos)
	if err != nil {
		return nil, nil, err
	}
	hidden = make([]float32, d.d)
	if err := d.xn.b.DownloadF32(hidden); err != nil {
		return nil, nil, err
	}
	return logits, hidden, nil
}

// StepNHidden advances the decoder by k tokens like StepN and ADDITIONALLY returns
// the k final hidden rows [k·dim] (§T455): the rows are still resident in the
// decoder's scratch after the step, so the extra cost is one small download. This is
// what lets Medusa draft the NEXT window from the current verify pass — one batched
// step per round instead of two.
func (d *GPTDecoder) StepNHidden(tokens []int, pos int) (logits, hidden []float32, err error) {
	logits, err = d.StepN(tokens, pos)
	if err != nil {
		return nil, nil, err
	}
	hidden = make([]float32, len(tokens)*d.d)
	if err := d.xn.b.DownloadF32(hidden); err != nil {
		return nil, nil, err
	}
	return logits, hidden, nil
}

// StepHidden is the Llama decoder's sibling (§T447): Step plus the final RMSNorm'd
// hidden row still resident in the decoder's scratch.
func (d *Decoder) StepHidden(token, pos int) (logits, hidden []float32, err error) {
	logits, err = d.Step(token, pos)
	if err != nil {
		return nil, nil, err
	}
	hidden = make([]float32, d.d)
	if err := d.xn.b.DownloadF32(hidden); err != nil {
		return nil, nil, err
	}
	return logits, hidden, nil
}

// StepNHidden is the Llama decoder's sibling of GPTDecoder.StepNHidden (§T455).
func (d *Decoder) StepNHidden(tokens []int, pos int) (logits, hidden []float32, err error) {
	logits, err = d.StepN(tokens, pos)
	if err != nil {
		return nil, nil, err
	}
	hidden = make([]float32, len(tokens)*d.d)
	if err := d.xn.b.DownloadF32(hidden); err != nil {
		return nil, nil, err
	}
	return logits, hidden, nil
}

// HiddenStepper is a Stepper that can also expose the final hidden state(s) of a
// step — the hook auxiliary decoding heads need. Both batched decoders ([Decoder]
// and [GPTDecoder], Metal or Vulkan) satisfy it.
type HiddenStepper interface {
	Stepper
	StepHidden(token, pos int) (logits, hidden []float32, err error)
	StepNHidden(tokens []int, pos int) (logits, hidden []float32, err error)
}

var (
	_ HiddenStepper = (*Decoder)(nil)
	_ HiddenStepper = (*GPTDecoder)(nil)
)

// MedusaGenerate decodes with Medusa chain drafting on a batched decoder (§T446/§T447/
// §T455, the GPU sibling of nlp.MedusaGenerate; works for both the Llama and the GPT
// decoder via HiddenStepper). It uses the §T419 lastTok-lead-window convention so a
// round is ONE batched StepNHidden: the window [lastTok, x̂₂…x̂_{K+1}] is verified in a
// single step whose logits accept the longest nlp.TypicalAcceptance(ε,δ)-prefix and
// supply the greedy bonus token, while its HIDDEN rows let the K heads draft the NEXT
// window's proposals host-side ([dim,vocab] projections, negligible). KV rollback is
// free — the position-addressed cache overwrites rejected rows on the next round.
// Compared to draft-model speculative decoding the drafting cost is ~zero (the §T434
// lesson: the draft's own steps ate the win on dispatch-bound decoders); a round costs
// ~1 step for up to K+1 tokens. ε/δ ≤ 0 select the reference defaults. Greedy-anchored,
// not distribution-exact (typical acceptance). Heads must be trained for THIS base
// model (nlp.MedusaHeads on its rollouts/corpus, against its ForwardHidden) and match
// its dim/vocab.
func MedusaGenerate(dec HiddenStepper, heads *nlp.MedusaHeads, prompt []int, maxNew int, epsilon, delta float64) ([]int, nlp.SpecStats, error) {
	var stats nlp.SpecStats
	if len(prompt) == 0 {
		return nil, stats, fmt.Errorf("llamagpu: MedusaGenerate needs a non-empty prompt")
	}
	if heads == nil || len(heads.W) == 0 {
		return nil, stats, fmt.Errorf("llamagpu: MedusaGenerate needs ≥1 Medusa head")
	}
	for k, w := range heads.W {
		if w.Ndim() != 2 || w.Shape()[1] != dec.Vocab() {
			return nil, stats, fmt.Errorf("llamagpu: Medusa head %d shape %v != [dim,%d]", k, w.Shape(), dec.Vocab())
		}
		if w.Dtype() != tensor.F32 {
			return nil, stats, fmt.Errorf("llamagpu: Medusa head %d dtype %v, want F32 (the decoder is f32)", k, w.Dtype())
		}
	}
	if epsilon <= 0 {
		epsilon = nlp.MedusaEpsilon
	}
	if delta <= 0 {
		delta = nlp.MedusaDelta
	}
	out := append([]int(nil), prompt...)
	pos := 0
	if len(prompt) > 1 { // prefill all but the last token in one batched step
		if _, err := dec.StepN(prompt[:len(prompt)-1], 0); err != nil {
			return nil, stats, err
		}
		pos = len(prompt) - 1
	}
	lastTok := prompt[len(prompt)-1]
	dim := heads.W[0].Shape()[0]
	v := dec.Vocab()
	gen := 0
	var proposals []int // drafted by the heads from the PREVIOUS round's hidden rows
	for gen < maxNew && pos < dec.Ctx() {
		// window = lead token + proposals, clamped to context and budget (each round emits
		// at most len(window) tokens: the accepted proposals + the greedy bonus).
		if room := min(maxNew-gen, dec.Ctx()-pos) - 1; len(proposals) > room {
			if room < 0 {
				room = 0
			}
			proposals = proposals[:room]
		}
		window := append([]int{lastTok}, proposals...)
		vlog, vhid, err := dec.StepNHidden(window, pos)
		if err != nil {
			return nil, stats, err
		}
		if len(vhid) != len(window)*dim {
			return nil, stats, fmt.Errorf("llamagpu: Medusa heads dim %d != decoder hidden %d", dim, len(vhid)/len(window))
		}
		// accept the longest typical prefix of the proposals; row j-1 is the base's
		// distribution after window[..j-1], scoring proposal window[j].
		m := 0
		stats.Proposed += len(window) - 1
		for j := 1; j < len(window); j++ {
			probs := softmaxF64(vlog[(j-1)*v : j*v])
			if !nlp.TypicalAcceptance(probs, window[j], epsilon, delta) {
				break
			}
			m++
			stats.Accepted++
		}
		// greedy bonus after the last verified position, then draft the next round's
		// proposals from that position's hidden row (head k predicts offset +2+k, i.e.
		// the tokens FOLLOWING the bonus — the §T419 lead convention keeps this exact).
		bonus := argmaxF32(vlog[m*v : (m+1)*v])
		emit := append(append([]int(nil), window[1:1+m]...), bonus)
		if rem := maxNew - gen; len(emit) > rem {
			emit = emit[:rem]
		}
		out = append(out, emit...)
		gen += len(emit)
		h := vhid[m*dim : (m+1)*dim]
		proposals = proposals[:0]
		for _, w := range heads.W {
			proposals = append(proposals, headArgmax(h, w))
		}
		pos += m + 1
		lastTok = bonus
	}
	return out, stats, nil
}

// headArgmax computes argmax_v hidden·W[:,v] host-side — one [dim,vocab] projection.
func headArgmax(hidden []float32, w *tensor.Tensor) int {
	wd := w.Storage().F32()
	dim, vocab := w.Shape()[0], w.Shape()[1]
	best, bestV := 0, float32(math.Inf(-1))
	for v := range vocab {
		var s float32
		for i := range dim {
			s += hidden[i] * wd[i*vocab+v]
		}
		if s > bestV {
			best, bestV = v, s
		}
	}
	return best
}

func argmaxF32(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

// softmaxF64 converts one f32 logit row to a stable f64 probability distribution.
func softmaxF64(logits []float32) []float64 {
	mx := math.Inf(-1)
	for _, v := range logits {
		if float64(v) > mx {
			mx = float64(v)
		}
	}
	var sum float64
	p := make([]float64, len(logits))
	for i, v := range logits {
		p[i] = math.Exp(float64(v) - mx)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p
}
