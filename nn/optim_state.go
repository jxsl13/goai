package nn

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// Optimizer state checkpointing (§T-resume). A model checkpoint alone is NOT
// enough to resume an interrupted training run: every adaptive optimizer carries
// per-parameter state (SGD's velocity, Adam's m/v moments, Adafactor's factored
// row/column moments, Muon's momentum buffer) plus a step counter that drives
// bias correction. Restarting from the weights alone silently restarts that state
// at zero, and the resumed trajectory diverges from the uninterrupted one — a
// measured 1566% of one full step for Adam interrupted at step 5 of 10.
//
// In plain terms: this file lets you write the optimizer's "memory" to disk next
// to the model weights, and read it back so training continues exactly where it
// stopped.
//
// COVERAGE IS PARTIAL AND DELIBERATE. Optimizers opt in by implementing
// [StatefulOptimizer]; the ones wired today are [SGD], [Adam] (and therefore
// AdamW, the same type), [Lion], [Adafactor] and [Muon]. The remaining optimizers
// in this package (LAMB, Shampoo, SOAP, Sophia, AdEMAMix, Schedule-Free, Prodigy,
// D-Adapt-Adam, Adam-mini, MARS, PSGD-Kron, Q-GaLore, Grokfast-MA and the
// Lookahead/SAM/cautious/GaLore/APOLLO wrappers) do NOT implement it yet and will
// fail the type assertion — see TestOptimizerStateCoverage in the tests, which
// lists the wired and unwired sets so the gap stays visible. Extending one is
// mechanical: mirror the State/LoadState pair below.

// StatefulOptimizer is an [Optimizer] whose internal state can be captured and
// restored, which is what makes an interrupted training run resumable.
//
// It is an OPTIONAL interface: the core [Optimizer] interface is unchanged, so
// optimizers opt in one at a time and no existing code breaks. Detect support
// with a type assertion:
//
//	if so, ok := opt.(nn.StatefulOptimizer); ok {
//		err := nn.SaveOptimizerState("opt.safetensors", so)
//	}
//
// PARAMETER IDENTITY IS BY INDEX. Optimizer state is per-parameter, but after a
// restart the caller rebuilds the model, so the parameter POINTERS differ and
// cannot be used as keys. State is therefore keyed by the parameter's position in
// the optimizer's Params slice. THE CALLER MUST REBUILD THE MODEL WITH IDENTICAL
// STRUCTURE and construct the optimizer over the parameters in the same order.
// LoadState validates what it can — parameter count and per-parameter shapes —
// and returns an error naming the mismatch, but it cannot detect a permutation of
// same-shaped parameters. Reuse the same model-construction code path.
type StatefulOptimizer interface {
	Optimizer

	// State returns a snapshot of the optimizer's internal state as named
	// tensors, safe to serialize. Buffers are copied, so the returned map is
	// decoupled from further Step calls. Per-parameter buffers are keyed
	// "<name>/<index>"; scalars (step counters and the like) are stored as
	// 1-element F64 tensors under a bare name. Every implementation includes
	// "param_count" so a structural mismatch is caught on load.
	State() map[string]*tensor.Tensor

	// LoadState restores state previously produced by State. It validates the
	// parameter count, every per-parameter shape, and rejects unknown keys;
	// on any error the optimizer is left unchanged.
	LoadState(map[string]*tensor.Tensor) error
}

// --- serialization ---

// SaveOptimizerState writes opt's state to path in safetensors format — the
// repo's checkpoint format — so an optimizer checkpoint sits next to the model
// checkpoint and both can be loaded with the same tooling.
//
// In plain terms: save the optimizer's memory alongside your weights, so a crash
// costs you at most the steps since the last save rather than the whole run.
func SaveOptimizerState(path string, opt StatefulOptimizer) error {
	st := opt.State()
	if err := safetensors.SaveFile(path, st, map[string]string{"format": "goai-optimizer-state"}); err != nil {
		return fmt.Errorf("nn: save optimizer state %q: %w", path, err)
	}
	return nil
}

// LoadOptimizerState reads a checkpoint written by [SaveOptimizerState] into opt.
// opt must be freshly constructed over the rebuilt model's parameters, in the
// same order as the run that saved the file (see [StatefulOptimizer]); a mismatch
// in parameter count or shape is reported as an error and opt is left unchanged.
func LoadOptimizerState(path string, opt StatefulOptimizer) error {
	ts, _, err := safetensors.LoadFile(path)
	if err != nil {
		return fmt.Errorf("nn: load optimizer state %q: %w", path, err)
	}
	if err := opt.LoadState(ts); err != nil {
		return fmt.Errorf("nn: load optimizer state %q: %w", path, err)
	}
	return nil
}

// --- shared state plumbing ---

// stateKey formats the per-parameter key "<name>/<index>".
func stateKey(name string, i int) string { return name + "/" + strconv.Itoa(i) }

// stateWriter accumulates an optimizer's state map. Nil buffers are skipped, so a
// disabled feature (SGD without momentum, Adafactor without a first moment) leaves
// no keys behind — and loading such a file into a differently-configured optimizer
// then trips the unknown-key check.
type stateWriter struct{ m map[string]*tensor.Tensor }

func newStateWriter(nParams int) *stateWriter {
	w := &stateWriter{m: make(map[string]*tensor.Tensor)}
	w.scalar("param_count", float64(nParams))
	return w
}

// scalar stores a scalar as a 1-element F64 tensor. Scalars ride in the SAME map
// as the buffers (rather than a parallel map[string]float64) so that the whole
// state round-trips through one safetensors file with no second channel, and so a
// checkpoint is self-describing to any safetensors reader.
func (w *stateWriter) scalar(name string, v float64) {
	w.m[name] = tensor.FromFloat64(tensor.Shape{1}, []float64{v})
}

// buf stores one per-parameter buffer, shaped so the checkpoint is inspectable and
// shape validation on load is direct.
func (w *stateWriter) buf(name string, i int, shape tensor.Shape, data []float64) {
	if data == nil {
		return
	}
	w.m[stateKey(name, i)] = tensor.FromFloat64(shape, append([]float64(nil), data...))
}

func (w *stateWriter) done() map[string]*tensor.Tensor { return w.m }

// stateReader reads a state map, tracking which keys were consumed so anything
// left over can be reported as an unknown key (a wrong-optimizer or
// wrong-configuration checkpoint). It latches the first error; callers check once
// via finish, having staged their writes so nothing is mutated before then.
type stateReader struct {
	src  map[string]*tensor.Tensor
	used map[string]bool
	err  error
}

// newStateReader validates the parameter count up front — the single most likely
// structural mismatch after a model rebuild.
func newStateReader(src map[string]*tensor.Tensor, nParams int) *stateReader {
	r := &stateReader{src: src, used: make(map[string]bool, len(src))}
	got, ok := r.scalar("param_count")
	if !ok {
		return r
	}
	if int(got) != nParams {
		r.fail(fmt.Errorf("param count mismatch: checkpoint has %d parameters, optimizer has %d "+
			"(rebuild the model with identical structure)", int(got), nParams))
	}
	return r
}

func (r *stateReader) fail(err error) {
	if r.err == nil {
		r.err = err
	}
}

// scalar reads a 1-element scalar tensor.
func (r *stateReader) scalar(name string) (float64, bool) {
	t, ok := r.src[name]
	if !ok {
		r.fail(fmt.Errorf("missing key %q", name))
		return 0, false
	}
	r.used[name] = true
	if t.Numel() != 1 {
		r.fail(fmt.Errorf("key %q: expected a 1-element scalar, got shape %v", name, t.Shape()))
		return 0, false
	}
	return t.AtF64(tensor.Unravel(0, t.Shape())...), true
}

// buf reads one per-parameter buffer into dst, requiring the recorded shape to
// match want exactly. A nil dst means "this optimizer has no such buffer" — the
// key must then be absent, which the unknown-key sweep in finish enforces.
func (r *stateReader) buf(name string, i int, want tensor.Shape, dst []float64) {
	if dst == nil {
		return
	}
	k := stateKey(name, i)
	t, ok := r.src[k]
	if !ok {
		r.fail(fmt.Errorf("missing key %q (parameter %d)", k, i))
		return
	}
	r.used[k] = true
	if !t.Shape().Equal(want) {
		r.fail(fmt.Errorf("key %q: shape mismatch, checkpoint has %v, parameter %d needs %v",
			//perfscan:ignore PS3013 pointer-to-fmt in r.fail error path; cold
			k, t.Shape(), i, want))
		return
	}
	//perfscan:ignore PS1001 per-element AtF64 in checkpoint LoadState; one-time cold path
	for j := range dst {
		dst[j] = t.AtF64(tensor.Unravel(j, t.Shape())...)
	}
}

// finish reports the latched error, or an unknown-key error if the checkpoint
// carried state this optimizer does not have (a different optimizer, or the same
// one configured differently — e.g. momentum enabled when saving, disabled now).
func (r *stateReader) finish() error {
	if r.err != nil {
		return r.err
	}
	var extra []string
	for k := range r.src {
		if !r.used[k] {
			extra = append(extra, k)
		}
	}
	if len(extra) > 0 {
		sort.Strings(extra)
		return fmt.Errorf("unknown key(s) in optimizer state: %v "+
			"(checkpoint was written by a different optimizer or configuration)", extra)
	}
	return nil
}

// --- SGD ---

// State captures SGD's momentum velocity buffers ("vel/<i>", one per parameter).
// Plain SGD (Momentum == 0) keeps no buffers, so its state is just the parameter
// count — saving and loading it is a no-op that still validates structure.
func (s *SGD) State() map[string]*tensor.Tensor {
	w := newStateWriter(len(s.Params))
	for i, p := range s.Params {
		if s.vel != nil {
			w.buf("vel", i, p.Shape(), s.vel[i])
		}
	}
	return w.done()
}

// LoadState restores SGD's velocity buffers. See [StatefulOptimizer] for the
// identical-structure requirement.
func (s *SGD) LoadState(st map[string]*tensor.Tensor) error {
	r := newStateReader(st, len(s.Params))
	staged := make([][]float64, len(s.Params))
	for i, p := range s.Params {
		if s.vel != nil {
			//perfscan:ignore PS2008,PS3064 alloc in SGD LoadState; checkpoint load one-time | staged rows in SGD LoadState; checkpoint one-time
			staged[i] = make([]float64, p.Numel())
			r.buf("vel", i, p.Shape(), staged[i])
		}
	}
	if err := r.finish(); err != nil {
		return err
	}
	for i := range staged {
		if staged[i] != nil {
			copy(s.vel[i], staged[i])
		}
	}
	return nil
}

// --- Adam / AdamW ---

// State captures Adam's first and second moments ("m/<i>", "v/<i>") and the step
// counter "t". The step counter matters as much as the moments: it drives the
// bias correction 1−βᵗ, so dropping it makes the first resumed steps far too
// large. AdamW is the same type and is covered by the same pair.
func (a *Adam) State() map[string]*tensor.Tensor {
	w := newStateWriter(len(a.Params))
	w.scalar("t", float64(a.t))
	for i, p := range a.Params {
		w.buf("m", i, p.Shape(), a.m[i])
		w.buf("v", i, p.Shape(), a.v[i])
	}
	return w.done()
}

// LoadState restores Adam's moments and step counter.
func (a *Adam) LoadState(st map[string]*tensor.Tensor) error {
	r := newStateReader(st, len(a.Params))
	t, _ := r.scalar("t")
	m := make([][]float64, len(a.Params))
	v := make([][]float64, len(a.Params))
	for i, p := range a.Params {
		//perfscan:ignore PS2008,PS3064 alloc in Adam LoadState; checkpoint one-time | row make in Adam LoadState; checkpoint one-time
		m[i] = make([]float64, p.Numel())
		//perfscan:ignore PS2008,PS3064 alloc in Adam LoadState; checkpoint one-time | row make in Adam LoadState; checkpoint one-time
		v[i] = make([]float64, p.Numel())
		r.buf("m", i, p.Shape(), m[i])
		r.buf("v", i, p.Shape(), v[i])
	}
	if err := r.finish(); err != nil {
		return err
	}
	a.t = int(t)
	for i := range m {
		copy(a.m[i], m[i])
		copy(a.v[i], v[i])
	}
	return nil
}

// --- Lion ---

// State captures Lion's single momentum buffer ("m/<i>"). Lion has no step
// counter (no bias correction), so the moments are the whole story.
func (l *Lion) State() map[string]*tensor.Tensor {
	w := newStateWriter(len(l.Params))
	for i, p := range l.Params {
		w.buf("m", i, p.Shape(), l.m[i])
	}
	return w.done()
}

// LoadState restores Lion's momentum buffers.
func (l *Lion) LoadState(st map[string]*tensor.Tensor) error {
	r := newStateReader(st, len(l.Params))
	m := make([][]float64, len(l.Params))
	for i, p := range l.Params {
		//perfscan:ignore PS2008,PS3064 alloc in Lion LoadState; checkpoint one-time | row make in Lion LoadState; checkpoint one-time
		m[i] = make([]float64, p.Numel())
		r.buf("m", i, p.Shape(), m[i])
	}
	if err := r.finish(); err != nil {
		return err
	}
	for i := range m {
		copy(l.m[i], m[i])
	}
	return nil
}

// --- Muon ---

// State captures Muon's momentum buffers ("buf/<i>"), shaped like the (2-D)
// parameters. The Newton-Schulz orthogonalization is stateless, so the momentum
// is the complete state.
func (mu *Muon) State() map[string]*tensor.Tensor {
	w := newStateWriter(len(mu.Params))
	for i, p := range mu.Params {
		w.buf("buf", i, p.Shape(), mu.buf[i])
	}
	return w.done()
}

// LoadState restores Muon's momentum buffers.
func (mu *Muon) LoadState(st map[string]*tensor.Tensor) error {
	r := newStateReader(st, len(mu.Params))
	b := make([][]float64, len(mu.Params))
	for i, p := range mu.Params {
		//perfscan:ignore PS2008,PS3064 alloc in Muon LoadState; checkpoint one-time | row make in Muon LoadState; checkpoint one-time
		b[i] = make([]float64, p.Numel())
		r.buf("buf", i, p.Shape(), b[i])
	}
	if err := r.finish(); err != nil {
		return err
	}
	for i := range b {
		copy(mu.buf[i], b[i])
	}
	return nil
}

// --- Adafactor ---

// State captures Adafactor's factored second moments ("r/<i>" of shape [rows] and
// "c/<i>" of shape [cols]) for rank-2 parameters, the full second moment ("v/<i>")
// for the others, the optional first moment ("m/<i>", present only when Beta1 > 0)
// and the step counter "t" that drives the 1/√t relative-step schedule.
//
// Because the key SET depends on configuration, loading an Adafactor checkpoint
// into a differently-configured Adafactor (momentum on vs off) is rejected as an
// unknown-key or missing-key error rather than silently half-restored.
func (a *Adafactor) State() map[string]*tensor.Tensor {
	w := newStateWriter(len(a.Params))
	w.scalar("t", float64(a.t))
	for i, p := range a.Params {
		if p.Ndim() == 2 {
			w.buf("r", i, tensor.Shape{p.Shape()[0]}, a.r[i])
			w.buf("c", i, tensor.Shape{p.Shape()[1]}, a.c[i])
		} else {
			w.buf("v", i, p.Shape(), a.v[i])
		}
		if a.m != nil {
			w.buf("m", i, p.Shape(), a.m[i])
		}
	}
	return w.done()
}

// LoadState restores Adafactor's factored/full second moments, optional first
// moment and step counter.
func (a *Adafactor) LoadState(st map[string]*tensor.Tensor) error {
	r := newStateReader(st, len(a.Params))
	rr := make([][]float64, len(a.Params))
	cc := make([][]float64, len(a.Params))
	vv := make([][]float64, len(a.Params))
	mm := make([][]float64, len(a.Params))
	t, _ := r.scalar("t")
	for i, p := range a.Params {
		if p.Ndim() == 2 {
			//perfscan:ignore PS2008,PS3064 alloc in Adafactor LoadState; checkpoint one-time | row make in Adafactor LoadState; checkpoint one-time
			rr[i] = make([]float64, p.Shape()[0])
			//perfscan:ignore PS2008,PS3064 alloc in Adafactor LoadState; checkpoint one-time | row make in Adafactor LoadState; checkpoint one-time
			cc[i] = make([]float64, p.Shape()[1])
			r.buf("r", i, tensor.Shape{p.Shape()[0]}, rr[i])
			r.buf("c", i, tensor.Shape{p.Shape()[1]}, cc[i])
		} else {
			//perfscan:ignore PS2008,PS3064 alloc in Adafactor LoadState; checkpoint one-time | row make in Adafactor LoadState; checkpoint one-time
			vv[i] = make([]float64, p.Numel())
			r.buf("v", i, p.Shape(), vv[i])
		}
		if a.m != nil {
			//perfscan:ignore PS2008,PS3064 alloc in Adafactor LoadState; checkpoint one-time | row make in Adafactor LoadState; checkpoint one-time
			mm[i] = make([]float64, p.Numel())
			r.buf("m", i, p.Shape(), mm[i])
		}
	}
	if err := r.finish(); err != nil {
		return err
	}
	a.t = int(t)
	for i := range a.Params {
		if rr[i] != nil {
			copy(a.r[i], rr[i])
			copy(a.c[i], cc[i])
		}
		if vv[i] != nil {
			copy(a.v[i], vv[i])
		}
		if mm[i] != nil {
			copy(a.m[i], mm[i])
		}
	}
	return nil
}

// Compile-time assertions that the wired optimizers satisfy the optional
// interface (the coverage test enumerates the unwired ones).
var (
	_ StatefulOptimizer = (*SGD)(nil)
	_ StatefulOptimizer = (*Adam)(nil)
	_ StatefulOptimizer = (*Lion)(nil)
	_ StatefulOptimizer = (*Muon)(nil)
	_ StatefulOptimizer = (*Adafactor)(nil)
)
