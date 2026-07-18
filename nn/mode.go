package nn

// Train/eval mode propagation (§T-mode). Some layers behave differently while
// training than at inference: [Dropout] zeros activations, [DropPath] drops whole
// residual branches. Both DEFAULT TO TRAINING MODE, and until this file existed
// the only way to switch one off was to hold the concrete *Dropout — so a user who
// composed a model out of [Sequential] had no generic way to reach a nested
// dropout, and it stayed active at inference, silently corrupting predictions.
//
// In plain terms: this file is the "put the whole model into inference mode"
// switch that every other framework spells model.eval().
//
// THE CORE [Layer] INTERFACE IS UNCHANGED. Layers opt in by implementing [Mode];
// containers opt in by implementing [Composite]. Anything that implements neither
// is simply skipped, so every existing layer keeps compiling and behaving exactly
// as before.

// Mode is a [Layer] whose behaviour differs between training and inference.
//
// It is an OPTIONAL interface — deliberately NOT part of [Layer], which would
// break every existing implementation. Detect support with a type assertion, or
// let [SetTrain] do it for you:
//
//	if m, ok := l.(nn.Mode); ok {
//		m.Eval()
//	}
//
// Implemented in this package by [Dropout], [DropPath] and [Sequential]. Note that
// the ScheduleFree OPTIMIZER also has Train/Eval methods with an unrelated meaning
// (weight-averaging phase, not layer behaviour); it is not a [Layer] and is never
// reached by the walk below.
type Mode interface {
	// Train switches the layer to training mode (stochastic behaviour ON).
	Train()
	// Eval switches the layer to inference mode (deterministic; for Dropout and
	// DropPath, the identity).
	Eval()
}

// Composite is a [Layer] that owns child layers and is willing to expose them, so
// that a generic walk such as [SetTrain] can descend into it.
//
// This is the SECOND optional interface, and it exists because of a hard Go limit:
// a walk cannot discover the sublayers of an arbitrary struct. Reflection can see
// unexported fields but cannot legally call methods on the values it reads out of
// them, and exported-field reflection would still miss layers held in maps, slices
// of concrete types or closures. Rather than fake it, the walk asks the type.
//
// In plain terms: if your model type is not a [Sequential], add four lines and the
// mode switch reaches your dropouts.
//
//	func (m *MyModel) Sublayers() []nn.Layer {
//		return []nn.Layer{m.Attn, m.Drop, m.MLP}
//	}
type Composite interface {
	// Sublayers returns the layers this one owns, in any order. The returned
	// slice is walked but not retained or mutated; nil entries are skipped.
	Sublayers() []Layer
}

// SetTrain puts root and every layer reachable from it into training mode
// (training = true) or inference mode (training = false). It is the generic form
// of model.train()/model.eval().
//
//	model := nn.NewSequential(lin, nn.ReLU(), nn.NewDropout(0.1, 7), head)
//	nn.SetTrain(model, false) // inference: dropout is now the identity
//	nn.SetTrain(model, true)  // back to training
//
// HOW FAR IT REACHES, EXACTLY. SetTrain visits root, then recurses into the
// children of any layer implementing [Composite] (which [Sequential] does), and
// calls Train/Eval on any layer implementing [Mode]. A layer that implements
// NEITHER interface is skipped without error — that is not a failure, most layers
// are mode-agnostic.
//
// THE DOCUMENTED LIMIT: a custom struct that OWNS layers but does NOT implement
// [Composite] is a dead end. The walk sees an opaque leaf, sets its own mode if it
// implements [Mode], and CANNOT reach the dropout inside it. Go offers no sound
// way around this — see [Composite]. If your model hides its sublayers, either
// implement Sublayers (four lines) or call Eval on the nested layers yourself.
//
// Cycles are not expected in a layer tree and are not detected; a layer graph that
// contains itself will recurse forever.
func SetTrain(root Layer, training bool) {
	if root == nil {
		return
	}
	if m, ok := root.(Mode); ok {
		if training {
			m.Train()
		} else {
			m.Eval()
		}
	}
	if c, ok := root.(Composite); ok {
		for _, sub := range c.Sublayers() {
			if sub != nil {
				SetTrain(sub, training)
			}
		}
	}
}

// Sublayers returns the chained layers, making [Sequential] walkable by
// [SetTrain].
func (s *Sequential) Sublayers() []Layer { return s.Layers }

// Train switches every layer in the chain (recursively, including nested
// [Sequential]s) to training mode. Layers that are not mode-dependent are skipped.
func (s *Sequential) Train() {
	for _, l := range s.Layers {
		if l != nil {
			SetTrain(l, true)
		}
	}
}

// Eval switches every layer in the chain (recursively, including nested
// [Sequential]s) to inference mode — the call you need before running predictions
// on a model that contains [Dropout] or [DropPath]. See [SetTrain] for the reach
// of the walk and its one documented limit.
func (s *Sequential) Eval() {
	for _, l := range s.Layers {
		if l != nil {
			SetTrain(l, false)
		}
	}
}

// Compile-time assertions for the mode-aware layers in this package.
var (
	_ Mode      = (*Dropout)(nil)
	_ Mode      = (*DropPath)(nil)
	_ Mode      = (*Sequential)(nil)
	_ Composite = (*Sequential)(nil)
)
