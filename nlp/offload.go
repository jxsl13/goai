package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
)

// transformerLayerOps is the complete primitive surface used inside GPT and
// Llama-family blocks, including optional LoRA/bias/QK-norm/Granite paths. A
// per-layer offload assignment routes this whole set together so an overflow
// layer does not bounce back to the model's primary device between sublayers.
var transformerLayerOps = []backend.Op{
	backend.OpAdd,
	backend.OpMul,
	backend.OpAXPY,
	backend.OpMatMul,
	backend.OpAddBias,
	backend.OpGELU,
	backend.OpSiLU,
	backend.OpLayerNorm,
	backend.OpRMSNorm,
	backend.OpRoPE,
	backend.OpMHA,
	backend.OpMHAMasked,
	backend.OpMHASelect,
	backend.OpReshape,
	backend.OpSoftCap,
}

func validateLayerBackends(names []backend.Name, layers int) error {
	if len(names) == 0 {
		return nil
	}
	if len(names) != layers {
		return fmt.Errorf("nlp: offload plan has %d layer backends, want %d", len(names), layers)
	}
	for layer, name := range names {
		if name == "" {
			return fmt.Errorf("nlp: offload plan layer %d has an empty backend", layer)
		}
	}
	return nil
}

func contextForLayer(ctx *backend.Context, names []backend.Name, layer int) *backend.Context {
	if len(names) == 0 {
		return ctx
	}
	return ctx.WithLayerBackend(names[layer], transformerLayerOps...)
}
