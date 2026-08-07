//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// TestLMTrainStepOnGPU trains a complete miniature language-model step ENTIRELY on the GPU and checks
// the cross-entropy loss collapses — the end-to-end proof of the whole LM training pipeline on device:
//
//	x     = Embed(tokens, E)                 (input embedding)
//	h     = RMSNorm(x, g1)
//	a     = softmax((h·Wq)(h·Wk)ᵀ/√d)·(h·Wv) (attention)
//	y     = x + a·Wo                          (residual)
//	logits= y·Wout
//	loss  = cross_entropy(logits, targets)
//
// Backward runs on device through the LM loss (cross-entropy), the head, the attention block, and the
// input embedding; DeviceAdam updates E, Wq,Wk,Wv,Wo, g1, Wout. The loss must drop as the model memorizes
// the fixed token→target mapping.
func TestLMTrainStepOnGPU(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const V, L, D, steps = 32, 8, 32, 900
	const eps = 1e-5
	scale := float32(1.0 / math.Sqrt(float64(D)))
	rng := rand.New(rand.NewSource(31))

	must := func(err error) {
		if err != nil {
			t.Fatal(err)
		}
	}
	rm := func(r, c int, f func() float32) *cuda.DeviceF32 {
		d, err := cuda.NewDeviceF32(r, c)
		must(err)
		h := make([]float32, r*c)
		for i := range h {
			h[i] = f()
		}
		must(d.UploadF32(h))
		return d
	}
	n := func(r, c int) *cuda.DeviceF32 { d, err := cuda.NewDeviceF32(r, c); must(err); return d }
	small := func() float32 { return float32(rng.NormFloat64() * 0.1) }

	tokens := make([]int32, L)
	targets := make([]int32, L)
	for i := range tokens {
		tokens[i] = int32(rng.Intn(V))
		targets[i] = int32(rng.Intn(V))
	}

	E := rm(V, D, small)
	Wq, Wk, Wv, Wo := rm(D, D, small), rm(D, D, small), rm(D, D, small), rm(D, D, small)
	g1 := rm(1, D, func() float32 { return 1 })
	Wout := rm(D, V, small)
	params := []*cuda.DeviceF32{E, Wq, Wk, Wv, Wo, g1, Wout}
	for _, p := range params {
		defer p.Free()
	}

	x := n(L, D)
	h := n(L, D)
	Q, K, V2, S := n(L, D), n(L, D), n(L, D), n(L, L)
	a, y := n(L, D), n(L, D)
	logits := n(L, V)
	dlogits := n(L, V)
	dY, dA := n(L, D), n(L, D)
	dQ, dK, dV := n(L, D), n(L, D), n(L, D)
	dh, tmp, dx := n(L, D), n(L, D), n(L, D)
	dE := n(V, D)
	dWq, dWk, dWv, dWo := n(D, D), n(D, D), n(D, D), n(D, D)
	dg1 := n(1, D)
	dWout := n(D, V)
	grads := []*cuda.DeviceF32{dE, dWq, dWk, dWv, dWo, dg1, dWout}
	for _, b := range append([]*cuda.DeviceF32{x, h, Q, K, V2, S, a, y, logits, dlogits, dY, dA, dQ, dK, dV, dh, tmp, dx}, grads...) {
		defer b.Free()
	}

	opt, err := cuda.NewDeviceAdam([]int{V * D, D * D, D * D, D * D, D * D, D, D * V}, 1e-2, 0.9, 0.999, 1e-8, 0)
	must(err)
	defer opt.Free()

	hostLg := make([]float32, L*V)
	loss := func() float64 {
		must(logits.DownloadF32(hostLg))
		var l float64
		for i := 0; i < L; i++ {
			mx := math.Inf(-1)
			for j := 0; j < V; j++ {
				if v := float64(hostLg[i*V+j]); v > mx {
					mx = v
				}
			}
			var z float64
			for j := 0; j < V; j++ {
				z += math.Exp(float64(hostLg[i*V+j]) - mx)
			}
			l += -(float64(hostLg[i*V+int(targets[i])]) - mx - math.Log(z))
		}
		return l / float64(L)
	}

	forward := func() {
		must(cuda.EmbedForward(x, E, tokens))
		must(cuda.RMSNormForward(h, x, g1, eps))
		must(cuda.MatMul(h, Wq, Q))
		must(cuda.MatMul(h, Wk, K))
		must(cuda.MatMul(h, Wv, V2))
		must(cuda.MatMulGradX(Q, K, S)) // S = Q·Kᵀ
		must(cuda.Scale(S, scale))
		must(S.Softmax())
		must(cuda.MatMul(S, V2, a))
		must(cuda.MatMul(a, Wo, y))
		must(y.Add(x)) // residual
		must(cuda.MatMul(y, Wout, logits))
	}

	forward()
	l0 := loss()
	for step := 0; step < steps; step++ {
		forward()
		must(cuda.CrossEntropyBackward(dlogits, logits, targets, 1.0/float32(L)))
		// head
		must(cuda.MatMulGradX(dlogits, Wout, dY)) // dY = dlogits·Woutᵀ
		must(cuda.MatMulGradW(y, dlogits, dWout)) // dWout = yᵀ·dlogits
		// o-proj (y = x + a·Wo): daProj = dY
		must(cuda.MatMulGradX(dY, Wo, dA))
		must(cuda.MatMulGradW(a, dY, dWo))
		// attention
		must(cuda.AttentionBackward(dQ, dK, dV, Q, K, V2, dA, scale))
		// QKV proj
		must(cuda.MatMulGradX(dQ, Wq, dh))
		must(cuda.MatMulGradX(dK, Wk, tmp))
		must(dh.Add(tmp))
		must(cuda.MatMulGradX(dV, Wv, tmp))
		must(dh.Add(tmp))
		must(cuda.MatMulGradW(h, dQ, dWq))
		must(cuda.MatMulGradW(h, dK, dWk))
		must(cuda.MatMulGradW(h, dV, dWv))
		// RMSNorm1: dx = norm_input_grad, then add the residual grad dY -> total dx into the embedding
		must(cuda.RMSNormBackward(dx, dg1, x, dh, g1, eps))
		must(dx.Add(dY))
		// embedding
		must(cuda.EmbedBackward(dE, dx, tokens))
		must(opt.Step(params, grads))
	}
	forward()
	l1 := loss()

	t.Logf("LM training step on GPU: cross-entropy %.4e -> %.4e over %d steps (all on device)", l0, l1, steps)
	if l1 > l0*0.25 {
		t.Fatalf("LM step did not train on GPU: CE %.4e -> %.4e (want < %.4e)", l0, l1, l0*0.25)
	}
}
