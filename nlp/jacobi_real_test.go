package nlp_test

import (
	"testing"

	"github.com/jxsl13/goai/nlp"
)

// §T441: JacobiGenerate on a REAL model is lossless — token-for-token equal to the model's own
// sequential greedy Generate — and converges in at most genLen parallel iterations.
func TestJacobiGenerateMatchesGreedy(t *testing.T) {
	model, prompt := loadGPTModel(t)
	prompt = prompt[:2]
	genLen := model.Config.Ctx - len(prompt)
	got, iters, err := model.JacobiGenerate(prompt, genLen, 100)
	if err != nil {
		t.Fatal(err)
	}
	want, err := model.Generate(prompt, genLen, nlp.Greedy())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("length %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token[%d]: jacobi %d != greedy %d (iters=%d)", i, got[i], want[i], iters)
		}
	}
	if iters < 1 || iters > genLen {
		t.Fatalf("iterations %d out of [1,%d]", iters, genLen)
	}
	t.Logf("jacobi == sequential greedy: %d tokens in %d parallel iterations", genLen, iters)
}

// §T441: context clamping and the empty-prompt error path.
func TestJacobiGenerateBounds(t *testing.T) {
	model, prompt := loadGPTModel(t)
	if _, _, err := model.JacobiGenerate(nil, 4, 10); err == nil {
		t.Fatal("empty prompt accepted")
	}
	out, _, err := model.JacobiGenerate(prompt, 10*model.Config.Ctx, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != model.Config.Ctx {
		t.Fatalf("clamped length %d, want ctx %d", len(out), model.Config.Ctx)
	}
}
