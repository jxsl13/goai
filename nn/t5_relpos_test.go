package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1 (arXiv:1910.10683 §2.1 / HF _relative_position_bucket, bidirectional 32/128 ⇒
// half=16, max_exact=8): direction offsets positive relatives by 16; small |rel| are exact; large
// ones are log-spaced and clamped to 15. Values computed independently from the HF algorithm.
func TestT5BucketBidirectional(t *testing.T) {
	for _, c := range []struct {
		rel, want int
	}{
		{0, 0}, {-3, 3}, {-7, 7}, {-8, 8}, {-100, 15}, // negative side: bucket = |rel| (exact) or log
		{3, 19}, {7, 23}, {8, 24}, {100, 31}, // positive side: + 16
	} {
		if got := nn.T5RelativePositionBucket(c.rel, true, 32, 128); got != c.want {
			t.Errorf("bucket(%d, bidi) = %d, want %d", c.rel, got, c.want)
		}
	}
}

// §V16 tier-1 (causal 32/128 ⇒ max_exact=16): future positions (rel>0) collapse to bucket 0; past
// positions bucket by −rel, exact then log-spaced (clamped to 31).
func TestT5BucketCausal(t *testing.T) {
	for _, c := range []struct {
		rel, want int
	}{
		{5, 0}, {0, 0}, {100, 0}, // future & self → 0
		{-5, 5}, {-15, 15}, {-16, 16}, {-100, 30}, // past → exact then log
	} {
		if got := nn.T5RelativePositionBucket(c.rel, false, 32, 128); got != c.want {
			t.Errorf("bucket(%d, causal) = %d, want %d", c.rel, got, c.want)
		}
	}
}

// §V16 tier-1 (clamp): distances beyond max_distance saturate at the last bucket of their half.
func TestT5BucketClamp(t *testing.T) {
	if got := nn.T5RelativePositionBucket(1_000_000, true, 32, 128); got != 31 {
		t.Errorf("very large positive bidi bucket = %d, want 31", got)
	}
	if got := nn.T5RelativePositionBucket(-1_000_000, false, 32, 128); got != 31 {
		t.Errorf("very large past causal bucket = %d, want 31", got)
	}
}

// §V16 tier-1 (bucket matrix): entry [i][j] = bucket(j−i); the diagonal (rel 0) is bucket 0.
func TestT5BucketMatrix(t *testing.T) {
	m, err := nn.T5RelativePositionBuckets(3, 3, true, 32, 128)
	if err != nil {
		t.Fatal(err)
	}
	// [0][0]=bucket(0)=0; [0][1]=bucket(1)=17 (+16); [1][0]=bucket(-1)=1; [0][2]=bucket(2)=18.
	want := [][]int{{0, 17, 18}, {1, 0, 17}, {2, 1, 0}}
	for i := range m {
		for j := range m[i] {
			if m[i][j] != want[i][j] {
				t.Errorf("matrix[%d][%d]=%d, want %d", i, j, m[i][j], want[i][j])
			}
		}
	}
}

// §V2: defaults (numBuckets/maxDistance ≤ 0 → 32/128).
func TestT5BucketDefaults(t *testing.T) {
	def, err := nn.T5RelativePositionBuckets(4, 4, true, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	ref, _ := nn.T5RelativePositionBuckets(4, 4, true, nn.T5DefaultNumBuckets, nn.T5DefaultMaxDistance)
	for i := range def {
		for j := range def[i] {
			if def[i][j] != ref[i][j] {
				t.Fatalf("default at [%d][%d] differs", i, j)
			}
		}
	}
}

func TestT5BucketErrors(t *testing.T) {
	if _, err := nn.T5RelativePositionBuckets(3, 3, true, 33, 128); err == nil {
		t.Error("odd numBuckets: want error")
	}
	if _, err := nn.T5RelativePositionBuckets(0, 3, true, 32, 128); err == nil {
		t.Error("non-positive length: want error")
	}
	if _, err := nn.T5RelativePositionBuckets(3, 3, true, 2, 128); err == nil {
		t.Error("too-small numBuckets: want error")
	}
}

func ExampleT5RelativePositionBucket() {
	// Adjacent key one step ahead of the query (rel=+1) under the bidirectional scheme: bucket 17
	// (the positive-direction half starts at 16, plus the exact distance 1).
	fmt.Println(nn.T5RelativePositionBucket(1, true, 32, 128))
	// Output:
	// 17
}

// §V16 tier-1 (§R232, zero init): a fresh bias adds nothing until trained.
func TestT5BiasZeroInit(t *testing.T) {
	b, err := nn.NewT5RelativeBias(32, 2, 128, true, tensor.F64)
	if err != nil {
		t.Fatal(err)
	}
	bias, err := b.Bias(backend.NewContext(), 4, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i := range bias.Numel() {
		if bias.AtF64(tensor.Unravel(i, bias.Shape())...) != 0 {
			t.Fatal("zero-init bias should be all zero")
		}
	}
}

// §V16 tier-1 (gather correctness): Bias[i][j][h] = Table[bucket(j−i)][h] — the module gathers the
// learned per-head bias by relative-position bucket (the point of §R232 wired up).
func TestT5BiasGather(t *testing.T) {
	const nb, nh = 32, 2
	b, _ := nn.NewT5RelativeBias(nb, nh, 128, true, tensor.F64)
	for k := 0; k < nb; k++ {
		for h := 0; h < nh; h++ {
			b.Table.SetF64(float64(k*10+h), k, h) // distinct per (bucket, head)
		}
	}
	bias, err := b.Bias(backend.NewContext(), 3, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !bias.Shape().Equal(tensor.Shape{3, 3, nh}) {
		t.Fatalf("bias shape %v, want [3 3 2]", bias.Shape())
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			bucket := nn.T5RelativePositionBucket(j-i, true, nb, 128)
			for h := 0; h < nh; h++ {
				if got, want := bias.AtF64(i, j, h), float64(bucket*10+h); got != want {
					t.Errorf("Bias[%d][%d][%d]=%g, want Table[bucket %d][%d]=%g", i, j, h, got, bucket, h, want)
				}
			}
		}
	}
}

// §V-GRAD: finite-difference check of the gradient into the Table (accumulated over the pairs each
// bucket covers).
func TestT5BiasGradCheck(t *testing.T) {
	b, _ := nn.NewT5RelativeBias(8, 2, 16, false, tensor.F64) // small causal table
	for k := 0; k < 8; k++ {
		for h := 0; h < 2; h++ {
			b.Table.SetF64(0.1*float64(k)-0.2*float64(h), k, h)
		}
	}
	sumLoss := func(ctx *backend.Context) (*tensor.Tensor, error) {
		bias, err := b.Bias(ctx, 5, 5)
		if err != nil {
			return nil, err
		}
		return exec1(ctx, backend.OpSum, backend.ReduceAttrs{Axes: []int{0, 1, 2}}, bias)
	}
	tape := autograd.NewTape()
	loss, err := sumLoss(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(b.Table)
	if g == nil {
		t.Fatal("nil table gradient")
	}
	const eps = 1e-6
	for i := range b.Table.Numel() {
		c := tensor.Unravel(i, b.Table.Shape())
		o := b.Table.AtF64(c...)
		b.Table.SetF64(o+eps, c...)
		lp, _ := sumLoss(backend.NewContext())
		b.Table.SetF64(o-eps, c...)
		lm, _ := sumLoss(backend.NewContext())
		b.Table.SetF64(o, c...)
		if num, ana := (lp.AtF64()-lm.AtF64())/(2*eps), g.AtF64(c...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("table grad %v: numeric %.8g vs analytic %.8g", c, num, ana)
		}
	}
}

func TestT5BiasErrors(t *testing.T) {
	if _, err := nn.NewT5RelativeBias(32, 0, 128, true, tensor.F64); err == nil {
		t.Error("numHeads 0: want error")
	}
	if _, err := nn.NewT5RelativeBias(33, 2, 128, true, tensor.F64); err == nil {
		t.Error("odd numBuckets: want error")
	}
}

func ExampleT5RelativeBias() {
	b, _ := nn.NewT5RelativeBias(32, 1, 128, true, tensor.F64)
	b.Table.SetF64(2.5, nn.T5RelativePositionBucket(1, true, 32, 128), 0) // bias for rel=+1
	bias, _ := b.Bias(backend.NewContext(), 2, 2)
	fmt.Printf("%.1f\n", bias.AtF64(0, 1, 0)) // key one step ahead of query
	// Output:
	// 2.5
}

// ExampleT5RelativeBias_BiasRow shows the incremental-decode row: the bias a single query at an
// absolute position sees over all keys, without building the rows before it. It prints the value
// alongside the same entry taken from the full table, which is what "bit-identical to Bias sliced
// at that row" means in practice.
func ExampleT5RelativeBias_BiasRow() {
	b, _ := nn.NewT5RelativeBias(32, 1, 128, true, tensor.F64)
	b.Table.SetF64(2.5, nn.T5RelativePositionBucket(1, true, 32, 128), 0) // bias for rel=+1

	ctx := backend.NewContext()
	row, _ := b.BiasRow(ctx, 0, 2) // query at position 0, keys 0..1
	full, _ := b.Bias(ctx, 2, 2)   // the whole [2,2] table
	fmt.Printf("%.1f %.1f\n", row.AtF64(1, 0), full.AtF64(0, 1, 0))
	// Output:
	// 2.5 2.5
}
