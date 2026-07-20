package safetensors_test

import (
	"math"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// mkSharded writes a checkpoint split across multiple shards and returns the
// index path. WithMaxShardSize forces the split so a tensor genuinely lives in a
// different shard from its neighbours.
func mkSharded(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	m := map[string]*tensor.Tensor{}
	// Four ~4 MiB f32 tensors: with a 6 MiB shard cap they land in separate shards.
	for _, name := range []string{"layer0.weight", "layer1.weight", "layer2.weight", "layer3.weight"} {
		x := tensor.Zeros(tensor.F32, tensor.Shape{1 << 20})
		x.SetF64(float64(len(name)), 0) // a distinguishing nonzero
		m[name] = x
	}
	if err := safetensors.SaveSharded(dir, m, map[string]string{"fmt": "goai"}, safetensors.WithMaxShardSize(6<<20)); err != nil {
		t.Fatal(err)
	}
	return filepath.Join(dir, "model.safetensors.index.json")
}

// LoadShardedTensor must return each tensor bit-identical to LoadSharded's copy.
func TestLoadShardedTensorBitIdentical(t *testing.T) {
	idx := mkSharded(t)
	all, _, err := safetensors.LoadSharded(idx)
	if err != nil {
		t.Fatal(err)
	}
	for name, whole := range all {
		one, err := safetensors.LoadShardedTensor(idx, name)
		if err != nil {
			t.Fatalf("LoadShardedTensor(%q): %v", name, err)
		}
		if one.Dtype() != whole.Dtype() || !one.Shape().Equal(whole.Shape()) {
			t.Fatalf("%s: dtype/shape mismatch", name)
		}
		n := whole.Numel()
		for i := range n {
			idx2 := tensor.Unravel(i, whole.Shape())
			if math.Float64bits(one.AtF64(idx2...)) != math.Float64bits(whole.AtF64(idx2...)) {
				t.Errorf("%s elem %d differs", name, i)
				break
			}
		}
	}
}

// ShardedNames lists every tensor across shards, header-only, matching LoadSharded.
func TestShardedNames(t *testing.T) {
	idx := mkSharded(t)
	all, _, err := safetensors.LoadSharded(idx)
	if err != nil {
		t.Fatal(err)
	}
	infos, meta, err := safetensors.ShardedNames(idx)
	if err != nil {
		t.Fatal(err)
	}
	if meta["fmt"] != "goai" {
		t.Errorf("metadata not merged: %v", meta)
	}
	if len(infos) != len(all) {
		t.Fatalf("ShardedNames returned %d tensors, LoadSharded has %d", len(infos), len(all))
	}
	for _, in := range infos {
		w, ok := all[in.Name]
		if !ok {
			t.Errorf("ShardedNames reported %q not in LoadSharded", in.Name)
			continue
		}
		if in.Dtype != w.Dtype() || !in.Shape.Equal(w.Shape()) {
			t.Errorf("%s: info %v/%v vs %v/%v", in.Name, in.Dtype, in.Shape, w.Dtype(), w.Shape())
		}
	}
}

// The point: pulling one tensor from a multi-shard checkpoint reads only its
// shard, so it allocates far less than the whole model.
func TestLoadShardedTensorReadsOneShard(t *testing.T) {
	idx := mkSharded(t)
	var m0, m1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m0)
	x, err := safetensors.LoadShardedTensor(idx, "layer2.weight")
	runtime.ReadMemStats(&m1)
	if err != nil {
		t.Fatal(err)
	}
	if x.AtF64(0) != float64(len("layer2.weight")) {
		t.Errorf("layer2.weight[0] = %v", x.AtF64(0))
	}
	// The whole model is ~16 MiB across 4 shards; one 4 MiB tensor must cost far
	// less than materializing all of it.
	if got := (m1.TotalAlloc - m0.TotalAlloc) >> 20; got > 12 {
		t.Errorf("LoadShardedTensor allocated %d MiB — it did not read a single shard", got)
	}
}

func TestLoadShardedTensorErrors(t *testing.T) {
	idx := mkSharded(t)
	if _, err := safetensors.LoadShardedTensor(idx, "nope.weight"); err == nil {
		t.Error("want error for a tensor absent from the index")
	}
	if _, _, err := safetensors.ShardedNames("/no/such/index.json"); err == nil {
		t.Error("want error for a missing index")
	}
}
