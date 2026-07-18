package safetensors_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/jxsl13/goai/format/safetensors"
	"github.com/jxsl13/goai/tensor"
)

// goldenIndex is the sharded checkpoint written by the REAL Hugging Face
// writer (transformers save_pretrained → 5 shards + index; regenerate with
// scripts/golden_sharded_safetensors.py), and goldenSingle the same model
// saved unsharded — the equality anchor.
const (
	goldenIndex  = "testdata/sharded/model.safetensors.index.json"
	goldenSingle = "testdata/sharded_single.safetensors"
)

// TestLoadShardedGoldenEquality: loading the transformers-sharded checkpoint
// through the index must be indistinguishable from loading the same model's
// single-file export — identical names, dtypes, shapes, every element equal,
// and identical metadata ({"format": "pt"} on both paths).
func TestLoadShardedGoldenEquality(t *testing.T) {
	got, gotMeta, err := safetensors.LoadSharded(goldenIndex)
	if err != nil {
		t.Fatalf("LoadSharded (run scripts/golden_sharded_safetensors.py): %v", err)
	}
	want, wantMeta, err := safetensors.LoadFile(goldenSingle)
	if err != nil {
		t.Fatalf("LoadFile single: %v", err)
	}
	if !reflect.DeepEqual(gotMeta, wantMeta) {
		t.Errorf("metadata: sharded %v != single %v", gotMeta, wantMeta)
	}
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("got %d tensors, want %d (non-zero)", len(got), len(want))
	}
	var nonUniform bool
	for name, w := range want {
		g := got[name]
		if g == nil {
			t.Fatalf("tensor %q missing from sharded load", name)
		}
		if g.Dtype() != w.Dtype() || !g.Shape().Equal(w.Shape()) {
			t.Fatalf("%s: %v %v != %v %v", name, g.Dtype(), g.Shape(), w.Dtype(), w.Shape())
		}
		first := w.AtF64(tensor.Unravel(0, w.Shape())...)
		for i := range w.Numel() {
			idx := tensor.Unravel(i, w.Shape())
			if g.AtF64(idx...) != w.AtF64(idx...) {
				t.Fatalf("%s[%v]: %v != %v", name, idx, g.AtF64(idx...), w.AtF64(idx...))
			}
			if w.AtF64(idx...) != first {
				nonUniform = true
			}
		}
	}
	if !nonUniform {
		t.Error("golden weights are uniform — a permutation/merge bug could hide (§B68 spirit)")
	}
}

// TestLoadShardedDeterministic: two loads of the same index return equal maps —
// same names, same values (idempotence; shard iteration order is sorted).
func TestLoadShardedDeterministic(t *testing.T) {
	a, am, err := safetensors.LoadSharded(goldenIndex)
	if err != nil {
		t.Fatal(err)
	}
	b, bm, err := safetensors.LoadSharded(goldenIndex)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(am, bm) {
		t.Errorf("metadata differs across loads: %v %v", am, bm)
	}
	if len(a) != len(b) {
		t.Fatalf("tensor count differs: %d %d", len(a), len(b))
	}
	for name, ta := range a {
		tb := b[name]
		if tb == nil || !ta.Shape().Equal(tb.Shape()) || ta.Dtype() != tb.Dtype() {
			t.Fatalf("%s differs across loads", name)
		}
		for i := range ta.Numel() {
			idx := tensor.Unravel(i, ta.Shape())
			if ta.AtF64(idx...) != tb.AtF64(idx...) {
				t.Fatalf("%s[%v] differs across loads", name, idx)
			}
		}
	}
}

// writeShardedFixture builds a minimal valid two-shard checkpoint in dir and
// returns the index path. mutate can rewrite the index object (and/or the
// shard files on disk) to produce each hostile variant.
func writeShardedFixture(t *testing.T, dir string, mutate func(index map[string]any)) string {
	t.Helper()
	shard1 := map[string]*tensor.Tensor{
		"a": tensor.FromFloat32(tensor.Shape{2}, []float32{1, 2}),
		"b": tensor.FromFloat32(tensor.Shape{3}, []float32{3, 4, 5}),
	}
	shard2 := map[string]*tensor.Tensor{
		"c": tensor.FromFloat64(tensor.Shape{2, 2}, []float64{6, 7, 8, 9}),
	}
	if err := safetensors.SaveFile(filepath.Join(dir, "model-00001-of-00002.safetensors"), shard1, nil); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.SaveFile(filepath.Join(dir, "model-00002-of-00002.safetensors"), shard2, nil); err != nil {
		t.Fatal(err)
	}
	index := map[string]any{
		"metadata": map[string]any{"total_size": 52},
		"weight_map": map[string]any{
			"a": "model-00001-of-00002.safetensors",
			"b": "model-00001-of-00002.safetensors",
			"c": "model-00002-of-00002.safetensors",
		},
	}
	if mutate != nil {
		mutate(index)
	}
	data, err := json.Marshal(index)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestLoadShardedValid: the Go-written two-shard fixture merges completely.
func TestLoadShardedValid(t *testing.T) {
	idx := writeShardedFixture(t, t.TempDir(), nil)
	ts, _, err := safetensors.LoadSharded(idx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ts) != 3 || ts["a"] == nil || ts["b"] == nil || ts["c"] == nil {
		t.Fatalf("merged map wrong: %v", ts)
	}
	if ts["c"].AtF64(1, 1) != 9 {
		t.Errorf("c(1,1) = %v", ts["c"].AtF64(1, 1))
	}
	// Index "metadata" (total_size etc.) is opaque bookkeeping — the HF loader
	// never validates it, and neither do we: a wrong total_size still loads.
	idx2 := writeShardedFixture(t, t.TempDir(), func(index map[string]any) {
		index["metadata"] = map[string]any{"total_size": 999999999}
	})
	if _, _, err := safetensors.LoadSharded(idx2); err != nil {
		t.Errorf("bogus total_size must not fail (HF ignores it): %v", err)
	}
	// Unknown top-level keys are tolerated (forward compatibility).
	idx3 := writeShardedFixture(t, t.TempDir(), func(index map[string]any) {
		index["future_field"] = []any{1, 2, 3}
	})
	if _, _, err := safetensors.LoadSharded(idx3); err != nil {
		t.Errorf("unknown top-level key must be tolerated: %v", err)
	}
}

// TestLoadShardedHostile: every malformed or hostile index errors — and names
// the offender — rather than mis-loading.
func TestLoadShardedHostile(t *testing.T) {
	wm := func(index map[string]any) map[string]any { return index["weight_map"].(map[string]any) }
	cases := []struct {
		name    string
		mutate  func(t *testing.T, dir string, index map[string]any)
		wantErr string // substring the error must contain
	}{
		{"path traversal", func(t *testing.T, dir string, index map[string]any) {
			wm(index)["a"] = "../evil.safetensors"
		}, "path separator"},
		{"absolute path", func(t *testing.T, dir string, index map[string]any) {
			wm(index)["a"] = "/etc/passwd"
		}, "path separator"},
		{"backslash traversal", func(t *testing.T, dir string, index map[string]any) {
			wm(index)["a"] = `..\evil.safetensors`
		}, "path separator"},
		{"dot-dot shard name", func(t *testing.T, dir string, index map[string]any) {
			wm(index)["a"] = ".."
		}, "invalid shard filename"},
		{"empty shard name", func(t *testing.T, dir string, index map[string]any) {
			wm(index)["a"] = ""
		}, "empty shard filename"},
		{"absent shard file", func(t *testing.T, dir string, index map[string]any) {
			wm(index)["zz"] = "model-00099-of-00099.safetensors"
		}, "model-00099-of-00099.safetensors"},
		{"listed but missing from shard", func(t *testing.T, dir string, index map[string]any) {
			wm(index)["ghost"] = "model-00002-of-00002.safetensors"
		}, `tensor "ghost" listed in weight_map is missing from shard "model-00002-of-00002.safetensors"`},
		{"in shard but unmapped", func(t *testing.T, dir string, index map[string]any) {
			delete(wm(index), "b")
		}, `tensor "b" in shard "model-00001-of-00002.safetensors" is absent from the index's weight_map`},
		{"duplicate across shards", func(t *testing.T, dir string, index map[string]any) {
			// "a" also exists inside shard 2's file on disk.
			if err := safetensors.SaveFile(filepath.Join(dir, "model-00002-of-00002.safetensors"), map[string]*tensor.Tensor{
				"c": tensor.FromFloat64(tensor.Shape{2, 2}, []float64{6, 7, 8, 9}),
				"a": tensor.FromFloat32(tensor.Shape{2}, []float32{0, 0}),
			}, nil); err != nil {
				t.Fatal(err)
			}
		}, `tensor "a" in shard "model-00002-of-00002.safetensors" is absent`},
		{"empty weight_map", func(t *testing.T, dir string, index map[string]any) {
			index["weight_map"] = map[string]any{}
		}, "weight_map missing or empty"},
		{"weight_map missing", func(t *testing.T, dir string, index map[string]any) {
			delete(index, "weight_map")
		}, "weight_map missing or empty"},
		{"non-string shard value", func(t *testing.T, dir string, index map[string]any) {
			wm(index)["a"] = 7
		}, "must be a string"},
		{"metadata conflict across shards", func(t *testing.T, dir string, index map[string]any) {
			mk := func(n int, meta map[string]string, ts map[string]*tensor.Tensor) {
				name := fmt.Sprintf("model-%05d-of-00002.safetensors", n)
				if err := safetensors.SaveFile(filepath.Join(dir, name), ts, meta); err != nil {
					t.Fatal(err)
				}
			}
			mk(1, map[string]string{"format": "pt"}, map[string]*tensor.Tensor{
				"a": tensor.FromFloat32(tensor.Shape{2}, []float32{1, 2}),
				"b": tensor.FromFloat32(tensor.Shape{3}, []float32{3, 4, 5}),
			})
			mk(2, map[string]string{"format": "np"}, map[string]*tensor.Tensor{
				"c": tensor.FromFloat64(tensor.Shape{2, 2}, []float64{6, 7, 8, 9}),
			})
		}, `metadata key "format" conflicts across shards`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			idx := writeShardedFixture(t, dir, func(index map[string]any) { c.mutate(t, dir, index) })
			_, _, err := safetensors.LoadSharded(idx)
			if err == nil {
				t.Fatalf("must error")
			}
			if !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("error %q does not contain %q", err, c.wantErr)
			}
		})
	}
}

// TestLoadShardedHostileRaw covers hostile cases that need a hand-written
// index byte stream (duplicate JSON keys, malformed JSON, trailing data) —
// json.Marshal cannot produce these.
func TestLoadShardedHostileRaw(t *testing.T) {
	shardEntry := `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}`
	cases := []struct {
		name, index, wantErr string
	}{
		{"malformed json", `{"weight_map": `, "weight_map"},
		{"truncated json", `{"metadata"`, "parse"},
		{"top-level array", `["weight_map"]`, "expected"},
		{"trailing data", `{"weight_map":{"a":"model-00001-of-00001.safetensors"}} {}`, "trailing data"},
		{"duplicate tensor key", `{"weight_map":{"a":"model-00001-of-00001.safetensors","a":"model-00001-of-00001.safetensors"}}`, `duplicate tensor "a"`},
		{"duplicate weight_map", `{"weight_map":{"a":"model-00001-of-00001.safetensors"},"weight_map":{"a":"model-00001-of-00001.safetensors"}}`, "duplicate weight_map"},
		{"NUL in shard name", `{"weight_map":{"a":"bad\\u0000name"}}`, "NUL"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			// A valid one-tensor shard so only the index is at fault.
			var buf []byte
			buf = append(buf, byte(len(shardEntry)), 0, 0, 0, 0, 0, 0, 0)
			buf = append(buf, shardEntry...)
			buf = append(buf, make([]byte, 8)...)
			if err := os.WriteFile(filepath.Join(dir, "model-00001-of-00001.safetensors"), buf, 0o644); err != nil {
				t.Fatal(err)
			}
			idx := filepath.Join(dir, "model.safetensors.index.json")
			if err := os.WriteFile(idx, []byte(c.index), 0o644); err != nil {
				t.Fatal(err)
			}
			_, _, err := safetensors.LoadSharded(idx)
			if err == nil {
				t.Fatalf("must error")
			}
			if !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("error %q does not contain %q", err, c.wantErr)
			}
		})
	}

	if _, _, err := safetensors.LoadSharded(filepath.Join(t.TempDir(), "nope.index.json")); err == nil {
		t.Error("missing index file must error")
	}
}

// LoadSharded reads a Hugging Face sharded checkpoint through its
// model.safetensors.index.json: the index's weight_map says which shard file
// holds each tensor, and the merged result is what LoadFile would have
// returned for a single-file checkpoint.
func ExampleLoadSharded() {
	dir, _ := os.MkdirTemp("", "sharded")
	defer os.RemoveAll(dir)

	// A checkpoint split across two shards, as transformers writes it.
	_ = safetensors.SaveFile(filepath.Join(dir, "model-00001-of-00002.safetensors"),
		map[string]*tensor.Tensor{"embed": tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})}, nil)
	_ = safetensors.SaveFile(filepath.Join(dir, "model-00002-of-00002.safetensors"),
		map[string]*tensor.Tensor{"head": tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4})}, nil)
	_ = os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(`{
		"metadata": {"total_size": 32},
		"weight_map": {
			"embed": "model-00001-of-00002.safetensors",
			"head":  "model-00002-of-00002.safetensors"
		}
	}`), 0o644)

	tensors, _, err := safetensors.LoadSharded(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		panic(err)
	}
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	fmt.Println(names, tensors["head"].AtF64(1))
	// Output: [embed head] 4
}
