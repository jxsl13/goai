package safetensors

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// shardedWriteFixture returns a non-uniform mixed-dtype checkpoint (§B68
// spirit: no all-equal tensors) whose per-tensor data sizes are, in sorted
// name order: alpha 48, beta 40, delta.bias 16, delta.weight 12, eps 16,
// eta 28, gamma 40, zeta 24 — 224 data bytes total. The sizes are chosen so
// specific WithMaxShardSize values force 1, 2, and 7 shards.
func shardedWriteFixture() map[string]*tensor.Tensor {
	return map[string]*tensor.Tensor{
		"alpha":        tensor.FromFloat32(tensor.Shape{4, 3}, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
		"beta":         tensor.FromFloat64(tensor.Shape{5}, []float64{-1, 0.5, 2.25, -3.75, 1e-3}),
		"gamma":        tensor.FromFloat32(tensor.Shape{10}, []float32{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}),
		"delta.bias":   f16Tensor(tensor.F16, tensor.Shape{8}, []uint16{0x3C00, 0x4000, 0x4200, 0xBC00, 0x0001, 0x7BFF, 0x8000, 0x3555}),
		"delta.weight": f16Tensor(tensor.BF16, tensor.Shape{6}, []uint16{0x3F80, 0x4000, 0xC000, 0x0080, 0x7F7F, 0x3EAB}),
		"eps":          tensor.FromFloat32(tensor.Shape{2, 2}, []float32{1.5, -2.5, 3.5, -4.5}),
		"zeta":         tensor.FromFloat64(tensor.Shape{3}, []float64{10, 20, 30}),
		"eta":          tensor.FromFloat32(tensor.Shape{7}, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}),
	}
}

// fixtureDataBytes is Σ numel×dtype-size over shardedWriteFixture — the
// total_size the index must carry (data bytes, NOT file bytes).
const fixtureDataBytes = 224

// assertTensorsEqual fails unless got and want hold identical tensors:
// same names, dtypes, shapes, every element equal.
func assertTensorsEqual(t *testing.T, got, want map[string]*tensor.Tensor) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d tensors, want %d", len(got), len(want))
	}
	for name, w := range want {
		g := got[name]
		if g == nil {
			t.Fatalf("tensor %q missing", name)
		}
		if g.Dtype() != w.Dtype() || !g.Shape().Equal(w.Shape()) {
			t.Fatalf("%s: %v %v != %v %v", name, g.Dtype(), g.Shape(), w.Dtype(), w.Shape())
		}
		for i := range w.Numel() {
			idx := tensor.Unravel(i, w.Shape())
			if g.AtF64(idx...) != w.AtF64(idx...) {
				t.Fatalf("%s[%v]: %v != %v", name, idx, g.AtF64(idx...), w.AtF64(idx...))
			}
		}
	}
}

// TestSaveShardedRoundTrip: LoadSharded(SaveSharded(x)) == x tensor-for-tensor
// including metadata, across budgets forcing 1, 2, and 7 shards. The
// single-shard case mirrors HF's fallback: plain model.safetensors, NO index,
// loaded with LoadFile.
func TestSaveShardedRoundTrip(t *testing.T) {
	tensors := shardedWriteFixture()
	meta := map[string]string{"format": "pt", "origin": "goai"}
	cases := []struct {
		name       string
		max        int64
		wantShards int // 1 means the single-file fallback
	}{
		{"single", 1 << 30, 1},
		{"two", 120, 2},
		{"seven", 40, 7},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := SaveSharded(dir, tensors, meta, WithMaxShardSize(c.max)); err != nil {
				t.Fatal(err)
			}
			indexPath := filepath.Join(dir, "model.safetensors.index.json")
			if c.wantShards == 1 {
				if _, err := os.Stat(indexPath); !os.IsNotExist(err) {
					t.Fatalf("single-shard save must not write an index (HF fallback): %v", err)
				}
				got, gotMeta, err := LoadFile(filepath.Join(dir, "model.safetensors"))
				if err != nil {
					t.Fatal(err)
				}
				assertTensorsEqual(t, got, tensors)
				if !reflect.DeepEqual(gotMeta, meta) {
					t.Errorf("metadata: %v != %v", gotMeta, meta)
				}
				return
			}
			got, gotMeta, err := LoadSharded(indexPath)
			if err != nil {
				t.Fatal(err)
			}
			assertTensorsEqual(t, got, tensors)
			if !reflect.DeepEqual(gotMeta, meta) {
				t.Errorf("metadata: %v != %v", gotMeta, meta)
			}
			var raw struct {
				WeightMap map[string]string `json:"weight_map"`
			}
			data, err := os.ReadFile(indexPath)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(data, &raw); err != nil {
				t.Fatal(err)
			}
			shardSet := map[string]bool{}
			for _, f := range raw.WeightMap {
				shardSet[f] = true
			}
			if len(shardSet) != c.wantShards {
				t.Errorf("got %d shards, want %d (weight_map %v)", len(shardSet), c.wantShards, raw.WeightMap)
			}
		})
	}
}

// TestSaveShardedNaming: shard files carry HF's exact suffix format
// "-{idx+1:05d}-of-{nb_shards:05d}" — five-digit zero-padding on BOTH numbers
// — the index is <base>.safetensors.index.json, and WithBaseName renames the
// whole family.
func TestSaveShardedNaming(t *testing.T) {
	dir := t.TempDir()
	if err := SaveSharded(dir, shardedWriteFixture(), nil, WithMaxShardSize(40)); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"model-00001-of-00007.safetensors",
		"model-00002-of-00007.safetensors",
		"model-00003-of-00007.safetensors",
		"model-00004-of-00007.safetensors",
		"model-00005-of-00007.safetensors",
		"model-00006-of-00007.safetensors",
		"model-00007-of-00007.safetensors",
		"model.safetensors.index.json",
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, e.Name())
	}
	sort.Strings(names)
	if !reflect.DeepEqual(names, want) {
		t.Errorf("files %v, want %v", names, want)
	}

	dir2 := t.TempDir()
	if err := SaveSharded(dir2, shardedWriteFixture(), nil, WithMaxShardSize(120), WithBaseName("ckpt")); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"ckpt-00001-of-00002.safetensors", "ckpt-00002-of-00002.safetensors", "ckpt.safetensors.index.json"} {
		if _, err := os.Stat(filepath.Join(dir2, f)); err != nil {
			t.Errorf("expected %s: %v", f, err)
		}
	}
	if _, _, err := LoadSharded(filepath.Join(dir2, "ckpt.safetensors.index.json")); err != nil {
		t.Errorf("renamed family must load: %v", err)
	}
}

// TestSaveShardedOversizedOwnShard pins the exact HF branch structure for a
// tensor bigger than the budget (split_state_dict_into_shards_factory): it
// becomes its own oversized shard appended at its position WITHOUT closing
// the shard being packed — so with sorted order a, b(big), c and a budget
// fitting a+c, the result is shard 1 = {b}, shard 2 = {a, c}.
func TestSaveShardedOversizedOwnShard(t *testing.T) {
	tensors := map[string]*tensor.Tensor{
		"a": tensor.FromFloat32(tensor.Shape{4}, []float32{1, 2, 3, 4}), // 16 B
		"b": tensor.FromFloat32(tensor.Shape{25}, make([]float32, 25)),  // 100 B > budget
		"c": tensor.FromFloat32(tensor.Shape{4}, []float32{5, 6, 7, 8}), // 16 B
	}
	for i := range 25 {
		tensors["b"].SetF64(float64(i)*0.5, i)
	}
	dir := t.TempDir()
	if err := SaveSharded(dir, tensors, nil, WithMaxShardSize(48)); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &idx); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"b": "model-00001-of-00002.safetensors",
		"a": "model-00002-of-00002.safetensors",
		"c": "model-00002-of-00002.safetensors",
	}
	if !reflect.DeepEqual(idx.WeightMap, want) {
		t.Errorf("weight_map %v, want %v (HF oversized-tensor branch)", idx.WeightMap, want)
	}
	got, _, err := LoadSharded(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}
	assertTensorsEqual(t, got, tensors)
}

// TestSaveShardedIndexFormat: the emitted index carries exactly the HF key
// shapes — top-level {"metadata", "weight_map"}, metadata.total_size = Σ
// tensor DATA bytes (headers excluded — asserted strictly below file size) —
// shard names match the golden's pattern, and the bytes carry the
// transformers serialization style (2-space indent, sorted keys, trailing
// newline).
func TestSaveShardedIndexFormat(t *testing.T) {
	tensors := shardedWriteFixture()
	dir := t.TempDir()
	if err := SaveSharded(dir, tensors, map[string]string{"format": "pt"}, WithMaxShardSize(120)); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(dir, "model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}

	// Byte style: transformers json.dumps(index, indent=2, sort_keys=True)+"\n".
	if !strings.HasPrefix(string(data), "{\n  \"metadata\": {\n    \"total_size\": ") {
		t.Errorf("index prefix %q lacks the transformers layout", data[:min(len(data), 40)])
	}
	if !strings.HasSuffix(string(data), "}\n") {
		t.Errorf("index must end with closing brace + newline, got %q", data[len(data)-4:])
	}

	// Key shapes: exactly the golden's top-level keys and metadata value type.
	var top map[string]json.RawMessage
	if err := json.Unmarshal(data, &top); err != nil {
		t.Fatal(err)
	}
	if len(top) != 2 || top["metadata"] == nil || top["weight_map"] == nil {
		t.Fatalf("top-level keys wrong: %v", top)
	}
	var md map[string]int64
	if err := json.Unmarshal(top["metadata"], &md); err != nil {
		t.Fatal(err)
	}
	if len(md) != 1 || md["total_size"] != fixtureDataBytes {
		t.Errorf("metadata %v, want {total_size: %d} (data bytes, not file bytes)", md, fixtureDataBytes)
	}
	var wm map[string]string
	if err := json.Unmarshal(top["weight_map"], &wm); err != nil {
		t.Fatal(err)
	}
	if len(wm) != len(tensors) {
		t.Errorf("weight_map has %d entries, want %d", len(wm), len(tensors))
	}
	// total_size must be below the summed FILE sizes (files add headers).
	var fileBytes int64
	pat := regexp.MustCompile(`^model-\d{5}-of-\d{5}\.safetensors$`)
	for _, f := range wm {
		if !pat.MatchString(f) {
			t.Errorf("shard name %q does not match the HF pattern", f)
		}
		fi, err := os.Stat(filepath.Join(dir, f))
		if err != nil {
			t.Fatal(err)
		}
		fileBytes += fi.Size()
	}
	if md["total_size"] >= fileBytes {
		t.Errorf("total_size %d must be data bytes only, below file bytes %d", md["total_size"], fileBytes)
	}
}

// TestMarshalIndexMatchesTransformersGolden: re-serializing the parsed T835
// golden index (written by the REAL transformers save_pretrained) through
// this package's index marshaler reproduces the golden's bytes exactly —
// proof that SaveSharded's index serialization is byte-compatible with
// transformers' json.dumps(index, indent=2, sort_keys=True) + "\n".
func TestMarshalIndexMatchesTransformersGolden(t *testing.T) {
	want, err := os.ReadFile(goldenIndexInternal)
	if err != nil {
		t.Fatalf("read golden (run scripts/golden_sharded_safetensors.py): %v", err)
	}
	var idx shardedIndex
	if err := json.Unmarshal(want, &idx); err != nil {
		t.Fatal(err)
	}
	got, err := marshalIndex(idx)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Errorf("marshalIndex diverges from the transformers golden bytes:\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// goldenIndexInternal duplicates sharded_test.go's goldenIndex constant for
// the internal test package (the two test packages cannot share identifiers).
const goldenIndexInternal = "testdata/sharded/model.safetensors.index.json"

// TestSaveShardedDeterministic: two saves of the same checkpoint produce
// byte-identical files (T721 deterministic-write discipline: sorted-name
// packing over the unordered Go map, deterministic Save, sorted-key index).
func TestSaveShardedDeterministic(t *testing.T) {
	tensors := shardedWriteFixture()
	meta := map[string]string{"format": "pt"}
	dirA, dirB := t.TempDir(), t.TempDir()
	if err := SaveSharded(dirA, tensors, meta, WithMaxShardSize(40)); err != nil {
		t.Fatal(err)
	}
	if err := SaveSharded(dirB, tensors, meta, WithMaxShardSize(40)); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dirA)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 8 { // 7 shards + index
		t.Fatalf("expected 8 files, got %d", len(entries))
	}
	for _, e := range entries {
		a, err := os.ReadFile(filepath.Join(dirA, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(filepath.Join(dirB, e.Name()))
		if err != nil {
			t.Fatalf("%s missing from second save: %v", e.Name(), err)
		}
		if string(a) != string(b) {
			t.Errorf("%s differs between two saves", e.Name())
		}
	}
}

// TestSaveShardedCleansStale: like the HF writer, a save removes files a
// previous save under the same base name left behind — shrinking sharded →
// single must delete the stale index and shards (which would otherwise shadow
// the fresh save on a later LoadSharded), and growing single → sharded must
// delete the stale single file. Unrelated files are untouched.
func TestSaveShardedCleansStale(t *testing.T) {
	tensors := shardedWriteFixture()
	dir := t.TempDir()
	unrelated := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(unrelated, []byte("keep"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := SaveSharded(dir, tensors, nil, WithMaxShardSize(40)); err != nil {
		t.Fatal(err)
	}
	if err := SaveSharded(dir, tensors, nil); err != nil { // default 5 GB → single
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, e.Name())
	}
	sort.Strings(names)
	if want := []string{"model.safetensors", "notes.txt"}; !reflect.DeepEqual(names, want) {
		t.Fatalf("after shrink, files %v, want %v (stale shards/index must be removed)", names, want)
	}
	// Grow back: the stale single file must vanish.
	if err := SaveSharded(dir, tensors, nil, WithMaxShardSize(120)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); !os.IsNotExist(err) {
		t.Errorf("stale single-file checkpoint must be removed on a sharded save: %v", err)
	}
	if _, err := os.Stat(unrelated); err != nil {
		t.Errorf("unrelated file must survive: %v", err)
	}
}

// TestSaveShardedErrors: degenerate inputs error up front, before any file is
// written — empty map, non-positive budget, hostile base names, and (if ever
// constructible) unsupported dtypes are all rejected with a named cause.
func TestSaveShardedErrors(t *testing.T) {
	valid := map[string]*tensor.Tensor{"a": tensor.FromFloat32(tensor.Shape{2}, []float32{1, 2})}
	cases := []struct {
		name    string
		tensors map[string]*tensor.Tensor
		opts    []ShardOption
		wantErr string
	}{
		{"empty map", map[string]*tensor.Tensor{}, nil, "no tensors"},
		{"nil map", nil, nil, "no tensors"},
		{"zero max", valid, []ShardOption{WithMaxShardSize(0)}, "must be positive"},
		{"negative max", valid, []ShardOption{WithMaxShardSize(-5)}, "must be positive"},
		{"empty base name", valid, []ShardOption{WithBaseName("")}, "empty shard filename"},
		{"base name traversal", valid, []ShardOption{WithBaseName("../evil")}, "path separator"},
		{"base name slash", valid, []ShardOption{WithBaseName("a/b")}, "path separator"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := t.TempDir()
			err := SaveSharded(dir, c.tensors, nil, c.opts...)
			if err == nil {
				t.Fatal("must error")
			}
			if !strings.Contains(err.Error(), c.wantErr) {
				t.Errorf("error %q does not contain %q", err, c.wantErr)
			}
			entries, rerr := os.ReadDir(dir)
			if rerr == nil && len(entries) != 0 {
				t.Errorf("failed save must not leave files: %v", entries)
			}
		})
	}
}

// TestSaveShardedPythonExport writes a sharded export plus a JSON reference
// of every tensor's exact values into $GOAI_SHARDED_EXPORT_DIR, for
// scripts/verify_sharded_export.py to load with the REAL Hugging Face
// consumer (the venv's safetensors/torch stack walking the index exactly as
// transformers' loader does) and assert equality. Skipped unless the env var
// is set — this is the manual real-consumer gate, not a CI test. Run:
//
//	.venv/bin/python scripts/verify_sharded_export.py
func TestSaveShardedPythonExport(t *testing.T) {
	dir := os.Getenv("GOAI_SHARDED_EXPORT_DIR")
	if dir == "" {
		t.Skip("set GOAI_SHARDED_EXPORT_DIR to write the python-verification export (scripts/verify_sharded_export.py)")
	}
	tensors := shardedWriteFixture()
	if err := SaveSharded(dir, tensors, map[string]string{"format": "pt"}, WithMaxShardSize(80)); err != nil {
		t.Fatal(err)
	}
	type refTensor struct {
		Dtype  string    `json:"dtype"`
		Shape  []int     `json:"shape"`
		Values []float64 `json:"values"`
	}
	ref := make(map[string]refTensor, len(tensors))
	for name, tt := range tensors {
		dn, err := dtypeName(tt.Dtype())
		if err != nil {
			t.Fatal(err)
		}
		vals := make([]float64, tt.Numel())
		for i := range vals {
			vals[i] = tt.AtF64(tensor.Unravel(i, tt.Shape())...)
		}
		ref[name] = refTensor{Dtype: dn, Shape: tt.Shape(), Values: vals}
	}
	data, err := json.MarshalIndent(ref, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "reference.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	fmt.Printf("sharded export + reference.json written to %s\n", dir)
}
