package nlp_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestJLensPromptSetsIntact pins the ported jacobian-lens prompt sets (§R250,
// T815): the 11 experiment-replication sets and 6 lens-evaluation sets from the
// reference repository (Apache-2.0, see testdata/jlens_prompts/NOTICE) must all
// be present, valid JSON, and non-trivial — so the material for replicating the
// paper's experiments against GoAI's J-lens stays available and uncorrupted.
func TestJLensPromptSetsIntact(t *testing.T) {
	files, err := filepath.Glob("testdata/jlens_prompts/*.json")
	if err != nil {
		t.Fatal(err)
	}
	const want = 17 // 11 experiments + 6 lens-evals in the reference data/ tree
	if len(files) != want {
		t.Fatalf("expected %d prompt-set files, found %d", want, len(files))
	}
	for _, f := range files {
		raw, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("%s: %v", f, err)
		}
		var v any
		if err := json.Unmarshal(raw, &v); err != nil {
			t.Fatalf("%s: invalid JSON: %v", f, err)
		}
		if len(raw) < 64 {
			t.Fatalf("%s: implausibly small (%d bytes)", f, len(raw))
		}
	}
	if _, err := os.Stat("testdata/jlens_prompts/NOTICE"); err != nil {
		t.Fatalf("Apache-2.0 NOTICE missing: %v", err)
	}
}
