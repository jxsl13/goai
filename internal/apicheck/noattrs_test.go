package apicheck

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// forbiddenAttrPatterns are the fingerprints of the old stringly-typed op-params
// bag (ADR-0014 replaced it with a sealed interface + one struct per op). If any
// reappears, op parameters have regressed to map[string]any and §V20/§C14/§I6 are
// violated.
var forbiddenAttrPatterns = []string{
	"Attrs map[string]", // the old `type Attrs map[string]any`
	"attrs.Int(",        // the old typed accessors on an Attrs value
	"attrs.Bool(",
	"attrs.Float(",
	"attrs.Ints(",
}

// TestNoStringKeyedAttrs guards §V20: op parameters must stay typed. It greps
// every Go source file (this checker excepted, since it names the patterns) for
// the old accessor/map-bag fingerprints and fails if one has crept back in.
func TestNoStringKeyedAttrs(t *testing.T) {
	root := moduleRoot(t)
	var offenders []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "testdata", ".venv", "docs", ".claude":
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		// this guard file necessarily contains the pattern strings themselves.
		if strings.HasSuffix(path, "noattrs_test.go") {
			return nil
		}
		b, rerr := os.ReadFile(path)
		if rerr != nil {
			return rerr
		}
		src := string(b)
		for _, pat := range forbiddenAttrPatterns {
			if strings.Contains(src, pat) {
				rel, _ := filepath.Rel(root, path)
				offenders = append(offenders, rel+": "+pat)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(offenders) > 0 {
		t.Errorf("string-keyed op params reintroduced (§V20/§C14 — use a typed backend.Attrs struct, not map[string]any accessors):\n  %s",
			strings.Join(offenders, "\n  "))
	}
}
