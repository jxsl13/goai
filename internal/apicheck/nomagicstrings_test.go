package apicheck

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// backendNameLiterals are the values of the backend.Name enum (backend/names.go).
// Outside their definition and the DeviceKind stringer, a bare string literal with
// one of these values is a magic string that should be the typed constant
// (backend.CPU, backend.Metal, …) — ADR-0015, §C15.
var backendNameLiterals = map[string]bool{
	"cpu": true, "ref": true, "metal": true, "cuda": true, "vulkan": true,
}

// magicStringExempt reports whether rel is a file where a backend-name-valued
// literal is legitimate: the Name enum's own definition, or anything in the tensor
// package — which owns the SEPARATE tensor.DeviceKind enum whose canonical strings
// happen to share the spelling ("cpu"/"metal"/…), used by its stringer and tests.
// The GPU backends' own device.String() derives from Kind(), so they carry no
// literal and are NOT exempt.
func magicStringExempt(rel string) bool {
	// internal/docgraph validates SPEC VOCABULARY, not backends: §R's conf
	// levels are literally "high|med|low|ref" (FORMAT.md) — its "ref" is a
	// research-confidence value, not a backend reference (§V40 tooling).
	//
	// leadership/collect.go emits the §V38 evidence JSON, where "cpu" is the HOST
	// hardware-descriptor field name (the machine's CPU model string), not a
	// backend selector. The key is already recorded in published evidence
	// metadata, so it is a frozen schema field: renaming it would desync the
	// collector from the evidence it has already written (§T987).
	return rel == "backend/names.go" || strings.HasPrefix(rel, "tensor/") ||
		strings.HasPrefix(rel, "internal/docgraph/") ||
		rel == "internal/benchcompare/leadership/collect.go"
}

// TestNoMagicBackendNameStrings guards §C15/§V21: backends are referred to by the
// typed backend.Name constants, never a bare "metal"/"cpu"/… string literal. It
// inspects string BasicLits via go/ast (so comments and import paths are ignored)
// and fails on any backend-name literal outside the allowlisted definition files.
func TestNoMagicBackendNameStrings(t *testing.T) {
	root := moduleRoot(t)
	fset := token.NewFileSet()
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
		rel, _ := filepath.Rel(root, path)
		rel = filepath.ToSlash(rel) // windows: Rel yields backslashes (§T565)
		if magicStringExempt(rel) || strings.HasSuffix(path, "nomagicstrings_test.go") {
			return nil
		}
		f, perr := parser.ParseFile(fset, path, nil, 0)
		if perr != nil {
			return nil // unparseable (e.g. build-tagged in isolation) — skip
		}
		ast.Inspect(f, func(n ast.Node) bool {
			lit, ok := n.(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				return true
			}
			if v, e := strconv.Unquote(lit.Value); e == nil && backendNameLiterals[v] {
				offenders = append(offenders, rel+":"+strconv.Itoa(fset.Position(lit.Pos()).Line)+" ("+strconv.Quote(v)+")")
			}
			return true
		})
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(offenders) > 0 {
		t.Errorf("magic backend-name string literals (%d) — use the typed backend.Name constants (backend.CPU/Metal/Ref/CUDA/Vulkan), §C15/ADR-0015:\n  %s",
			len(offenders), strings.Join(offenders, "\n  "))
	}
}
