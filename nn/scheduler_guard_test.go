package nn_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
	"strings"
	"testing"
)

// bindableOptimizers lists every optimizer type in this package that BindLR is
// known to reach, together with which of its four resolver rules does the work.
//
// Keeping this list honest is the whole point of the guard below. BindLR reaches
// most optimizers by reflecting on an exported LR field rather than through an
// interface the compiler checks, so an optimizer that renames or drops that field
// — or a newly added one that never had it — would not fail to build. It would
// bind at run time only if some other rule happened to apply, and otherwise
// return an error to a caller who may not be checking. A scheduler that silently
// never moves the learning rate is the failure this guard exists to prevent, and
// it is invisible in a green test suite.
var bindableOptimizers = map[string]string{
	// Rule 3: exported LR float64 field, found by reflection.
	"Adafactor":     "LR field",
	"AdamMini":      "LR field",
	"AdEMAMix":      "LR field",
	"APOLLO":        "LR field",
	"CautiousAdamW": "LR field",
	"DAdaptAdam":    "LR field",
	"GaLore":        "LR field",
	"LAMB":          "LR field",
	"MARS":          "LR field",
	"Muon":          "LR field",
	"Adam":          "LR field",
	"Lion":          "LR field",
	"SGD":           "LR field",
	"Prodigy":       "LR field",
	"PSGDKron":      "LR field",
	"QGaLore":       "LR field",
	"ScheduleFree":  "LR field",
	"Shampoo":       "LR field",
	"SOAP":          "LR field",
	"Sophia":        "LR field",

	// Rule 4: no rate of its own; recursion into an exported Base Optimizer.
	"Grokfast":   "Base recursion",
	"GrokfastMA": "Base recursion",
	"Lookahead":  "Base recursion",

	// Rule 2: fan-out, one handle per group.
	"ParamGroups": "group fan-out",

	// Takes a closure rather than a GradFn, and wraps a base optimizer that is
	// itself schedulable; scheduling SAM means scheduling what it wraps.
	"SAM": "wraps a schedulable base",
}

// TestEveryOptimizerIsAccountedForByBindLR walks this package's own source and
// fails when an optimizer exists that bindableOptimizers does not mention.
//
// It deliberately checks accounting rather than behaviour: constructing all ~24
// optimizers with valid hyperparameters would make the test a maintenance burden
// in its own right, and the behavioural half is already covered (the reachability
// tests exercise each of the four resolver rules, and BindLR returning an error
// for an unreachable optimizer is pinned separately). What is NOT otherwise
// covered is a new optimizer landing and nobody checking whether a scheduler can
// drive it — which is exactly what this catches.
//
// If this fails: add the type with the rule that reaches it, and confirm with a
// real BindLR call that the rule actually applies. Do not add it to the map to
// quiet the test.
func TestEveryOptimizerIsAccountedForByBindLR(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", nil, 0)
	if err != nil {
		t.Fatalf("parse package source: %v", err)
	}
	found := map[string]bool{}
	for name, pkg := range pkgs {
		if strings.HasSuffix(name, "_test") {
			continue
		}
		for path, file := range pkg.Files {
			if strings.HasSuffix(path, "_test.go") {
				continue
			}
			for _, decl := range file.Decls {
				fn, ok := decl.(*ast.FuncDecl)
				if !ok || fn.Recv == nil || fn.Name.Name != "Step" {
					continue
				}
				if recv := receiverTypeName(fn); recv != "" {
					found[recv] = true
				}
			}
		}
	}
	if len(found) == 0 {
		t.Fatal("found no Step methods at all — the AST walk is broken, " +
			"so this guard would pass vacuously")
	}

	var missing []string
	for typ := range found {
		// Not optimizers: these carry a Step method with an unrelated meaning.
		switch typ {
		case "RWKVBlock", "ReduceLROnPlateau", "LRScheduler":
			continue
		}
		if _, ok := bindableOptimizers[typ]; !ok {
			missing = append(missing, typ)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Fatalf("optimizer(s) %v define Step but are not accounted for in "+
			"bindableOptimizers, so nothing verifies a scheduler can drive them; "+
			"a scheduler that silently never changes the learning rate looks "+
			"identical to a working one in a green suite", missing)
	}

	// Reverse direction: a stale entry for a type that no longer exists means the
	// map is drifting away from the source it claims to describe.
	var stale []string
	for typ := range bindableOptimizers {
		if !found[typ] {
			stale = append(stale, typ)
		}
	}
	sort.Strings(stale)
	if len(stale) > 0 {
		t.Fatalf("bindableOptimizers lists %v, which no longer define Step "+
			"in this package; remove the stale entries", stale)
	}
}

func receiverTypeName(fn *ast.FuncDecl) string {
	if len(fn.Recv.List) == 0 {
		return ""
	}
	switch e := fn.Recv.List[0].Type.(type) {
	case *ast.StarExpr:
		if id, ok := e.X.(*ast.Ident); ok {
			return id.Name
		}
	case *ast.Ident:
		return e.Name
	}
	return ""
}
