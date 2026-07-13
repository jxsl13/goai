// Package apicheck enforces the public-API documentation gate (§C13, §V19): every
// exported symbol in a public (non-internal) package must carry a godoc comment,
// and every user-facing package must ship at least one runnable Example. It works
// by parsing source with go/ast (it never imports the packages), so build-tagged
// cgo backends are checked too. Run via `make apicheck` before commit/push.
package apicheck

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// exampleExempt lists module-relative package dirs that need NOT ship an Example:
// the root registration shims and the backend implementation subpackages, which
// are activated by blank import and expose no user-called API surface.
var exampleExempt = map[string]bool{
	".":                 true, // top-level register_*.go blank-import shims
	"backend/ref":       true,
	"backend/cpu":       true,
	"backend/cuda":      true,
	"backend/metal":     true,
	"backend/vulkan":    true,
	"backend/npu":       true,
	"internal/bench":    true,
	"internal/npy":      true,
	"internal/simd":     true,
	"internal/apicheck": true,
}

// typeExampleExempt lists "pkg.Type" entries that need NOT ship a runnable
// Example — types where a standalone usage example is not meaningful (interfaces,
// tiny config/option structs, error/marker types). This is the "where meaningful"
// escape hatch for the per-type example rule (§C13/§V19); keep it short and each
// entry justified.
// methodExampleExempt lists "pkg.Type.Method" entries that need NOT be exercised
// in a runnable Example (§C13 "where meaningful" — the justification lives on each
// entry; trivial accessors already visible through their type's example, contract
// methods like Params() shown across the optimizer examples, etc.).
var methodExampleExempt = map[string]bool{
	"backend.Name.String": true, // fmt.Stringer contract — invoked implicitly by every Printf in the examples
}

var typeExampleExempt = map[string]bool{
	"backend.Op":                    true, // opcode enum; shown via ops/nn examples, not per-op
	"llamagpu.Stepper":              true, // parameter-constraint interface (Decoder/GPTDecoder); shown via ExampleSpeculativeGenerate
	"llamagpu.HiddenStepper":        true, // parameter-constraint interface (Decoder/GPTDecoder + StepHidden); consumed by MedusaGenerate
	"backend.Attrs":                 true, // internal op-parameter bag
	"backend.Kernel":                true, // function type (backend author surface)
	"backend.Backend":               true, // backend-author interface, not user-called
	"backend.Recorder":              true, // autograd-integration interface
	"backend.QuantMatMuler":         true, // optional accelerator capability; shown via nn.QuantLinear
	"backend.ResidentWeight":        true, // resident-weight handle interface; shown via nn.QuantLinear
	"backend.ResidentQuantMatMuler": true, // optional resident-upload capability; shown via nn.QuantLinear
	"backend.Context":               true, // plumbing; used implicitly by every op example
	"tensor.Dtype":                  true, // scalar enum
	"tensor.DeviceKind":             true, // device enum
	"tensor.Device":                 true, // interface; construction is internal
	"tensor.Shape":                  true, // used throughout every tensor example
	"tensor.Pool":                   true, // allocator internals
	"tensor.Allocator":              true, // interface; construction is internal
	"tensor.Storage":                true, // interface (type-erased backing store)
	"tensor.PoolOption":             true, // functional-option type (shown via its setters)
	"tensor.Strides":                true, // []int stride vector, used implicitly everywhere
	"autograd.Op":                   true, // re-export alias
	"autograd.VJP":                  true, // VJP-rule function type (backend/kernel-author surface)
	"autograd.VJPMulti":             true, // multi-output VJP-rule function type (backend/kernel-author surface)
	"autograd.Variable":             true, // internal tape graph node; users work with Tensor+Tape
	"nlp.NextLogits":                true, // callback function type (shown in BeamSearch example)
	"nlp.JacobiStep":                true, // callback function type (shown in ExampleJacobiDecode)
	"nlp.SamplerOption":             true, // functional-option type (shown via WithTemperature etc.)
	"nlp.GenerateOption":            true, // functional-option type (shown via WithBackend, §T361)
	"nlp.MirostatOption":            true, // functional-option type (shown via WithMirostatTau etc.)
	"nlp.UnigramOption":             true, // functional-option type (shown via WithUnigram* setters)
	"nlp.BPEOption":                 true, // functional-option type (shown via WithBPEUnkID)
	"nlp.WatermarkOption":           true, // functional-option type (shown via WithWatermarkGamma etc.)
	"nlp.WordPieceOption":           true, // functional-option type (shown via WithWordPieceUnk etc.)
	"nlp.MedusaHeadsOption":         true, // functional-option type (shown via WithMedusaHeadsDtype)
	"nlp.ChatRenderOption":          true, // functional-option type (shown via WithGenerationPrompt/WithoutBOS)
	"vision.CNNOption":              true, // functional-option type (shown via WithChannels/WithKernel/WithDtype)
	"vision.ViTOption":              true, // functional-option type (shown via WithViTDim/WithViTHeads etc.)
	"nlp.Beam":                      true, // returned by BeamSearch, shown in ExampleBeamSearch
	"nlp.Block":                     true, // internal transformer block (part of GPT)
	"nlp.LlamaBlock":                true, // internal transformer block (part of Llama, shown in ExampleLlama)
	"nlp.QuantBlock":                true, // quantized transformer block (part of QuantLlama, shown in ExampleQuantLlama)
	"nn.QuantSwiGLU":                true, // quantized FFN sublayer (part of QuantLlama, shown in ExampleQuantLlama)
	"nlp.GPTConfig":                 true, // config struct
	"nlp.GPT":                       true, // needs safetensors weight fixtures; covered by GPT tests
	"nlp.KVCache":                   true, // decode state; needs a full model, covered by decode tests
	"nlp.LlamaCache":                true, // Llama decode state; needs a full model, covered by llama_decode tests
	"nlp.StreamCache":               true, // StreamingLLM decode state; needs a full model, covered by streaming tests
	"nlp.Tokenizer":                 true, // needs a GPT-2 vocab fixture; covered by tokenizer tests
	"nlp.MHA":                       true, // needs four projection matrices; used within GPT/decode
	"nn.Layer":                      true, // interface (Linear/Sequential are the impls)
	"nn.Optimizer":                  true, // interface (SGD/Adam/Lion are the impls)
	"nn.Activation":                 true, // built via ReLU()/GELU()/…, shown in ExampleSequential
	"nn.PrefOption":                 true, // functional-option type (shown via Beta/ReferencePoint)
	"nn.GRPOOption":                 true, // functional-option type (shown via WithKLBeta)
	"nn.GSPOOption":                 true, // functional-option type (shown via WithGSPOClipEpsilon)
	"nn.LionOption":                 true, // functional-option type (shown via WithLionBetas)
	"nn.LAMBOption":                 true, // functional-option type (shown via WithLAMBBetas)
	"nn.AdapterOption":              true, // functional-option type (shown via AdapterActivation)
	"nn.SpectralNormOption":         true, // functional-option type (shown via WithSpectralNormIters)
	"nn.GrokfastOption":             true, // functional-option type (shown via WithGrokfastLambda/Alpha)
	"nn.GrokfastMAOption":           true, // functional-option type (shown via WithGrokfastMA*)
	"nn.ShampooOption":              true, // functional-option type (shown via WithShampooEps)
	"nn.GPTQOption":                 true, // functional-option type (shown via WithGPTQDamp)
	"nn.SparseGPTOption":            true, // functional-option type (shown via WithSparseGPTDamp/Block)
	"nn.HQQOption":                  true, // functional-option type (shown via WithHQQLpNorm/Iters)
	"nn.AWQOption":                  true, // functional-option type (shown via WithAWQAlpha/Grid)
	"nn.SOAPOption":                 true, // functional-option type (shown via WithSOAPBetas/Eps/Freq)
	"nn.MoDSelection":               true, // opaque routing handle from MixtureOfDepths.Route (shown via ExampleMixtureOfDepths)
	"nn.MuonOption":                 true, // functional-option type (shown via WithMuonMomentum)
	"nn.AdafactorOption":            true, // functional-option type (shown via WithAdafactor*)
	"nn.GaLoreOption":               true, // functional-option type (shown via WithGaLore*)
	"nn.SophiaOption":               true, // functional-option type (shown via WithSophia*)
	"nn.AdEMAMixOption":             true, // functional-option type (shown via WithAdEMAMix*)
	"nn.CautiousAdamWOption":        true, // functional-option type (shown via WithCautious*)
	"nn.SAMOption":                  true, // functional-option type (shown via WithSAM*)
	"autograd.CheckpointFunc":       true, // callback-signature type (shown via ExampleCheckpoint)
	"nn.ScheduleFreeOption":         true, // functional-option type (shown via WithScheduleFree*)
	"nn.LookaheadOption":            true, // functional-option type (shown via WithLookahead*)
	"nn.LossScaler":                 true, // mixed-precision helper; shown in the AMP training tests
	"nn.MixedPrecision":             true, // mixed-precision helper; shown in the AMP training tests
	"format/gguf.QuantType":         true, // enum; shown via QMatMul usage
	"format/gguf.File":              true, // returned by ReadFile, shown in ExampleReadFile
	"format/gguf.RawFile":           true, // returned by ReadRaw, shown in ExampleReadRaw
	"format/gguf.QuantTensor":       true, // a still-quantized tensor from ReadRaw, shown in ExampleReadRaw
	"rl.Env":                        true, // interface (Chain is the example impl)
	"rl.Reinforce":                  true, // policy-gradient agent; convergence shown in rl tests
}

func moduleRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found walking up from cwd")
		}
		dir = parent
	}
}

// recvTypeName returns the (exported?) receiver type name of a method decl.
func recvTypeName(fd *ast.FuncDecl) (string, bool) {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return "", false
	}
	expr := fd.Recv.List[0].Type
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}
	if id, ok := expr.(*ast.Ident); ok {
		return id.Name, true
	}
	return "", false
}

// exported names of a value/type spec plus whether the decl (block or spec) has a
// doc comment.
func specSymbols(gd *ast.GenDecl) []struct {
	name   string
	hasDoc bool
} {
	var out []struct {
		name   string
		hasDoc bool
	}
	blockDoc := gd.Doc != nil
	for _, s := range gd.Specs {
		switch sp := s.(type) {
		case *ast.TypeSpec:
			if sp.Name.IsExported() {
				out = append(out, struct {
					name   string
					hasDoc bool
				}{sp.Name.Name, blockDoc || sp.Doc != nil || sp.Comment != nil})
			}
		case *ast.ValueSpec:
			doc := blockDoc || sp.Doc != nil || sp.Comment != nil
			for _, n := range sp.Names {
				if n.IsExported() {
					out = append(out, struct {
						name   string
						hasDoc bool
					}{n.Name, doc})
				}
			}
		}
	}
	return out
}

// TestPublicAPIDocumentedWithExamples fails if any exported symbol in a public
// package lacks a doc comment, or any user-facing package ships no Example.
func TestPublicAPIDocumentedWithExamples(t *testing.T) {
	root := moduleRoot(t)

	// collect package dirs (relative) that hold non-test .go files
	pkgDirs := map[string]bool{}
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			base := d.Name()
			if base == ".git" || base == "testdata" || base == ".venv" || base == "docs" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".go") {
			rel, _ := filepath.Rel(root, filepath.Dir(path))
			rel = filepath.ToSlash(rel) // windows: Rel yields backslashes (§T565)
			pkgDirs[rel] = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	var missingDoc, missingExample, missingTypeExample, missingMethodExample []string
	fset := token.NewFileSet()
	dirs := make([]string, 0, len(pkgDirs))
	for d := range pkgDirs {
		dirs = append(dirs, d)
	}
	sort.Strings(dirs)

	for _, rel := range dirs {
		if strings.HasPrefix(rel, "internal") && rel != "internal/apicheck" {
			// internal packages: not public API — skip entirely
			continue
		}
		if implPkg(rel) {
			// backend implementation subpackages are activated by blank import and
			// registered via init; users go through backend.Default(), never these
			// types directly — not part of the public API surface.
			continue
		}
		abs := filepath.Join(root, rel)
		pkgs, perr := parser.ParseDir(fset, abs, nil, parser.ParseComments)
		if perr != nil {
			t.Fatalf("parse %s: %v", rel, perr)
		}

		exportedSyms := 0 // funcs + types exported (user-facing surface)
		hasExample := false
		var typeNames []string
		exampleIdents := map[string]bool{}    // identifiers appearing in Example bodies
		exampleCalls := map[string]bool{}     // method names CALLED via selector in Example bodies (§C13 per-method rule)
		methodOwners := map[string][]string{} // exported type → its exported methods
		for _, pkg := range pkgs {
			for name, file := range pkg.Files {
				isTest := strings.HasSuffix(name, "_test.go")
				for _, decl := range file.Decls {
					switch dd := decl.(type) {
					case *ast.FuncDecl:
						if isTest {
							if dd.Recv == nil && strings.HasPrefix(dd.Name.Name, "Example") {
								hasExample = true
								ast.Inspect(dd.Body, func(n ast.Node) bool {
									if id, ok := n.(*ast.Ident); ok {
										exampleIdents[id.Name] = true
									}
									if call, ok := n.(*ast.CallExpr); ok {
										if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
											exampleCalls[sel.Sel.Name] = true
										}
									}
									return true
								})
								// ExampleType_Method → credit "Type.Method" directly
								if rest := strings.TrimPrefix(dd.Name.Name, "Example"); rest != "" {
									if i := strings.IndexByte(rest, '_'); i > 0 && i+1 < len(rest) {
										exampleCalls[rest[i+1:]] = true
									}
								}
								// ExampleType / ExampleType_Method → credit "Type"
								if rest := strings.TrimPrefix(dd.Name.Name, "Example"); rest != "" && rest[0] != '_' {
									if i := strings.IndexByte(rest, '_'); i >= 0 {
										rest = rest[:i]
									}
									exampleIdents[rest] = true
								}
							}
							continue
						}
						if !dd.Name.IsExported() {
							continue
						}
						// method: only require doc if receiver type is exported
						if rn, ok := recvTypeName(dd); ok {
							if ast.IsExported(rn) {
								methodOwners[rn] = append(methodOwners[rn], dd.Name.Name)
							}
							if !ast.IsExported(rn) {
								continue
							}
							if dd.Doc == nil {
								missingDoc = append(missingDoc, rel+": ("+rn+")."+dd.Name.Name)
							}
							continue
						}
						exportedSyms++
						if dd.Doc == nil {
							missingDoc = append(missingDoc, rel+": func "+dd.Name.Name)
						}
					case *ast.GenDecl:
						if isTest {
							continue
						}
						for _, s := range specSymbols(dd) {
							if _, isType := anyTypeSpec(dd, s.name); isType {
								exportedSyms++
								typeNames = append(typeNames, s.name)
							}
							if !s.hasDoc {
								missingDoc = append(missingDoc, rel+": "+s.name)
							}
						}
						// every exported struct FIELD is public-facing API too and
						// needs its own doc/inline comment (§C13 — "docs for everything
						// public facing"). Enforced per named exported field.
						for _, spec := range dd.Specs {
							ts, ok := spec.(*ast.TypeSpec)
							if !ok || !ts.Name.IsExported() {
								continue
							}
							st, ok := ts.Type.(*ast.StructType)
							if !ok {
								continue
							}
							for _, fld := range st.Fields.List {
								for _, nm := range fld.Names {
									if nm.IsExported() && fld.Doc == nil && fld.Comment == nil {
										missingDoc = append(missingDoc, rel+": "+ts.Name.Name+"."+nm.Name+" (field)")
									}
								}
							}
						}
					}
				}
			}
		}
		if exportedSyms > 0 && !hasExample && !exampleExempt[rel] {
			missingExample = append(missingExample, rel)
		}
		// per-type example coverage (§C13/§V19): every exported user-facing type
		// should carry a runnable Example (Go's ExampleType / ExampleType_Method
		// convention) so even simple methods are shown in use — unless the type is
		// on the "not meaningfully exampled" allowlist ("where meaningful").
		if !exampleExempt[rel] {
			for _, tn := range typeNames {
				if typeExampleExempt[rel+"."+tn] {
					continue
				}
				// op-parameter structs (backend.*Attrs, ADR-0014): tiny typed
				// parameter bags for a specific op, the same "not meaningfully
				// exampled on their own" category as backend.Attrs — they are shown
				// via the op's own Example (e.g. AttnAttrs in the attention examples),
				// not a standalone one. Category rule so new ops need no allowlist edit.
				if rel == "backend" && strings.HasSuffix(tn, "Attrs") {
					continue
				}
				// covered if the type name OR its New<Type> constructor appears in a
				// runnable Example body (types are usually built via a constructor,
				// so the bare type name rarely appears).
				if !exampleIdents[tn] && !exampleIdents["New"+tn] {
					missingTypeExample = append(missingTypeExample, rel+"."+tn)
				}
			}
		}
		// per-METHOD example coverage (§C13/§V19 class (d), user feedback 2026-07-13):
		// every exported method on a user-facing type must be exercised in a
		// runnable Example — credited when the method name is CALLED via a selector
		// in any Example body or a dedicated ExampleType_Method exists. Always-on
		// since the §T569 sweep (74 methods covered across 8 packages).
		if !exampleExempt[rel] {
			for _, tn := range typeNames {
				if typeExampleExempt[rel+"."+tn] {
					continue
				}
				if rel == "backend" && strings.HasSuffix(tn, "Attrs") {
					continue
				}
				for _, mn := range methodOwners[tn] {
					if exampleCalls[mn] {
						continue
					}
					if methodExampleExempt[rel+"."+tn+"."+mn] {
						continue
					}
					missingMethodExample = append(missingMethodExample, rel+"."+tn+"."+mn)
				}
			}
		}
	}

	sort.Strings(missingDoc)
	if len(missingDoc) > 0 {
		t.Errorf("undocumented exported symbols (%d) — every public API symbol needs a godoc comment (§V19):\n  %s",
			len(missingDoc), strings.Join(missingDoc, "\n  "))
	}
	if len(missingExample) > 0 {
		t.Errorf("public packages without a runnable Example (%d) — every user-facing package needs one (§V19):\n  %s",
			len(missingExample), strings.Join(missingExample, "\n  "))
	}
	sort.Strings(missingMethodExample)
	if len(missingMethodExample) > 0 {
		t.Errorf("exported methods not exercised in any runnable Example (%d) — every method needs one where meaningful (§C13/§V19 class (d)); call it in an Example or allowlist in methodExampleExempt:\n  %s",
			len(missingMethodExample), strings.Join(missingMethodExample, "\n  "))
	}
	sort.Strings(missingTypeExample)
	if len(missingTypeExample) > 0 {
		t.Errorf("exported types without a runnable Example (%d) — even simple types/methods need one where meaningful (§C13/§V19); add ExampleType or allowlist in typeExampleExempt:\n  %s",
			len(missingTypeExample), strings.Join(missingTypeExample, "\n  "))
	}
}

// implPkg reports whether rel is a backend implementation subpackage (registered
// via blank import, not called directly), which is exempt from the public-API gate.
func implPkg(rel string) bool {
	switch rel {
	case "backend/ref", "backend/cpu", "backend/cuda", "backend/metal", "backend/vulkan", "backend/npu":
		return true
	}
	return false
}

// anyTypeSpec reports whether name is declared by a TypeSpec in gd.
func anyTypeSpec(gd *ast.GenDecl, name string) (bool, bool) {
	for _, s := range gd.Specs {
		if ts, ok := s.(*ast.TypeSpec); ok && ts.Name.Name == name {
			return true, true
		}
	}
	return false, false
}
