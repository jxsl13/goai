// Command cichange classifies the change between two git revisions so CI can run
// exactly the work the change requires (§T571/§T579/§T583/§T584).
//
// In plain terms: if a push only edits documentation, running the full test matrix
// burns CI minutes for nothing; if it edits one package, only that package and its
// (transitive) importers can be affected. This tool proves both safely — and every
// rule about what counts as documentation or as globally significant is CONFIGURABLE,
// so the tool is generic rather than tied to one repository's layout.
//
// Modes:
//
//	cichange [flags] <base-rev> <head-rev>            → "docs-only" | "code"
//	cichange -impact [flags] <base-rev> <head-rev>    → "none" | "all" | "./pkg ..."
//	cichange -validate [flags] <base-rev> <head-rev>  → soundness report; exit 1 on
//	                                                    under-selection (§V27)
//
// Rule flags (§T584), all repeatable; precedence full-rerun > pkg-rerun > ignore:
//
//	-ignore <path>            file or directory with no CI impact (docs); relative
//	                          paths resolve against the repository root, and every
//	                          path is absolutized before comparison
//	-ignore-regex <re>        repo-relative paths matching → no CI impact
//	-full-rerun-regex <re>    repo-relative paths matching → the full suite runs
//	-pkg-rerun-regex <re>     repo-relative paths matching → their owning package's
//	                          tests run regardless of content analysis
//	-no-default-rules         start from an empty rule set instead of the built-in
//	                          defaults (docs/ and root *.md/*.txt ignored;
//	                          go.sum, Makefile, .github/ full-rerun)
//
// Safety (§V26, fail-open): "docs-only"/"none" require positive proof. Parse errors,
// unattributable paths, deleted packages, and unmodeled go.mod constructs classify as
// code / all. Exits non-zero only on usage errors (2) and -validate under-selection (1).
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	var (
		impactMode   = flag.Bool("impact", false, "print the affected package list instead of the binary verdict")
		validateMode = flag.Bool("validate", false, "judge -impact against the go list oracle (§V27)")
		runMode      = flag.Bool("run", false, "transparent end-to-end mode: report every decision, then EXECUTE the selected tests (args after -- go to go test)")
		noDefaults   = flag.Bool("no-default-rules", false, "start from an empty rule set")
		ignore       stringList
		ignoreRe     stringList
		fullRe       stringList
		pkgRe        stringList
		alwaysRun    stringList
	)
	flag.Var(&ignore, "ignore", "path (file or dir) with no CI impact; repeatable")
	flag.Var(&ignoreRe, "ignore-regex", "repo-relative path regex with no CI impact; repeatable")
	flag.Var(&fullRe, "full-rerun-regex", "repo-relative path regex forcing the full suite; repeatable")
	flag.Var(&pkgRe, "pkg-rerun-regex", "repo-relative path regex forcing the owning package's tests; repeatable")
	flag.Var(&alwaysRun, "always-run", "module-relative package added to every non-empty selection (whole-tree meta-tests the import closure cannot reach); repeatable")
	flag.Parse()
	if flag.NArg() < 2 {
		fmt.Fprintln(os.Stderr, "usage: cichange [-impact|-validate|-run] [rule flags] <base-rev> <head-rev> [-- go-test-args]")
		os.Exit(2)
	}
	if flag.NArg() > 2 && !*runMode {
		fmt.Fprintln(os.Stderr, "usage: extra arguments are only valid with -run (they are passed to go test)")
		os.Exit(2)
	}
	if !*noDefaults {
		dIg, dIgRe, dFull, dPkg, dAlways := defaultRules()
		ignore = append(stringList(dIg), ignore...)
		ignoreRe = append(stringList(dIgRe), ignoreRe...)
		fullRe = append(stringList(dFull), fullRe...)
		pkgRe = append(stringList(dPkg), pkgRe...)
		alwaysRun = append(stringList(dAlways), alwaysRun...)
	}
	cfg, err := newConfig(".", ignore, ignoreRe, fullRe, pkgRe, alwaysRun)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	base, head := flag.Arg(0), flag.Arg(1)
	switch {
	case *runMode: // §T587: full transparency, then execute — the last step IS the tests
		os.Exit(Run(cfg, ".", base, head, flag.Args()[2:], os.Stdout))
	case *impactMode:
		fmt.Println(Impact(cfg, ".", base, head))
	case *validateMode: // §T583/§V27: judge the selector against the toolchain oracle
		report, ok := Validate(cfg, ".", base, head)
		fmt.Println(report)
		if !ok {
			os.Exit(1)
		}
	default:
		fmt.Println(Classify(cfg, ".", base, head))
	}
}
