// Command cichange classifies the change between two git revisions as either
// "docs-only" or "code" so CI can skip expensive pipelines for pure
// documentation pushes (§T571).
//
// In plain terms: if a push only edits markdown files or code COMMENTS, the
// compiled program cannot have changed, so running the full test matrix
// would burn CI minutes for nothing. This tool proves that safely: markdown
// and docs/ files are documentation by path; for a modified .go file it
// parses BOTH versions and compares their abstract syntax trees with all
// comments stripped — byte-identical trees mean only comments (godoc) moved.
//
// Safety (§V26, fail-open): the answer is "docs-only" ONLY on positive proof.
// Every uncertain case — parse errors, added/deleted files that are not pure
// docs, workflow or build files, unreadable revisions, and DIRECTIVE comments
// (//go:build, //go:embed, //go:generate, //line, cgo preambles), which change
// compilation despite being comments — classifies as "code".
//
// Usage:
//
//	go run ./internal/cichange <base-rev> <head-rev>          → "docs-only" | "code"
//	go run ./internal/cichange -impact <base-rev> <head-rev>  → "none" | "all" | "./pkg ..."
//
// The -impact mode (§T579) narrows further: instead of the binary docs/code verdict it
// prints exactly the packages whose tests are affected by the change — the reverse
// import closure of the changed packages (see Impact). CI runs `go test` on that list,
// the full suite on "all", and nothing on "none". Exits non-zero only on usage errors.
package main

import (
	"fmt"
	"os"
)

func main() {
	args := os.Args[1:]
	mode := ""
	if len(args) > 0 && (args[0] == "-impact" || args[0] == "-validate") {
		mode = args[0]
		args = args[1:]
	}
	if len(args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: cichange [-impact|-validate] <base-rev> <head-rev>")
		os.Exit(2)
	}
	switch mode {
	case "-impact":
		fmt.Println(Impact(".", args[0], args[1]))
	case "-validate": // §T583/§V27: judge the selector against the toolchain oracle
		report, ok := Validate(".", args[0], args[1])
		fmt.Println(report)
		if !ok {
			os.Exit(1)
		}
	default:
		fmt.Println(Classify(".", args[0], args[1]))
	}
}
