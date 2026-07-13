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
// Usage: go run ./internal/cichange <base-rev> <head-rev>
// Prints "docs-only" or "code" on stdout; exits non-zero only on usage errors.
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: cichange <base-rev> <head-rev>")
		os.Exit(2)
	}
	fmt.Println(Classify(".", os.Args[1], os.Args[2]))
}
