//go:build !race

package autograd

// raceBuild reports whether this binary was built with the race detector.
//
// It exists because the race build CHANGES FLOATING-POINT RESULTS in this package: the same MLA
// backward, on the same inputs, digests differently under -race than without it, on the
// pre-change implementation as much as on the current one. The race build inhibits optimizations
// the compiler is otherwise free to make — multiply-add fusion among them, which the Go spec
// permits it to choose — so a bit-exact golden pins a value that is only the value of one build
// mode. Tests that assert exact bits skip when this is true; the equivalence tests beside them,
// which compare two runs of the SAME binary, do not need to.
const raceBuild = false
