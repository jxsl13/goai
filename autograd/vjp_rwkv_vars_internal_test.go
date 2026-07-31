package autograd

// Captured from the WKV VJP on BOTH typed branches.
//
// REGENERATED for the linear-time backward (T1002). The previous values pinned the O(seq^2) reverse
// pass, which computed each gradient by accumulating one term per (t,i) pair; the linear form sums
// the same quantities in a different association and through a running-maximum rescale, so the
// results differ in the last few ulps and every checksum moved. Permitted by ADR-01KYTPF84PEC0,
// which allows regenerating a REGRESSION pin — as this is — when a measured win justifies it.
//
// A checksum validates nothing about correctness the moment it is regenerated. The gates that
// actually held during the rewrite were the finite-difference gradchecks, in particular
// TestWKVVJPGradCheckSweep, which drives a NON-UNIFORM upstream gradient and so can see errors in
// how g_t weights the accumulation (A-SUM-LOSS-GRADCHECK-IS-BLIND-TO-UPSTREAM-WEIGHTS-001).
// From here these values are again a regression pin and nothing more.
var wkvGolden = []wkvCase{
	{8, 4, false, 14378710810260724611},
	{9, 5, false, 13852544513577913090},
	{7, 6, false, 2113992002686129295},
	{6, 7, false, 1501583438141904683},
	{5, 1, false, 17682951080143194645},
	{13, 9, false, 16989578202813079745},
	{8, 4, true, 1725716987406327857},
	{9, 5, true, 16012051177630346572},
	{7, 6, true, 8645532654449763136},
	{6, 7, true, 9902839556237547607},
	{5, 1, true, 12009649596831189522},
}
