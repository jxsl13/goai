package nn_test

// Captured from the unoptimized NSABranches. These are the contract: any edit that moves a
// bit of cmp, slc or win fails TestNSABranchesBitIdentical.
var nsaGolden = []nsaGeom{
	{16, 8, 2, 4, 4, 4, 4835280107336373710},
	{17, 8, 2, 4, 2, 5, 7302874317112048350},   // partial trailing block
	{24, 16, 4, 8, 3, 40, 8946234414237082},    // window longer than the sequence
	{13, 12, 3, 5, 1, 2, 1895264672924377056},  // topN 1: only the query's own block
	{9, 6, 1, 1, 4, 3, 12277609431880901705},   // blockSize 1
	{32, 16, 2, 3, 5, 7, 18314563674876454232}, // blockSize not dividing seq
}
