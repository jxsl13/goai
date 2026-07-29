package nn_test

// Captured from MoBAAttention before the P·V rewrite. These are the contract.
var mobaGolden = []mobaGeom{
	{16, 8, 2, 4, 2, 12239871357196401836},
	{17, 12, 3, 4, 3, 7751208384696701755},  // partial trailing block
	{24, 16, 4, 8, 1, 18288404271511032251}, // topK 1: the query's own block only
	{13, 6, 1, 5, 9, 7018154370496109414},   // topK above the block count
	{9, 8, 2, 1, 4, 4664138288464045535},    // blockSize 1
}
