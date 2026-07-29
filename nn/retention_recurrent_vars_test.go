package nn_test

// {seq, d, checksum} captured from RetentionRecurrent before the output-loop rewrite.
// d covers every remainder class of a 4-way unroll, including d below 4.
var retRecGolden = [][3]uint64{
	{8, 4, 11745445307611582043},
	{9, 5, 18398891191647496149},
	{7, 6, 16035914877525288453},
	{6, 7, 1327639983572837419},
	{5, 1, 14840848543952262410},
	{11, 9, 15267879205651532955},
}
