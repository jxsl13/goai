package nn_test

// {seq, d, chunkSize, checksum} captured from RetentionChunkwise BEFORE the output loop was
// blocked. d covers every remainder class of a 4-way unroll, including d below 4, and the
// chunk sizes cover chunks that do and do not divide the sequence.
var retChunkGolden = [][4]uint64{
	{16, 8, 4, 14325735947702323436},
	{17, 5, 3, 5465374177224374500},
	{12, 6, 5, 1488148167538442047},
	{9, 7, 2, 7476204293191651738},
	{8, 1, 4, 16623413897792316992},
	{14, 9, 7, 3934885807211874841},
}
