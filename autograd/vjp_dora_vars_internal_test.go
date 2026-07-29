package autograd

// Captured from the DoRA VJP before the column-blocking rewrite. Column counts cover every
// remainder class of a 4-way unroll, including cols below 4, on BOTH typed branches.
var doraGolden = []doraCase{
	{8, 4, false, 1634692811834633562},
	{9, 5, false, 5499694303601261554},
	{7, 6, false, 17685140276865287335},
	{6, 7, false, 8113786338826772694},
	{5, 1, false, 4807239877578101620},
	{11, 9, false, 2741709920039939588},
	{8, 4, true, 13852667492458949144},
	{9, 5, true, 13156824943179456679},
	{7, 6, true, 3232263782787042199},
	{6, 7, true, 18360015967117329028},
	{5, 1, true, 5125009563605750920},
}
