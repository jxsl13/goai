package autograd

// Captured from the WKV VJP before the column-gather rewrite, on BOTH typed branches.
var wkvGolden = []wkvCase{
	{8, 4, false, 17005050383803545071},
	{9, 5, false, 16189487588981292315},
	{7, 6, false, 441206035031203374},
	{6, 7, false, 17695744759978740607},
	{5, 1, false, 12028742404785902220},
	{13, 9, false, 15474419715581153627},
	{8, 4, true, 18048111106146760907},
	{9, 5, true, 1708085259003098230},
	{7, 6, true, 11157397904931941344},
	{6, 7, true, 5399447111320283579},
	{5, 1, true, 12009649596831189522},
}
