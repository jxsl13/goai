package nn_test

// Captured from the sort.Slice / sort.SliceStable implementation, before the move to
// slices.SortFunc / slices.SortStableFunc, the panel transpose and the selection.
var wandaGolden = []wandaCase{
	{8, 8, 5, 0.5, false, false, 17831350629602582628},
	{7, 12, 4, 0.25, false, false, 3344436033556774352},
	{5, 9, 3, 0.75, false, false, 9262969263942727291},
	{8, 8, 5, 0, true, false, 5180391591037310627},
	{6, 12, 4, 0, true, false, 2478852007814154555},
	{4, 16, 3, 0, true, false, 1912950787694039764},
	// F32 typed branch, captured before it was panelled and switched to a selection.
	{8, 8, 5, 0.5, false, true, 3632912123890759404},
	{7, 12, 4, 0.25, false, true, 12627138530392909576},
	{5, 9, 3, 0.75, false, true, 12891423146967092399},
	{8, 8, 5, 0, true, true, 5607561138992108342},
	{6, 12, 4, 0, true, true, 13481324836605973809},
}
