package nn_test

// Captured from the sort.Slice / sort.SliceStable implementation, before the move to
// slices.SortFunc / slices.SortStableFunc.
var wandaGolden = []wandaCase{
	{8, 8, 5, 0.5, false, 17831350629602582628},
	{7, 12, 4, 0.25, false, 3344436033556774352},
	{5, 9, 3, 0.75, false, 9262969263942727291},
	{8, 8, 5, 0, true, 5180391591037310627},
	{6, 12, 4, 0, true, 2478852007814154555},
	{4, 16, 3, 0, true, 1912950787694039764},
}
