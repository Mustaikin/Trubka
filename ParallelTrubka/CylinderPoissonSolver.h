#pragma once
void solveByFftBsor(
	double*** uin, double*** f,
	int nz, int nr, int nfi,
	double hz, double hr, double hfi, double eps);
