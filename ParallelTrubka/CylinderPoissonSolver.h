#pragma once
void solveByFftBsor(
    double*** uin, double*** f,
    double R, double z0, double z1,
    int nz, int nr, int nfi,
    double hz, double hr, double hfi, double eps);
