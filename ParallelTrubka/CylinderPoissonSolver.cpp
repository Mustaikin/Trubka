#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include "BSOR.h"
#include <fenv.h>
#include <signal.h>


const double PI = 3.14159265;

double*** uin, *** f, *** uinRe, *** uinIm, *** fRe, *** fIm, *** resReIm, *** uinRes, ***uin_temp;



// Выделение памяти
void mem_alloc(int nr, int nfi, int nz) {
    uin = new double** [nr];
    uin_temp = new double** [nr];
    f = new double** [nr];
    uinRes = new double** [nr];
    for (int i = 0; i < nr; i++) {
        uin[i] = new double* [nfi];
        uin_temp[i] = new double* [nfi];
        f[i] = new double* [nfi];
        uinRes[i] = new double* [nfi];
        for (int j = 0; j < nfi; j++) {
            uin[i][j] = new double[nz]();
            uin_temp[i][j] = new double[nz]();
            f[i][j] = new double[nz]();
            uinRes[i][j] = new double[nz]();
        }
    }

    uinRe = new double** [nfi];
    uinIm = new double** [nfi];
    fRe = new double** [nfi];
    fIm = new double** [nfi];

    for (int i = 0; i < nfi; i++) {
        uinRe[i] = new double* [nr];
        uinIm[i] = new double* [nr];
        fRe[i] = new double* [nr];
        fIm[i] = new double* [nr];
        for (int j = 0; j < nr; j++) {
            uinRe[i][j] = new double[nz];
            uinIm[i][j] = new double[nz];
            fRe[i][j] = new double[nz];
            fIm[i][j] = new double[nz];
        }
    }

    resReIm = new double** [2];
    for (int i = 0; i < 2; i++) {
        resReIm[i] = new double* [nr];
        for (int j = 0; j < nr; j++) {
            resReIm[i][j] = new double[nz];
        }
    }
}

// Освобождение памяти
void mem_delete(int nr, int nfi, int nz) {
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nfi; j++) {
            delete[] uin[i][j];
            delete[] uin_temp[i][j];
            delete[] f[i][j];
            delete[] uinRes[i][j];
        }
        delete[] uin[i];
        delete[] uin_temp[i];
        delete[] f[i];
        delete[] uinRes[i];
    }
    delete[] uin;
    delete[] uin_temp;
    delete[] f;
    delete[] uinRes;

    for (int i = 0; i < nfi; i++) {
        for (int j = 0; j < nr; j++) {
            delete[] uinRe[i][j];
            delete[] uinIm[i][j];
            delete[] fRe[i][j];
            delete[] fIm[i][j];
        }
        delete[] uinRe[i];
        delete[] uinIm[i];
        delete[] fRe[i];
        delete[] fIm[i];
    }
    delete[] uinRe;
    delete[] uinIm;
    delete[] fRe;
    delete[] fIm;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < nr; j++) {
            delete[] resReIm[i][j];
        }
        delete[] resReIm[i];
    }
    delete[] resReIm;
}


void swapCoords(double*** arr1, double*** arr2,
    int nz, int nr, int nfi, int flag)
{
    if (flag == 0) {
        for (int i = 0; i < nz; i++) {
            for (int j = 0; j < nr; j++) {
                for (int k = 0; k < nfi; k++) {
                    arr2[j][k][i] = arr1[i][j][k];
                }
            }
        }
    }
    else {
        for (int j = 0; j < nr; j++) {
            for (int k = 0; k < nfi; k++) {
                for (int i = 0; i < nz; i++) {
                    arr2[i][j][k] = arr1[j][k][i];
                }
            }
        }
    }
}
// Решатель
void solveByFftBsor(
    double*** uin, double*** f,
    int nz, int nr, int nfi,
    double hz, double hr, double hfi, double eps) {
    mem_alloc(nr, nfi, nz);
    swapCoords(uin, uin_temp, nz, nr, nfi, 0);

    double* phiSliceU = new double[nfi];
    double* phiSliceF = new double[nfi];

    // FFT по φ
    for (int i = 0; i < nr; i++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < nfi; j++) {
                phiSliceU[j] = uin_temp[i][j][k];
                phiSliceF[j] = f[i][j][k];
            }

            double** fftU = fft_handler(phiSliceU, nfi);
            double** fftF = fft_handler(phiSliceF, nfi);

            for (int m = 0; m < nfi; m++) {
                uinRe[m][i][k] = fftU[0][m];
                uinIm[m][i][k] = fftU[1][m];
                fRe[m][i][k] = fftF[0][m];
                fIm[m][i][k] = fftF[1][m];
            }

            for (int t = 0; t < 2; t++) {
                delete[] fftU[t];
                delete[] fftF[t];
            }
            delete[] fftU;
            delete[] fftF;
        }
    }

    delete[] phiSliceU;
    delete[] phiSliceF;

    // SOR
    double* lambdas = new double[nfi];
    double* omegas = new double[nfi];
    getLambdas(lambdas, nfi, hfi);
    getOmegas(omegas, lambdas, nfi, nr, nz, hr, hz);

    for (int m = 0; m < nfi; m++) {
        blockSOR(lambdas[m], m, uinRe[m], uinIm[m], fRe[m], fIm[m],
            hr, hz, omegas[m], eps, resReIm, nr, nfi, nz);

        // копируем результат обратно
        for (int i = 0; i < nr; i++) {
            for (int k = 0; k < nz; k++) {
                uinRe[m][i][k] = resReIm[0][i][k];
                uinIm[m][i][k] = resReIm[1][i][k];
            }
        }
    }


    // Обратное FFT
    double* phiSliceURe = new double[nfi];
    double* phiSliceUim = new double[nfi];
    for (int i = 0; i < nr; i++) {
        for (int k = 0; k < nz; k++) {
            for (int m = 0; m < nfi; m++) {
                phiSliceURe[m] = uinRe[m][i][k];
                phiSliceUim[m] = uinIm[m][i][k];
            }
            double* ifftU = ifft_handler(phiSliceURe, phiSliceUim, nfi);
            for (int j = 0; j < nfi; j++) {
                uinRes[i][j][k] = ifftU[j];
            }
            delete[] ifftU;
        }
    }

    delete[] phiSliceURe;
    delete[] phiSliceUim;
    delete[] lambdas;
    delete[] omegas;
    swapCoords(uinRes, uin, nz, nr, nfi, 1);
    std::cout << "Redy!\n";
    mem_delete(nr, nfi, nz);
}
 