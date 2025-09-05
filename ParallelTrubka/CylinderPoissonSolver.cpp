#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include "FFT.h"
#include "BSOR.h"
#include "VTK_writer.h"
#include <fenv.h>
#include <signal.h>
#include <time.h>

const double PI = 3.14159265;

void solveByFftBsor(
    double*** uin, double*** f,
    int nz, int nr, int nfi,
    double hz, double hr, double hfi, double eps)
{
    // создаём временный массив для результата
    double*** uinRes = new double** [nz];
    for (int k = 0; k < nz; k++) {
        uinRes[k] = new double* [nr];
        for (int i = 0; i < nr; i++) {
            uinRes[k][i] = new double[nfi]();
        }
    }

    // массивы Фурье-компонент [nfi][nr][nz]
    double*** uinRe = new double** [nfi];
    double*** uinIm = new double** [nfi];
    double*** fRe = new double** [nfi];
    double*** fIm = new double** [nfi];

    for (int m = 0; m < nfi; m++) {
        uinRe[m] = new double* [nr];
        uinIm[m] = new double* [nr];
        fRe[m] = new double* [nr];
        fIm[m] = new double* [nr];
        for (int i = 0; i < nr; i++) {
            uinRe[m][i] = new double[nz]();
            uinIm[m][i] = new double[nz]();
            fRe[m][i] = new double[nz]();
            fIm[m][i] = new double[nz]();
        }
    }

    // рабочий массив [2][nr][nz]
    double*** resReIm = new double** [2];
    for (int t = 0; t < 2; t++) {
        resReIm[t] = new double* [nr];
        for (int i = 0; i < nr; i++) {
            resReIm[t][i] = new double[nz]();
        }
    }

    double* phiSliceU = new double[nfi];
    double* phiSliceF = new double[nfi];

    // прямое FFT по углу φ
    for (int i = 0; i < nr; i++) {
        for (int k = 0; k < nz; k++) {
            for (int j = 0; j < nfi; j++) {
                phiSliceU[j] = uin[k][i][j];
                phiSliceF[j] = f[k][i][j];
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

    // вычисляем λ и ω
    double* lambdas = new double[nfi];
    double* omegas = new double[nfi];
    getLambdas(lambdas, nfi, hfi);
    getOmegas(omegas, lambdas, nfi, nr, nz, hr, hz);


    // решаем по блокам
    for (int m = 0; m < nfi; m++) {
        blockSOR(lambdas[m], m, uinRe[m], uinIm[m], fRe[m], fIm[m],
            hr, hz, omegas[m], eps, resReIm, nr, nfi, nz);

        for (int i = 0; i < nr; i++) {
            for (int k = 0; k < nz; k++) {
                uinRe[m][i][k] = resReIm[0][i][k];
                uinIm[m][i][k] = resReIm[1][i][k];
            }
        }
    }

    delete[] lambdas;
    delete[] omegas;

    double* phiSliceURe = new double[nfi];
    double* phiSliceUim = new double[nfi];

    // обратное FFT по углу φ
    for (int i = 0; i < nr; i++) {
        for (int k = 0; k < nz; k++) {
            for (int m = 0; m < nfi; m++) {
                phiSliceURe[m] = uinRe[m][i][k];
                phiSliceUim[m] = uinIm[m][i][k];
            }

            double* ifftU = ifft_handler(phiSliceURe, phiSliceUim, nfi);
            for (int j = 0; j < nfi; j++) {
                uinRes[k][i][j] = ifftU[j];
            }
            delete[] ifftU;
        }
    }
    delete[] phiSliceURe;
    delete[] phiSliceUim;

    // переносим результат в исходный массив uin
    for (int k = 0; k < nz; k++) {
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < nfi; j++) {
                uin[k][i][j] = uinRes[k][i][j];
            }
        }
    }

    // чистим память
    for (int k = 0; k < nz; k++) {
        for (int i = 0; i < nr; i++) {
            delete[] uinRes[k][i];
        }
        delete[] uinRes[k];
    }
    delete[] uinRes;

    for (int m = 0; m < nfi; m++) {
        for (int i = 0; i < nr; i++) {
            delete[] uinRe[m][i];
            delete[] uinIm[m][i];
            delete[] fRe[m][i];
            delete[] fIm[m][i];
        }
        delete[] uinRe[m];
        delete[] uinIm[m];
        delete[] fRe[m];
        delete[] fIm[m];
    }
    delete[] uinRe;
    delete[] uinIm;
    delete[] fRe;
    delete[] fIm;

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < nr; i++) {
            delete[] resReIm[t][i];
        }
        delete[] resReIm[t];
    }
    delete[] resReIm;
}
