#define _USE_MATH_DEFINES 
#include "BSOR.h"
#include "fftw3.h"
#include "iostream"
#include <cmath>
void getLambdas(double* lambdas, int nfi, double hfi) {
	for (int m = 0; m < nfi; m++) {
		int mVal = m <= nfi / 2 ? m : m - nfi;
		lambdas[m] = 4 * pow(sin(mVal * hfi / 2), 2) / pow(hfi, 2);
	}
}
void getOmegas(double* omegas, double* lambdas, int nfi, int nr, int nz, double hr, double hz) {
	double l_min = 4 / pow(hr, 2) * pow(sin(M_PI / (2 * nr + 1)), 2) + 4 / pow(hz, 2) * pow(sin(M_PI / (2 * nz + 1)), 2);
	double l_max = 4 / pow(hr, 2) * pow(sin(nr * M_PI / (2 * nr + 1)), 2) + 4 / pow(hz, 2) * pow(sin(nz * M_PI / (2 * nz + 1)), 2);
	omegas[0] = 2 / (1 + sqrt(1 - pow((l_max - l_min) / (l_max + l_min), 2)));
	for (int i = 1; i < nfi; i++) {
		double l_min_i = l_min + lambdas[i] / (nr * hr);
		double l_max_i = l_max + lambdas[i] / hr;
		omegas[i] = 2 / (1 + sqrt(1 - pow((l_max_i - l_min_i) / (l_max_i + l_min_i), 2)));
	}
}


double** fft_handler(double* vec, int n) {
	fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
	fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);

	for (int i = 0; i < n; i++) {
		in[i][0] = vec[i];
		in[i][1] = 0.0;
	}

	fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

	fftw_execute(plan);

	double** dataRI = new double* [2];
	dataRI[0] = new double[n];
	dataRI[1] = new double[n];

	for (int i = 0; i < n; i++) {
		dataRI[0][i] = out[i][0];
		dataRI[1][i] = out[i][1];
	}

	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);

	return dataRI;
}

double* ifft_handler(double* data_RE, double* data_IM, int n) {
	fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
	fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);

	for (int i = 0; i < n; i++) {
		in[i][0] = data_RE[i];
		in[i][1] = data_IM[i];
	}

	fftw_plan plan = fftw_plan_dft_1d(n, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);

	fftw_execute(plan);

	double* vecR = new double[n];
	for (int i = 0; i < n; i++) {
		vecR[i] = out[i][0] / n;
	}

	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);

	return vecR;
}

void tridiagonalMatrixSolve(double a, double b, double c, double* d, double* solve, int n) {
	double* alpha = new double[n - 1];
	double* betta = new double[n];

	alpha[0] = -c / b;
	betta[0] = d[0] / b;

	for (int i = 1; i < n - 1; i++) {
		double denom = b + a * alpha[i - 1];
		alpha[i] = -c / denom;
		betta[i] = (d[i] - a * betta[i - 1]) / denom;
	}

	betta[n - 1] = (d[n - 1] - a * betta[n - 2]) / (b + a * alpha[n - 2]);

	solve[n - 1] = betta[n - 1];

	for (int i = n - 2; i >= 0; i--) {
		solve[i] = alpha[i] * solve[i + 1] + betta[i];
	}
	delete[] alpha;
	delete[] betta;
}

void blockSOR(double lambda, int m, double** uRe, double** uIm, double** fRe, double** fIm,
	double hr, double hz, double omega, double eps, double*** resReIm, int nr, int nfi, int nz) {

	double** uSorRe = new double* [nr];
	double** uSorIm = new double* [nr];

	for (int i = 0; i < nr; i++) {
		uSorRe[i] = new double[nz];
		uSorIm[i] = new double[nz];
	}

	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nz; j++) {
			uSorRe[i][j] = uRe[i][j];
			uSorIm[i][j] = uIm[i][j];
		}
	}

	double invHr2 = 1 / pow(hr, 2);
	double invHz2 = 1 / pow(hz, 2);
	double a = invHz2;
	double c = a;
	double twoInvHr2 = 2 * invHr2;
	double twoInvHz2 = 2 * invHz2;
	bool stop = 0;

	double* dRe = new double[nz - 2];
	double* dIm = new double[nz - 2];
	while (!stop) {
		stop = 1;

		for (int i = 1; i < nr - 1; i++) {
			double ri = i * hr;
			double inv2RiHr = 1.0 / (2.0 * ri * hr);

			double b = -1 * (twoInvHr2 + (lambda / pow(ri, 2) + twoInvHz2));

			//Заполняем вектора для правой части СЛАУ
			for (int k = 1; k < nz - 1; k++) {
				double diffURe = uSorRe[i + 1][k] - uSorRe[i - 1][k];
				double diffUIm = uSorIm[i + 1][k] - uSorIm[i - 1][k];
				double summURe = uSorRe[i + 1][k] + uSorRe[i - 1][k];
				double summUIm = uSorIm[i + 1][k] + uSorIm[i - 1][k];
				dRe[k - 1] = fRe[i][k] - (summURe * invHr2) - (diffURe * inv2RiHr);
				dIm[k - 1] = fIm[i][k] - (summUIm * invHr2) - (diffUIm * inv2RiHr);
			}

			// Что это? Костыль?
			dRe[0] -= a * uSorRe[i][0];
			dRe[nz - 3] -= a * uSorRe[i][nz - 1];
			dIm[0] -= a * uSorIm[i][0];
			dIm[nz - 3] -= a * uSorIm[i][nz - 1];

			//Прогонка
			double* resRe = new double[nz - 2];
			double* resIm = new double[nz - 2];

			tridiagonalMatrixSolve(a, b, c, dRe, resRe, nz - 2);
			tridiagonalMatrixSolve(a, b, c, dIm, resIm, nz - 2);

			// Релаксация
			for (int k = 1; k < nz - 1; k++) {
				double uOldSorRe = uSorRe[i][k];
				double uOldSorIm = uSorIm[i][k];
				double uNewSorRe = omega * resRe[k - 1] + (1 - omega) * uOldSorRe;
				double uNewSorIm = omega * resIm[k - 1] + (1 - omega) * uOldSorIm;

				if (stop && (fabs(uNewSorRe - uOldSorRe) > eps || fabs(uNewSorIm - uOldSorIm) > eps)) {
					stop = false;
				}

				uSorRe[i][k] = uNewSorRe;
				uSorIm[i][k] = uNewSorIm;
			}

			if (m == 0) {
				for (int k = 1; k < nz - 1; k++) {
					uSorRe[0][k] = uSorRe[1][k];
					uSorIm[0][k] = uSorIm[1][k];
				}
			}
			delete[] resRe;
			delete[] resIm;
		}
	}
	delete[] dRe;
	delete[] dIm;

	for (int i = 0; i < nr; i++) {
		for (int k = 0; k < nz; k++) {
			resReIm[0][i][k] = uSorRe[i][k];
			resReIm[1][i][k] = uSorIm[i][k];
		}
	}

	for (int i = 0; i < nr; ++i) {
		delete[] uSorRe[i];
		delete[] uSorIm[i];
	}
	delete[] uSorRe;
	delete[] uSorIm;
}