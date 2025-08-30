#define _USE_MATH_DEFINES 
#include "BSOR.h"
#include "FFT.h"
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
	double* vecRI1D = new double[2 * n];
	for (int i = 0; i < n; i++) {
		vecRI1D[2 * i] = vec[i];
		vecRI1D[2 * i + 1] = 0.;
	}
	fft(vecRI1D, n, 0);
	double** dataRI = new double* [2];
	dataRI[0] = new double[n];
	dataRI[1] = new double[n];

	for (int i = 0; i < n; i++) {
		dataRI[0][i] = vecRI1D[2 * i];
		dataRI[1][i] = vecRI1D[2 * i + 1];
	}
	delete[] vecRI1D;
	return dataRI;
}

double* ifft_handler(double* data_RE, double* data_IM, int n) {
	double* vecRI1D = new double[2 * n];
	for (int i = 0; i < n; i++) {
		vecRI1D[2 * i] = data_RE[i];
		vecRI1D[2 * i + 1] = data_IM[i];
	}
	fft(vecRI1D, n, 1);
	double* vecR = new double[n];
	for (int i = 0; i < n; i++) {
		vecR[i] = vecRI1D[2 * i];
	}
	delete[] vecRI1D;
	return vecR;
}

void tridiagonalMatrixSolve(double a, double b, double c, double* d, double* solve, int n) {
	double* alpha = new double[n - 1];
	double* betta = new double[n];

	alpha[0] = -c / b;
	betta[0] = d[0] / b;

	for (int i = 1; i < n - 1; i++) {
		double denom = b + a * alpha[i - 1];
		/*std::cerr << "Bad denom at i=" << i
			<< " b=" << b
			<< " a=" << a
			<< " alpha[i-1]=" << alpha[i - 1]
			<< " denom=" << denom << std::endl;*/
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

	double invHr2 = 1 / (hr*hr);
	double invHz2 = 1 / (hz*hz);
	double a = invHz2;
	double c = a;
	double twoInvHr2 = 2 * invHr2;
	double twoInvHz2 = 2 * invHz2;
	bool stop = 0;
	int q = 0;
	double* dRe = new double[nz - 2];
	double* dIm = new double[nz - 2];
	double* resRe = new double[nz - 2];
	double* resIm = new double[nz - 2];
	while (!stop) {
		stop = 1;

		for (int i = 1; i < nr - 1; i++) {
			double ri = i * hr;
			double inv2RiHr = 1.0 / (2.0 * ri * hr);

			double b = -1 * (twoInvHr2 + (lambda / (ri * ri)) + twoInvHz2);
			//std::cout << b << "\n";
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
			for (int i = 0; i < nz; i++) {
				printf("%lf\n", dRe[i]);
			}
			//Прогонка
			tridiagonalMatrixSolve(a, b, c, dRe, resRe, nz - 2);
			tridiagonalMatrixSolve(a, b, c, dIm, resIm, nz - 2);

			// Релаксация
			for (int k = 1; k < nz - 1; k++) {
				double uOldSorRe = uSorRe[i][k];
				double uOldSorIm = uSorIm[i][k];
				double uNewSorRe = omega * resRe[k - 1] + (1 - omega) * uOldSorRe;
				double uNewSorIm = omega * resIm[k - 1] + (1 - omega) * uOldSorIm;
				std::cout << uOldSorRe << "\n" << uNewSorRe << "\n";
				if (stop && (fabs(uNewSorRe - uOldSorRe) > eps || fabs(uNewSorIm - uOldSorIm) > eps)) {
					stop = false;
				}
				/*printf("Re: %lf\n", fabs(uNewSorRe - uOldSorRe));
				printf("Im: %lf\n", fabs(uNewSorIm - uOldSorIm));*/
				uSorRe[i][k] = uNewSorRe;
				uSorIm[i][k] = uNewSorIm;
			}

			if (m == 0) {
				for (int k = 1; k < nz - 1; k++) {
					uSorRe[0][k] = uSorRe[1][k];
					uSorIm[0][k] = uSorIm[1][k];
				}
			}
			q++;
			//printf("%d\n", q);
		}
	}


	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nz; j++) {
			resReIm[0][i][j] = uSorRe[i][j];
			resReIm[1][i][j] = uSorIm[i][j];
		}
	}
	for (int i = 0; i < nr; i++) {
		delete[] uSorRe[i];
		delete[] uSorIm[i];
	}
	delete[] uSorRe;
	delete[] uSorIm;
	delete[] dRe;
	delete[] dIm;
	delete[] resRe;
	delete[] resIm;
}