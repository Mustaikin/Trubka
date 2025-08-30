#include "FFT.h"
#include <iostream>
#include "cmath"
const double PI = 3.14159265;
void fft(double* a, int n, bool invert) {

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j) {
            std::swap(a[2 * i], a[2 * j]);
            std::swap(a[2 * i + 1], a[2 * j + 1]);
        }
    }


    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        double wlen_re = cos(ang);
        double wlen_im = sin(ang);

        for (int i = 0; i < n; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;
            for (int j = 0; j < len / 2; j++) {
                int u_index = 2 * (i + j);
                int v_index = 2 * (i + j + len / 2);

                double u_re = a[u_index];
                double u_im = a[u_index + 1];

                double v_re = a[v_index] * w_re - a[v_index + 1] * w_im;
                double v_im = a[v_index] * w_im + a[v_index + 1] * w_re;

      
                a[u_index] = u_re + v_re;
                a[u_index + 1] = u_im + v_im;

      
                a[v_index] = u_re - v_re;
                a[v_index + 1] = u_im - v_im;


                double tmp_re = w_re * wlen_re - w_im * wlen_im;
                double tmp_im = w_re * wlen_im + w_im * wlen_re;
                w_re = tmp_re;
                w_im = tmp_im;
            }
        }
    }

    if (invert) {
        for (int i = 0; i < 2 * n; i++)
            a[i] /= n;
    }
}