#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <stdio.h>
#include "mpi.h"
#include <fstream>
#include <iostream>

#define _MAX_(X,Y) ((X)>(Y)?(X):(Y))
#define _MIN_(X,Y) ((X)<(Y)?(X):(Y))

#ifdef _MY_DBG_
#define DBG_ECHO(str) printf("Line: %d, msg: %s\n", __LINE__, str)
#else
#define DBG_ECHO(str) (void)0
#endif

int nstep;
int     rnk,
rnk_l,
rnk_r,
rnk_b,
rnk_t,
rnk_fi0,
rnk_fi,
rnk_ij[3],
size,
sz_x = 4,
sz_y = 1,
sz_z = 1;

double* buf_d;
int* buf_i;


const int G_NX = 128;                 //*< кол-во ячеек вдоль оси Z
const int G_NY = 10;                 //*< кол-во ячеек вдоль оси R
const int G_NZ = 32;                 //*< кол-во ячеек по углу
const int K = 1;                   //*< кол-во фикт. ячеек
const int FLD_COUNT = 100;           //*< кол-во переменных


const double G_D_MIN[3] = { 0.0, 0.0, 0.0 }; //*< domaine min coordinates
const double G_D_MAX[3] = { 0.120, 0.012, 2 * M_PI }; //*< domaine max coordinates
const double D_Vh = 0.024;  //диаметр трубы для входящего потока


const double z_center = 0.06; // точки на вершинах окружности по оси Z

const double radius = 0.05;
int rank_l, rank_r, rank_b, rank_t, rank_fi0, rank_fi;

int NX = G_NX;                 //*< кол-во ячеек вдоль оси Z на одном процессоре
int NY = G_NY;                 //*< кол-во ячеек вдоль оси R на одном процессоре
int NZ = G_NZ;                 //*< кол-во ячеек ячеек по углу на одном процессоре
int TOTAL_NX = NX + K * 2;
int TOTAL_NY = NY + K * 2;
int TOTAL_NZ = NZ + K * 2;

int BUF_SIZE = 0;

double D_MIN[3] = { G_D_MIN[0], G_D_MIN[1], G_D_MIN[2] }; //*< domaine min coordinates
double D_MAX[3] = { G_D_MAX[0], G_D_MAX[1], G_D_MAX[2] }; //*< domaine max coordinates


double DX[3] = { (D_MAX[0] - D_MIN[0]) / NX, (D_MAX[1] - D_MIN[1]) / NY, (D_MAX[2] - D_MIN[2]) / NZ };


const double T_MAX = 10;

const double TAU = 1.e-6;

const int SAVE_STEP = 1;
const int SAVE_TIME = 1000;

const int nMat = 2;//*<количество веществ
const int OsnV = 1; //*<количество основных веществ

const double gR = 8.314472;	//*< универсальная газовая постоянная

const double TRef = 298.15;

const double temp_in = 973.0; // температура втекающего газа метана
const double temp_0 = 873.15; // температура в нач момент времени
const double temp_bound = 1273.0; // температура нагревательного элемента
const double temp_bound_circle = 3000; // температура на границе дополнительной трубки
const double p0 = 101325.0; //давление на вытекании
const double vMixt = 5.0 * 2.2e-6; //заданный объемный расход смеси м3/с


double*** ro, *** ro_1, *** ru, *** rv, *** rw, *** rh, *** temp, *** temp_1, *** pidin, *** pidin_1, **** ry, **** ry_1; //*< консервативные переменные
double*** fluxx_ro, *** fluxx_ru, *** fluxx_rv, *** fluxx_rh, *** fluxx_rw, **** fluxx_ry;
double*** fluxy_ro, *** fluxy_ru, *** fluxy_rv, *** fluxy_rh, *** fluxy_rw, **** fluxy_ry;
double*** fluxz_ro, *** fluxz_ru, *** fluxz_rv, *** fluxz_rh, *** fluxz_rw, **** fluxz_ry;
double*** dv, *** du, *** dw, *** dpx, *** dpy, *** dpy1, *** dpz;
double*** S; // поправка скорости
double*** u, *** v, *** w, * y, * yr, * yl, * hii, * hir, * hil; // скорость сделать двухмерную
double*** QFx, *** QFy, *** QFz;

int lo[3], hi[3];              //*< нижняя и верхняя границы для [0] - X; [1] - Y [2] - FI

double Mm[nMat];   //*<молярная масса
double Mm1[nMat];   //*<молекулярная масса
double h0[nMat];   //*<энтальпия образования вещества

int         n_TCP; //!< количество слагаемых в Cp
double** k_TCP; //!< коэффициенты в выражениях Cp 
double* element_min_rank, * element_max_rank;
/**
 * инициализация параллельного окружения
 */
void init_parallel(int* argc, char*** argv);
void done_parallel();
/**
 * Межпроцессорный обмен
 */
void bnd_exch();
/**
 * Межпроцессорный обмен для указанной величины
 */
void exchange_fld(double*** fld);

/**
 * Выделение памяти
 */
void mem_alloc();

/**
 * Освобождение памяти
 */
void mem_free();

/**
 * Вывод в VTK-файл
 */
void save_vtk(int num);
/**
 * Начальные данные
 */
void init();
/**
 * Граничные условия
 */
void bnd_cond();
void bnd_cond_p();

/**
 * Вычисление потоков на границах ячеек
 */
void calc_fluxes();
void calc_fluxes1();

/**
 * Вычисление значений на новом временном слое
 */
void calc_new_fields();
void calc_new_pi();
/**
 * Один шаг по времени
 */
void calc_one_time_step();
/**
 * Теплоемкость при постоянном давлении
 */
double Calc_CP(int iM, double fT);
/**
 * Функция расчета производной теплоемкости
 */
double Calc_CP_(int iM, double fT);
/**
 * Уравнение состояния смеси идеальных газов
 */
void urs_mixt(double* y, double h, double p, double fT, int flag,
	double& tmp, double& gamma, double& r, double& h_);
/**
 * Вычисление температуры итерациями по Ньютону
 */
void Newton_Method(double e, double* y, double tmp1,
	double& tmp);

/**
 * Вычисление динамической составляющей давления
 */
void calc_pi();
/**
 * Вычисление корректирующей составляющей скорости
 */
void calc_s();

double Calc_ML_Nv(double* y, double tmp);
double Calc_KP_Nv(double* y, double tmp);
/**
 * Вычисление энтальпии
 */
void calc_h(double tmp,
	double hii[nMat]);

void ZhWENO(double* UU,
	double& Um, double& Up);

int step;
int main(int argc, char** argv) {
	init_parallel(&argc, &argv);
	mem_alloc();
	init();
	step = 0;
	save_vtk(0);
	double t = 0.;

	double starttime, endtime;
	starttime = MPI_Wtime();

	while (t < T_MAX) {
		t += TAU;
		step++;
		printf("%lf\n", t);
		fflush(stdout);
		calc_one_time_step();
		if (step - SAVE_TIME == 0) {
			endtime = MPI_Wtime();
			FILE* fp;
			fp = fopen("time.txt", "w");
			double tt = endtime - starttime;
			fprintf(fp, "%f ", tt);
			fclose(fp);
		}
		if (step % SAVE_STEP == 0) {
			save_vtk(step);
		}
	}
	done_parallel();
	mem_free();
	return 0;
}

void init_parallel(int* argc, char*** argv)
{
	MPI_Init(argc, argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
	if (size != sz_x * sz_y * sz_z) {
		fprintf(stderr, "Wrong number of processes.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
		exit(1);
	}

	NX = G_NX / sz_x;
	NY = G_NY / sz_y;
	NZ = G_NZ / sz_z;
	TOTAL_NX = NX + K * 2;
	TOTAL_NY = NY + K * 2;
	TOTAL_NZ = NZ + K * 2;

	rnk_l = (rnk % sz_x != 0) ? rnk - 1 : -1;
	rnk_r = (rnk % sz_x != sz_x - 1) ? rnk + 1 : -1;
	rnk_t = (rnk / sz_x != sz_y - 1) ? rnk + sz_x : -1;
	rnk_b = (rnk / sz_x != 0) ? rnk - sz_x : -1;
	rnk_fi0 = (rnk - sz_x * sz_y >= 0) ? rnk - sz_x * sz_y : -1;
	rnk_fi = (rnk + sz_x * sz_y <= sz_x * sz_y * sz_z - 1) ? rnk + sz_x * sz_y : -1;


	/*	near(1) = myid -1                           !! left
		if (mod(myid,NBx).eq.0) near(1) = -1

			 near(2) = myid +1                           !! right
			 if (mod(myid+1,NBx).eq.0) near(2) = -1

			 near(3) = myid -NBx                         !! back
			 if (near(3).lT. myid/npxy *npxy) near(3) = -1

			 near(4) = myid +NBx                         !! front
			 if (near(4).ge.(myid/npxy+1) *npxy) near(4) = -1

			 near(5) = myid -npxy                        !! down
			 if (near(5).lt.0) near(5) = -1

			 near(6) = myid +npxy                        !! up
			 if (near(6).gt.NBx*NBy*NBz - 1) near(6) = -1*/


			 //исправить!!!

	rnk_ij[0] = rnk % sz_x;
	rnk_ij[1] = rnk / sz_x;

	double g_sz[2] = { G_D_MAX[0] - G_D_MIN[0], G_D_MAX[1] - G_D_MIN[1] };
	double l_sz[2] = { g_sz[0] / sz_x, g_sz[1] / sz_y };
	for (int i = 0; i < 2; i++) {
		D_MIN[i] = G_D_MIN[i] + l_sz[i] * rnk_ij[i];
		D_MAX[i] = D_MIN[i] + l_sz[i];
	}
	DX[0] = (D_MAX[0] - D_MIN[0]) / NX;
	DX[1] = (D_MAX[1] - D_MIN[1]) / NY;
	DX[2] = (D_MAX[2] - D_MIN[2]) / NZ;

	BUF_SIZE = _MAX_(NX * K * FLD_COUNT * NY * K * FLD_COUNT, _MAX_(NX * K * FLD_COUNT * NZ * K * FLD_COUNT, NZ * K * FLD_COUNT * NY * K * FLD_COUNT));
}

void done_parallel()
{
	MPI_Finalize();
}


void mem_alloc()
{
	lo[0] = K;
	lo[1] = K;
	lo[2] = K;

	hi[0] = K + NX - 1;
	hi[1] = K + NY - 1;
	hi[2] = K + NZ - 1;

	ro = new double** [TOTAL_NX];
	ro_1 = new double** [TOTAL_NX];
	ru = new double** [TOTAL_NX];
	rv = new double** [TOTAL_NX];
	rh = new double** [TOTAL_NX];
	rw = new double** [TOTAL_NX];
	temp = new double** [TOTAL_NX];
	temp_1 = new double** [TOTAL_NX];
	pidin = new double** [TOTAL_NX];
	pidin_1 = new double** [TOTAL_NX];
	S = new double** [TOTAL_NX];
	for (int i = 0; i < TOTAL_NX; i++) {
		ro[i] = new double* [TOTAL_NY];
		ro_1[i] = new double* [TOTAL_NY];
		ru[i] = new double* [TOTAL_NY];
		rv[i] = new double* [TOTAL_NY];
		rh[i] = new double* [TOTAL_NY];
		rw[i] = new double* [TOTAL_NY];
		temp[i] = new double* [TOTAL_NY];
		temp_1[i] = new double* [TOTAL_NY];
		pidin[i] = new double* [TOTAL_NY];
		pidin_1[i] = new double* [TOTAL_NY];
		S[i] = new double* [TOTAL_NY];
		for (int j = 0; j < TOTAL_NY; j++) {
			ro[i][j] = new double[TOTAL_NZ];
			ro_1[i][j] = new double[TOTAL_NZ];
			ru[i][j] = new double[TOTAL_NZ];
			rv[i][j] = new double[TOTAL_NZ];
			rh[i][j] = new double[TOTAL_NZ];
			rw[i][j] = new double[TOTAL_NZ];
			temp[i][j] = new double[TOTAL_NZ];
			temp_1[i][j] = new double[TOTAL_NZ];
			pidin[i][j] = new double[TOTAL_NZ];
			pidin_1[i][j] = new double[TOTAL_NZ];
			S[i][j] = new double[TOTAL_NZ];
		}
	}

	QFx = new double** [TOTAL_NX + 1];
	QFy = new double** [TOTAL_NX];
	QFz = new double** [TOTAL_NX];


	for (int i = 0; i < TOTAL_NX + 1; i++) {
		QFx[i] = new double* [TOTAL_NY];
		for (int j = 0; j < TOTAL_NY; j++) {
			QFx[i][j] = new double[TOTAL_NZ];
		}
	}
	for (int i = 0; i < TOTAL_NX; i++) {
		QFy[i] = new double* [TOTAL_NY + 1];
		for (int j = 0; j < TOTAL_NY + 1; j++) {
			QFy[i][j] = new double[TOTAL_NZ];
		}
	}
	for (int i = 0; i < TOTAL_NX; i++) {
		QFz[i] = new double* [TOTAL_NY];
		for (int j = 0; j < TOTAL_NY; j++) {
			QFz[i][j] = new double[TOTAL_NZ + 1];
		}
	}

	ry = new double*** [nMat];
	ry_1 = new double*** [nMat];
	for (int i = 0; i < nMat; i++) {
		ry[i] = new double** [TOTAL_NX];
		ry_1[i] = new double** [TOTAL_NX];
		for (int j = 0; j < TOTAL_NX; j++) {
			ry[i][j] = new double* [TOTAL_NY];
			ry_1[i][j] = new double* [TOTAL_NY];
			for (int k = 0; k < TOTAL_NY; k++) {
				ry[i][j][k] = new double[TOTAL_NZ];
				ry_1[i][j][k] = new double[TOTAL_NZ];
			}
		}
	}

	fluxx_ro = new double** [TOTAL_NX + 1];
	fluxx_ru = new double** [TOTAL_NX + 1];
	fluxx_rv = new double** [TOTAL_NX + 1];
	fluxx_rh = new double** [TOTAL_NX + 1];
	fluxx_rw = new double** [TOTAL_NX + 1];
	for (int i = 0; i < TOTAL_NX + 1; i++) {
		fluxx_ro[i] = new double* [TOTAL_NY];
		fluxx_ru[i] = new double* [TOTAL_NY];
		fluxx_rv[i] = new double* [TOTAL_NY];
		fluxx_rh[i] = new double* [TOTAL_NY];
		fluxx_rw[i] = new double* [TOTAL_NY];
		for (int j = 0; j < TOTAL_NY + 1; j++) {
			fluxx_ro[i][j] = new double[TOTAL_NZ];
			fluxx_ru[i][j] = new double[TOTAL_NZ];
			fluxx_rv[i][j] = new double[TOTAL_NZ];
			fluxx_rh[i][j] = new double[TOTAL_NZ];
			fluxx_rw[i][j] = new double[TOTAL_NZ];

		}
	}

	fluxx_ry = new double*** [nMat];
	for (int i = 0; i < nMat; i++) {
		fluxx_ry[i] = new double** [TOTAL_NX + 1];
		for (int j = 0; j < TOTAL_NX + 1; j++) {
			fluxx_ry[i][j] = new double* [TOTAL_NY];
			for (int k = 0; k < TOTAL_NY + 1; k++) {
				fluxx_ry[i][j][k] = new double[TOTAL_NZ];
			}
		}
	}

	fluxy_ro = new double** [TOTAL_NX];
	fluxy_ru = new double** [TOTAL_NX];
	fluxy_rv = new double** [TOTAL_NX];
	fluxy_rh = new double** [TOTAL_NX];
	fluxy_rw = new double** [TOTAL_NX];
	for (int i = 0; i < TOTAL_NX; i++) {
		fluxy_ro[i] = new double* [TOTAL_NY + 1];
		fluxy_ru[i] = new double* [TOTAL_NY + 1];
		fluxy_rv[i] = new double* [TOTAL_NY + 1];
		fluxy_rh[i] = new double* [TOTAL_NY + 1];
		fluxy_rw[i] = new double* [TOTAL_NY + 1];
		for (int j = 0; j < TOTAL_NY; j++) {
			fluxy_ro[i][j] = new double[TOTAL_NZ];
			fluxy_ru[i][j] = new double[TOTAL_NZ];
			fluxy_rv[i][j] = new double[TOTAL_NZ];
			fluxy_rh[i][j] = new double[TOTAL_NZ];
			fluxy_rw[i][j] = new double[TOTAL_NZ];
		}
	}

	fluxy_ry = new double*** [nMat];
	for (int i = 0; i < nMat; i++) {
		fluxy_ry[i] = new double** [TOTAL_NX];
		for (int j = 0; j < TOTAL_NX; j++) {
			fluxy_ry[i][j] = new double* [TOTAL_NY + 1];
			for (int k = 0; k < TOTAL_NY; k++) {
				fluxy_ry[i][j][k] = new double[TOTAL_NZ + 1];
			}
		}
	}

	fluxz_ro = new double** [TOTAL_NX];
	fluxz_ru = new double** [TOTAL_NX];
	fluxz_rv = new double** [TOTAL_NX];
	fluxz_rh = new double** [TOTAL_NX];
	fluxz_rw = new double** [TOTAL_NX];
	for (int i = 0; i < TOTAL_NX; i++) {
		fluxz_ro[i] = new double* [TOTAL_NY];
		fluxz_ru[i] = new double* [TOTAL_NY];
		fluxz_rv[i] = new double* [TOTAL_NY];
		fluxz_rh[i] = new double* [TOTAL_NY];
		fluxz_rw[i] = new double* [TOTAL_NY];
		for (int j = 0; j < TOTAL_NY; j++) {
			fluxz_ro[i][j] = new double[TOTAL_NZ + 1];
			fluxz_ru[i][j] = new double[TOTAL_NZ + 1];
			fluxz_rv[i][j] = new double[TOTAL_NZ + 1];
			fluxz_rh[i][j] = new double[TOTAL_NZ + 1];
			fluxz_rw[i][j] = new double[TOTAL_NZ + 1];
		}
	}

	fluxz_ry = new double*** [nMat];
	for (int i = 0; i < nMat; i++) {
		fluxz_ry[i] = new double** [TOTAL_NX];
		for (int j = 0; j < TOTAL_NX; j++) {
			fluxz_ry[i][j] = new double* [TOTAL_NY + 1];
			for (int k = 0; k < TOTAL_NY; k++) {
				fluxz_ry[i][j][k] = new double[TOTAL_NZ];
			}
		}
	}

	u = new double** [TOTAL_NX];
	v = new double** [TOTAL_NX];
	w = new double** [TOTAL_NX];
	for (int i = 0; i < TOTAL_NX; i++) {
		u[i] = new double* [TOTAL_NY];
		v[i] = new double* [TOTAL_NY];
		w[i] = new double* [TOTAL_NY];
		for (int j = 0; j < TOTAL_NY; j++) {
			u[i][j] = new double[TOTAL_NZ];
			v[i][j] = new double[TOTAL_NZ];
			w[i][j] = new double[TOTAL_NZ];
		}
	}
	du = new double** [TOTAL_NX];
	dv = new double** [TOTAL_NX];
	dw = new double** [TOTAL_NX];
	for (int i = 0; i < TOTAL_NX; i++) {
		du[i] = new double* [TOTAL_NY];
		dv[i] = new double* [TOTAL_NY];
		dw[i] = new double* [TOTAL_NY];
		for (int j = 0; j < TOTAL_NY; j++) {
			du[i][j] = new double[TOTAL_NZ];
			dv[i][j] = new double[TOTAL_NZ];
			dw[i][j] = new double[TOTAL_NZ];
		}
	}
	dpx = new double** [TOTAL_NX];
	dpy = new double** [TOTAL_NX];
	dpy1 = new double** [TOTAL_NX];
	dpz = new double** [TOTAL_NX];
	for (int i = 0; i < TOTAL_NX; i++) {
		dpx[i] = new double* [TOTAL_NY];
		dpy[i] = new double* [TOTAL_NY];
		dpy1[i] = new double* [TOTAL_NY];
		dpz[i] = new double* [TOTAL_NY];
		for (int j = 0; j < TOTAL_NY; j++) {
			dpx[i][j] = new double[TOTAL_NZ];
			dpy[i][j] = new double[TOTAL_NZ];
			dpy1[i][j] = new double[TOTAL_NZ];
			dpz[i][j] = new double[TOTAL_NZ];
		}
	}
	y = new double[nMat];
	yl = new double[nMat];
	yr = new double[nMat];
	hii = new double[nMat];
	hil = new double[nMat];
	hir = new double[nMat];

	buf_d = new double[BUF_SIZE];
	buf_i = new int[BUF_SIZE];
	element_min_rank = new double[size];
	element_max_rank = new double[size];
}
void mem_free()
{
	for (int i = 0; i < TOTAL_NX; i++) {
		for (int j = 0; j < TOTAL_NY; j++) {
			delete[] ro[i][j];
			delete[] ro_1[i][j];
			delete[] ru[i][j];
			delete[] rv[i][j];
			delete[] rh[i][j];
			delete[] rw[i][j];
			delete[] temp[i][j];
			delete[] temp_1[i][j];
			delete[] pidin[i][j];
			delete[] pidin_1[i][j];
			delete[] S[i][j];
		}
		delete[] ro[i];
		delete[] ro_1[i];
		delete[] ru[i];
		delete[] rv[i];
		delete[] rh[i];
		delete[] rw[i];
		delete[] temp[i];
		delete[] temp_1[i];
		delete[] pidin[i];
		delete[] pidin_1[i];
		delete[] S[i];
	}
	delete[] ro;
	delete[] ro_1;
	delete[] ru;
	delete[] rv;
	delete[] rh;
	delete[] rw;
	delete[] temp;
	delete[] temp_1;
	delete[] pidin;
	delete[] pidin_1;
	delete[] S;

	for (int i = 0; i < nMat; i++) {
		for (int j = 0; j < TOTAL_NX; j++) {
			for (int k = 0; k < TOTAL_NY; k++) {
				delete[] ry[i][j][k];
				delete[] ry_1[i][j][k];
			}
			delete[] ry[i][j];
			delete[] ry_1[i][j];
		}
		delete[] ry[i];
		delete[] ry_1[i];
	}
	delete[] ry;
	delete[] ry_1;

	for (int i = 0; i < TOTAL_NX; i++) {
		delete[] u[i];
		delete[] v[i];
	}
	delete[] u;
	delete[] v;
	for (int i = 0; i < TOTAL_NX; i++) {
		delete[] du[i];
		delete[] dv[i];
	}
	delete[] du;
	delete[] dv;
	for (int i = 0; i < TOTAL_NX; i++) {
		delete[] dpx[i];
		delete[] dpy[i];
		delete[] dpy1[i];
	}
	delete[] dpx;
	delete[] dpy;
	delete[] dpy1;
	delete[] y;
	delete[] yr;
	delete[] yl;
	delete[] hii;
	delete[] hil;
	delete[] hir;

}

void save_vtk(int num) {
	char fName[50];
	sprintf(fName, "res_%014d_%04d.vts", num, rnk);
	FILE* fp = fopen(fName, "w");

	if (!fp) {
		printf("Error opening file: %s\n", fName);
		return;
	}

	//NX - кол-во ячеек по Z
	//NY - кол-во ячеек по R
	//NZ - кол-во ячеек по углу
	int x_min = rnk_ij[0] * NX; // по Z
	int x_max = (rnk_ij[0] + 1) * NX;
	int y_min = rnk_ij[1] * NY; // по R
	int y_max = (rnk_ij[1] + 1) * NY;
	int z_min = rnk_ij[2] * NZ;
	int z_max = (rnk_ij[2] + 1) * NZ; // по углу

	fprintf(fp, "<?xml version=\"1.0\"?>\n");
	fprintf(fp, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
	fprintf(fp, "  <StructuredGrid WholeExtent=\"%d %d %d %d %d %d\">\n",
		x_min, x_max, z_min, z_max, y_min, y_max);
	fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n",
		x_min, x_max, z_min, z_max, y_min, y_max);


	// Координаты точек
	fprintf(fp, "    <Points>\n");
	fprintf(fp, "      <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");

	double hugol = 2 * M_PI / (NZ);

	for (int i = y_min; i <= y_max; i++) {
		for (int j = z_min; j <= z_max; j++) {
			for (int k = x_min; k <= x_max; k++) {
				double x = (i * DX[1]) * cos(j * hugol);
				double y = (i * DX[1]) * sin(j * hugol);
				double z = (k * DX[0]);
				fprintf(fp, "%f %f %f\n", z, y, x);
			}
		}
	}


	fprintf(fp, "      </DataArray>\n");
	fprintf(fp, "    </Points>\n");


	fprintf(fp, "    <CellData Scalars=\"scalars\">\n");

	// Сохранение температуры
	fprintf(fp, "      <DataArray type=\"Float32\" Name=\"Temperature\" format=\"ascii\">\n");
	for (int i = lo[1]; i <= hi[1]; i++) {
		for (int j = lo[2]; j <= hi[2]; j++) {
			for (int k = lo[0]; k <= hi[0]; k++) {
				//	if (std::isnan(temp[k][i][j])) {
				//		abort();
				//	
				fprintf(fp, "%20.10f\n", temp[k][i][j]);

			}
		}
	}
	fprintf(fp, "      </DataArray>\n");

	// Сохранение плотности
	fprintf(fp, "      <DataArray type=\"Float32\" Name=\"Density\" format=\"ascii\">\n");
	for (int i = lo[1]; i <= hi[1]; i++) {
		for (int j = lo[2]; j <= hi[2]; j++) {
			for (int k = lo[0]; k <= hi[0]; k++) {
				//	if (std::isnan(ro[k][i][j])) {
				//		abort();
				//	}
				fprintf(fp, "%20.10f\n", ro[k][i][j]);
			}
		}
	}
	fprintf(fp, "      </DataArray>\n");

	// Сохранение C_H4
	fprintf(fp, "      <DataArray type=\"Float32\" Name=\"MassFraction_CH4\" format=\"ascii\">\n");
	for (int i = lo[1]; i <= hi[1]; i++) {
		for (int j = lo[2]; j <= hi[2]; j++) {
			for (int k = lo[0]; k <= hi[0]; k++) {
				//if (std::isnan(ry[0][k][i][j] / ro[k][i][j])) {
				//	abort();
				//}
				fprintf(fp, "%20.10f\n", ry[0][k][i][j] / ro[k][i][j]);
			}
		}
	}
	fprintf(fp, "      </DataArray>\n");


	//Сохранение С2_H4
	fprintf(fp, "      <DataArray type=\"Float32\" Name=\"MassFraction_C2_H4\" format=\"ascii\">\n");
	for (int i = lo[1]; i <= hi[1]; i++) {
		for (int j = lo[2]; j <= hi[2]; j++) {
			for (int k = lo[0]; k <= hi[0]; k++) {
				//if (std::isnan(ry[1][k][i][j] / ro[k][i][j])) {
				//	abort();
				//}
				fprintf(fp, "%20.10f\n", ry[1][k][i][j] / ro[k][i][j]);
			}
		}
	}
	fprintf(fp, "      </DataArray>\n");

	//Сохранение скорости
	fprintf(fp, "      <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
	for (int i = lo[1]; i <= hi[1]; i++) {
		for (int j = lo[2]; j <= hi[2]; j++) {
			for (int k = lo[0]; k <= hi[0]; k++) {
				//if (std::isnan(ru[k][i][j] / ro[k][i][j] *
				//	rv[k][i][j] / ro[k][i][j] *
				//	rw[k][i][j] / ro[k][i][j])) {
				//	abort();
				//}
				fprintf(fp, "%20.10f %20.10f %20.10f\n",
					ru[k][i][j] / ro[k][i][j],
					rv[k][i][j] / ro[k][i][j],
					rw[k][i][j] / ro[k][i][j]);
			}
		}
	}
	fprintf(fp, "      </DataArray>\n");

	//сохранение давления
	fprintf(fp, "      <DataArray type=\"Float32\" Name=\"Pressure\" format=\"ascii\">\n");
	for (int i = lo[1]; i <= hi[1]; i++) {
		for (int j = lo[2]; j <= hi[2]; j++) {
			for (int k = lo[0]; k <= hi[0]; k++) {
				//	if (std::isnan(pidin[k][i][j])) {
				//		abort();
				//	}
				fprintf(fp, "%20.10f\n", pidin[k][i][j] + p0);
			}
		}
	}
	fprintf(fp, "      </DataArray>\n");

	fprintf(fp, "    </CellData>\n");
	fprintf(fp, "  </Piece>\n");
	fprintf(fp, "  </StructuredGrid>\n");
	fprintf(fp, "</VTKFile>\n");

	fclose(fp);
	printf("File '%s' saved...\n", fName);

	if (rnk == 0) {
		sprintf(fName, "res_%014d.pvts", num);
		fp = fopen(fName, "w");
		fprintf(fp, "<?xml version=\"1.0\"?>\n");
		fprintf(fp, "<VTKFile type=\"PStructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
		fprintf(fp, "  <PStructuredGrid WholeExtent=\"%d %d %d %d %d %d\" GhostLevel=\"0\">\n", 0, G_NX, 0, G_NZ, 0, G_NY);
		fprintf(fp, "    <PPoints>\n<PDataArray type=\"Float32\" NumberOfComponents = \"3\" />\n</PPoints>\n");
		fprintf(fp, "		<PCellData Scalars=\"scalars\">\n");
		fprintf(fp, "			<PDataArray type=\"Float32\" Name=\"Temperature\" format=\"ascii\" />\n");
		fprintf(fp, "			<PDataArray type=\"Float32\" Name=\"Density\" format=\"ascii\" />\n");
		fprintf(fp, "			<PDataArray type=\"Float32\" Name=\"MassFraction_CH4\" format=\"ascii\" />\n");
		fprintf(fp, "			<PDataArray type=\"Float32\" Name=\"MassFraction_C2_H4\" format=\"ascii\" />\n");
		fprintf(fp, "			<PDataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\" />\n");
		fprintf(fp, "			<PDataArray type=\"Float32\" Name=\"Pressure\" format=\"ascii\" />\n");
		fprintf(fp, "		</PCellData>\n");

		for (int pid = 0; pid < size; pid++) {
			int i = pid % sz_x;
			int j = pid / sz_x;
			fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\" Source=\"res_%014d_%04d.vts\"/>\n", i * NX, (i + 1) * NX, 0 * NZ, (0 + 1) * NZ, j * NY, (j + 1) * NY, num, pid);
		}
		fprintf(fp, "  </PStructuredGrid>\n");

		fprintf(fp, "</VTKFile>\n");
		fclose(fp);

		printf("File '%s' saved...\n", fName);
	}
}


void init() {
	//расчитываем миниальный и максимальный диапазон ячеек для каждого процесса

	// читаем данные о коэффициентах для расчёта Сp
	int n;
	FILE* f = fopen("Cp.dat", "r");
	if (!f) {
		perror("Ошибка открытия файла Cp.dat");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	fscanf(f, "%d %d", &n, &n_TCP);

	k_TCP = new double* [nMat];
	for (int i = 0; i < nMat; i++) k_TCP[i] = new double[n_TCP];
	for (int i = 0; i < nMat; i++)
	{
		for (int j = 0; j < n_TCP; j++) fscanf(f, "%lf", &k_TCP[i][j]);
	}
	fclose(f);

	Mm1[0] = 16.04303; //methane
	Mm1[1] = 28.05418; //ehylene

	Mm[0] = 0.01604303; h0[0] = -7.489518e4 / Mm[0]; //methane
	Mm[1] = 0.02805418; h0[1] = 5.24554e4 / Mm[1]; //ethylene

	double r, p, u, v, e, gam, tmp, h_, w;
	for (int i = lo[0] - 1; i <= hi[0] + 1; i++) {
		for (int j = lo[1] - 1; j <= hi[1] + 1; j++) {
			for (int fi = lo[2] - 1; fi <= hi[2] + 1; fi++) {

				temp[i][j][fi] = temp_0;

				p = p0;
				u = 0.0;
				v = 0.0;
				w = 0.0;
				for (int iM = 0; iM < nMat; iM++) y[iM] = 0.0;
				y[0] = 1.;

				urs_mixt(y, 0.0, p, temp[i][j][fi], 2,
					tmp, gam, r, h_);

				ro[i][j][fi] = r;
				ru[i][j][fi] = r * u;
				rv[i][j][fi] = r * v;
				rh[i][j][fi] = r * h_;
				rw[i][j][fi] = r * w;
				pidin[i][j][fi] = 0.0;
				for (int iM = 0; iM < nMat; iM++)
					ry[iM][i][j][fi] = r * y[iM];
			}
		}
	}
}


void bnd_cond()
{
	double r, p, h_, tmp, gamma;
	double S_Vx = M_PI * (D_Vh / 2.0) * (D_Vh / 2.0);
	int fi_oz;
	double z_c, x_c, y_c;
	double hugol = 2 * M_PI / (NZ);
	double z_prev = z_center - radius, z_next = z_center + radius;
	double eps = 0.0001;
	double a = 0.2, b = 0.2;
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int fi = lo[2]; fi <= hi[2]; fi++) {
			double x1 = D_MIN[0] + (i - K) * DX[0];
			if (rnk_b == -1) {
				if (fi <= NZ / 2) fi_oz = fi + NZ / 2;
				else fi_oz = fi - NZ / 2;
				//нижняя стенка
				for (int j = 0; j < K; j++) { //составляющую давления отражаю
					pidin[i][lo[1] - j - 1][fi] = pidin[i][lo[1]][fi_oz];
				}
				//нижняя стенка - отражение (v=0)
				for (int j = 0; j < K; j++) {
					ro[i][lo[1] - j - 1][fi] = ro[i][lo[1]][fi_oz];
					ru[i][lo[1] - j - 1][fi] = ru[i][lo[1]][fi_oz];
					rv[i][lo[1] - j - 1][fi] = rv[i][lo[1]][fi_oz];
					rw[i][lo[1] - j - 1][fi] = rw[i][lo[1]][fi_oz];
					rh[i][lo[1] - j - 1][fi] = rh[i][lo[1]][fi_oz];
					for (int iM = 0; iM < nMat; iM++)
						ry[iM][i][lo[1] - j - 1][fi] = ry[iM][i][lo[1]][fi_oz];
				}
				//нижняя стенка температура
				for (int j = 0; j < K; j++) {
					temp[i][lo[1] - j - 1][fi] = temp[i][lo[1]][fi_oz];
				}
			}
			if (rnk_t == -1) {

				double z_start = rnk * DX[0] * NX;
				double z_end = (rnk + 1) * DX[0] * NX;
				double z_c = i * DX[0] + rnk * DX[0] * NX;

				for (int j = 0; j < K; j++) {
					if (z_prev <= z_c && z_c <= z_next) {
						x_c = G_D_MAX[1] * cos(fi * hugol);
						y_c = G_D_MAX[1] * sin(fi * hugol);
						if (pow(x_c, 2) / pow(a, 2) + pow(z_c - z_center, 2) / pow(b, 2) - pow(y_c, 2) <= eps && y_c > 0) {
							temp[i][hi[1] + j + 1][fi] = temp_bound_circle;
							printf("popal!\n");
						}
						else {
							temp[i][hi[1] + j + 1][fi] = temp_0;
						}
					}
					else {
						temp[i][hi[1] + j + 1][fi] = temp_0;
					}
				}

				//верхняя стенка
				for (int j = 0; j < K; j++) { //составляющую давления везде отражаю
					pidin[i][hi[1] + j + 1][fi] = pidin[i][hi[1] - j][fi];
				}
				//верхняя стенка - непротекание
				for (int j = 0; j < K; j++) {
					ro[i][hi[1] + j + 1][fi] = ro[i][hi[1] - j][fi];
					ru[i][hi[1] + j + 1][fi] = -ru[i][hi[1] - j][fi];
					rv[i][hi[1] + j + 1][fi] = -rv[i][hi[1] - j][fi];
					rw[i][hi[1] + j + 1][fi] = -rw[i][hi[1] - j][fi];
					rh[i][hi[1] + j + 1][fi] = rh[i][hi[1] - j][fi];
					for (int iM = 0; iM < nMat; iM++)
						ry[iM][i][hi[1] + j + 1][fi] = ry[iM][i][hi[1] - j][fi];
				}
				//верхняя стенка температура



			}
		}
	}

	// торцевые стенки
	for (int j = lo[1]; j <= hi[1]; j++) {
		for (int fi = lo[2]; fi <= hi[2]; fi++) {
			if (rnk_l == -1) {
				//втекание по всей левой стенке	
				for (int i = 0; i < K; i++) {
					for (int iM = 0; iM < nMat; iM++) y[iM] = 0.;
					y[0] = 0.6;
					y[1] = 0.4;
					temp[lo[0] - i - 1][j][fi] = temp_in;
					double fU = vMixt / S_Vx;
					double RR = D_Vh / 2.0;
					//double p_g = p0 + pidin[lo[0] + i][j][fi];
					//double p_g = p0 + 0.01;
					double p_g = p0 + 8 * Calc_ML_Nv(y, temp_in) * G_D_MAX[0] * vMixt / (M_PI * RR * RR * RR * RR);
					urs_mixt(y, 0.0, p_g, temp[lo[0] - i - 1][j][fi], 3,
						tmp, gamma, r, h_);

					ro[lo[0] - i - 1][j][fi] = r;

					double ui = 1.0 * fU;
					double vi = 0.0 * fU;
					double wi = 0.0 * fU;

					ru[lo[0] - i - 1][j][fi] = r * ui;
					rv[lo[0] - i - 1][j][fi] = r * vi;
					rw[lo[0] - i - 1][j][fi] = r * wi;
					rh[lo[0] - i - 1][j][fi] = r * h_;
					pidin[lo[0] - i - 1][j][fi] = p_g - p0;

					for (int iM = 0; iM < nMat; iM++)
						ry[iM][lo[0] - i - 1][j][fi] = r * y[iM];
				}
			}
			if (rnk_r == -1) {
				//вытекание в атмосферное давление	 
				for (int i = 0; i < K; i++) {
					ro[hi[0] + i + 1][j][fi] = ro[hi[0] - i][j][fi];
					ru[hi[0] + i + 1][j][fi] = ru[hi[0] - i][j][fi];
					rv[hi[0] + i + 1][j][fi] = rv[hi[0] - i][j][fi];
					rw[hi[0] + i + 1][j][fi] = rw[hi[0] - i][j][fi];
					rh[hi[0] + i + 1][j][fi] = rh[hi[0] - i][j][fi];
					temp[hi[0] + i + 1][j][fi] = temp[hi[0] - i][j][fi];
					pidin[hi[0] + i + 1][j][fi] = 0.0;
					for (int iM = 0; iM < nMat; iM++)
						ry[iM][hi[0] + i + 1][j][fi] = ry[iM][hi[0] - i][j][fi];
				}
			}
		}
	}
	//по радиусу
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			//нижняя граница
			if (rnk_fi0 == -1) {
				for (int fi = 0; fi < K; fi++) { //составляющую давления отражаю
					pidin[i][j][lo[2] - fi - 1] = pidin[i][j][hi[2] - fi];
				}
				//нижняя граница
				for (int fi = 0; fi < K; fi++) {
					ro[i][j][lo[2] - fi - 1] = ro[i][j][hi[2] - fi];
					ru[i][j][lo[2] - fi - 1] = ru[i][j][hi[2] - fi];
					rv[i][j][lo[2] - fi - 1] = rv[i][j][hi[2] - fi];
					rw[i][j][lo[2] - fi - 1] = rw[i][j][hi[2] - fi];
					rh[i][j][lo[2] - fi - 1] = rh[i][j][hi[2] - fi];
					for (int iM = 0; iM < nMat; iM++)
						ry[iM][i][j][lo[2] - fi - 1] = ry[iM][i][j][hi[2] - fi];
				}
				//нижняя стенка температура
				for (int fi = 0; fi < K; fi++) {
					temp[i][j][lo[2] - fi - 1] = temp[i][j][hi[2] - fi];
				}
			}

			//верхняя стенка
			if (rnk_fi == -1) {
				for (int fi = 0; fi < K; fi++) {
					pidin[i][j][hi[2] + fi + 1] = pidin[i][j][lo[2] + fi];
				}
				//верхняя стенка
				for (int fi = 0; fi < K; fi++) {
					ro[i][j][hi[2] + fi + 1] = ro[i][j][lo[2] + fi];
					ru[i][j][hi[2] + fi + 1] = ru[i][j][lo[2] + fi];
					rv[i][j][hi[2] + fi + 1] = rv[i][j][lo[2] + fi];
					rw[i][j][hi[2] + fi + 1] = rw[i][j][lo[2] + fi];
					rh[i][j][hi[2] + fi + 1] = rh[i][j][lo[2] + fi];
					for (int iM = 0; iM < nMat; iM++)
						ry[iM][i][j][hi[2] + fi + 1] = ry[iM][i][j][lo[2] + fi];
				}
				//верхняя стенка температура
				for (int fi = 0; fi < K; fi++) {
					temp[i][j][hi[2] + fi + 1] = temp[i][j][lo[2] + fi];
				}
			}
		}
	}


}

//для тестов
void bnd_cond_p()
{
	double r, p, h_, tmp, gamma;
	double S_Vx = M_PI * (D_Vh / 2.0) * (D_Vh / 2.0);
	int fi_oz;

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int fi = lo[2]; fi <= hi[2]; fi++) {
			double x1 = D_MIN[0] + (i - K) * DX[0];
			if (fi <= NZ / 2) fi_oz = fi + NZ / 2;
			else fi_oz = fi - NZ / 2;
			//нижняя стенка
			for (int j = 0; j < K; j++) { //составляющую давления отражаю
				pidin[i][lo[1] - j - 1][fi] = pidin[i][lo[1]][fi_oz];
			}
			//верхняя стенка
			if (rnk_t == -1) {
				for (int j = 0; j < K; j++) { //составляющую давления везде отражаю
					pidin[i][hi[1] + j + 1][fi] = pidin[i][hi[1] - j][fi];
				}
			}
		}
	}

	// торцевые стенки
	for (int j = lo[1]; j <= hi[1]; j++) {
		for (int fi = lo[2]; fi <= hi[2]; fi++) {
			//втекание по всей левой стенке	
			if (rnk_l == -1) {
				for (int i = 0; i < K; i++) {
					for (int iM = 0; iM < nMat; iM++) y[iM] = 0;
					y[0] = 0.6;
					y[1] = 0.4;

					temp[lo[0] - i - 1][j][fi] = temp_in;
					double fU = vMixt / S_Vx;
					double RR = D_Vh / 2.0;
					//double p_g = p0 + pidin[lo[0] + i][j][fi];
					//double p_g = p0 + 0.01;
					double p_g = p0 + 8 * Calc_ML_Nv(y, temp_in) * G_D_MAX[0] * vMixt / (M_PI * RR * RR * RR * RR);
					urs_mixt(y, 0.0, p_g, temp[lo[0] - i - 1][j][fi], 3,
						tmp, gamma, r, h_);
					pidin[lo[0] - i - 1][j][fi] = p_g - p0;
				}
			}
			//вытекание в атмосферное давление	
			if (rnk_r == -1) {
				for (int i = 0; i < K; i++) {
					pidin[hi[0] + i + 1][j][fi] = 0.0;
				}
			}
		}
	}
	//по радиусу
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			//нижняя граница
			if (rnk_fi0 == -1) {
				for (int fi = 0; fi < K; fi++) { //составляющую давления отражаю
					pidin[i][j][lo[2] - fi - 1] = pidin[i][j][hi[2] - fi];
				}
			}
			//верхняя стенка
			if (rnk_fi == -1) {
				for (int fi = 0; fi < K; fi++) {
					pidin[i][j][hi[2] + fi + 1] = pidin[i][j][lo[2] + fi];
				}
			}
		}
	}
}

void bnd_exch()
{
	exchange_fld(ro);
	exchange_fld(ru);
	exchange_fld(rv);
	exchange_fld(rw);
	exchange_fld(rh);
	exchange_fld(temp);
	exchange_fld(pidin);
	for (int im = 0; im < nMat; im++) {
		exchange_fld(ry[im]);
	}

}

void exchange_fld(double*** fld)
{
	MPI_Status st;
	int cnt, m;
	// left2right
	cnt = K * (hi[1] - lo[1] + 1) * K * (hi[2] - lo[2] + 1);
	if (rnk_r > -1) {
		m = 0;
		for (int i = hi[0] - K + 1; i <= hi[0]; i++) {
			for (int j = lo[1]; j <= hi[1]; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					buf_d[m++] = fld[i][j][k];
				}
			}
		}
		MPI_Send(buf_d, cnt, MPI_DOUBLE, rnk_r, 0, MPI_COMM_WORLD);
	}
	if (rnk_l > -1) {
		MPI_Recv(buf_d, cnt, MPI_DOUBLE, rnk_l, 0, MPI_COMM_WORLD, &st);
		m = 0;
		for (int i = lo[0] - K; i <= lo[0] - 1; i++) {
			for (int j = lo[1]; j <= hi[1]; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					fld[i][j][k] = buf_d[m++];
				}
			}
		}
	}
	// right2left
	cnt = K * (hi[1] - lo[1] + 1) * K * (hi[2] - lo[2] + 1);
	if (rnk_l > -1) {
		m = 0;
		for (int i = lo[0]; i <= lo[0] + K - 1; i++) {
			for (int j = lo[1]; j <= hi[1]; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					buf_d[m++] = fld[i][j][k];
				}
			}
		}
		MPI_Send(buf_d, cnt, MPI_DOUBLE, rnk_l, 1, MPI_COMM_WORLD);
	}
	if (rnk_r > -1) {
		MPI_Recv(buf_d, cnt, MPI_DOUBLE, rnk_r, 1, MPI_COMM_WORLD, &st);
		m = 0;
		for (int i = hi[0] + 1; i <= hi[0] + K; i++) {
			for (int j = lo[1]; j <= hi[1]; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					fld[i][j][k] = buf_d[m++];
				}
			}
		}
	}

	// bottom2top
	cnt = K * (hi[0] - lo[0] + 1) * K * (hi[2] - lo[2] + 1);
	if (rnk_t > -1) {
		m = 0;
		for (int i = lo[0]; i <= hi[0]; i++) {
			for (int j = hi[1] - K + 1; j <= hi[1]; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					buf_d[m++] = fld[i][j][k];
				}
			}
		}
		MPI_Send(buf_d, cnt, MPI_DOUBLE, rnk_t, 2, MPI_COMM_WORLD);
	}
	if (rnk_b > -1) {
		MPI_Recv(buf_d, cnt, MPI_DOUBLE, rnk_b, 2, MPI_COMM_WORLD, &st);
		m = 0;
		for (int i = lo[0]; i <= hi[0]; i++) {
			for (int j = lo[1] - K; j <= lo[1] - 1; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					fld[i][j][k] = buf_d[m++];
				}
			}
		}
	}

	// top2bottom
	cnt = K * (hi[0] - lo[0] + 1) * K * (hi[2] - lo[2] + 1);
	if (rnk_b > -1) {
		m = 0;
		for (int i = lo[0]; i <= hi[0]; i++) {
			for (int j = lo[1]; j <= lo[1] + K - 1; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					buf_d[m++] = fld[i][j][k];
				}
			}
		}
		MPI_Send(buf_d, cnt, MPI_DOUBLE, rnk_b, 3, MPI_COMM_WORLD);
	}
	if (rnk_t > -1) {
		MPI_Recv(buf_d, cnt, MPI_DOUBLE, rnk_t, 3, MPI_COMM_WORLD, &st);
		m = 0;
		for (int i = lo[0]; i <= hi[0]; i++) {
			for (int j = hi[1] + 1; j <= hi[1] + K; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					fld[i][j][k] = buf_d[m++];
				}
			}
		}
	}
	/*
		// front2back (от меньшего угла к большему)
		cnt = K * (hi[0] - lo[0] + 1) * K * (hi[1] - lo[1] + 1);
		if (rnk_fi0 > -1) {
			m = 0;
			for (int i = lo[0]; i <= hi[0]; i++) {
				for (int j = lo[1]; j <= hi[1]; j++) {
					for (int k = hi[2] - K + 1; k <= hi[2]; k++) {
						buf_d[m++] = fld[i][j][k];
					}
				}
			}
			MPI_Send(buf_d, cnt, MPI_DOUBLE, rnk_t, 4, MPI_COMM_WORLD);
		}
		if (rnk_b > -1) {
			MPI_Recv(buf_d, cnt, MPI_DOUBLE, rnk_b, 4, MPI_COMM_WORLD, &st);
			m = 0;
			for (int i = lo[0]; i <= hi[0]; i++) {
				for (int j = lo[1]; j <= hi[1]; j++) {
					for (int k = lo[2] - K; k <= lo[2] - 1; k++) {
						fld[i][j][k] = buf_d[m++];
					}
				}
			}
		}

		// back2front
		cnt = K * (hi[0] - lo[0] + 1) * K * (hi[1] - lo[1] + 1);
		if (rnk_b > -1) {
			m = 0;
			for (int i = lo[0]; i <= hi[0]; i++) {
				for (int j = lo[1]; j <= hi[1]; j++) {
					for (int k = lo[2]; k <= lo[2] + K - 1; k++) {
						buf_d[m++] = fld[i][j][k];
					}
				}
			}
			MPI_Send(buf_d, cnt, MPI_DOUBLE, rnk_b, 5, MPI_COMM_WORLD);
		}
		if (rnk_t > -1) {
			MPI_Recv(buf_d, cnt, MPI_DOUBLE, rnk_t, 5, MPI_COMM_WORLD, &st);
			m = 0;
			for (int i = lo[0]; i <= hi[0]; i++) {
				for (int j = lo[1]; j <= hi[1]; j++) {
					for (int k = hi[2] + 1; k <= hi[2] + K; k++) {
						fld[i][j][k] = buf_d[m++];
					}
				}
			}
		}*/

}


void calc_fluxes()
{
	double gammar, tempr, rr, ur, vr, wr, hr, gammal, templ, rl, ul, vl, wl, hl, r_, h_, tmp, pir, pil, rl_mix, t_ul, t_ur, t_vl, t_vr, t_wr, t_wl, tau_ul, tau_ur, tau_vr, tau_vl, tau_wl, tau_wr;
	double tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz, QF;
	for (int i = lo[0]; i <= hi[0] + 1; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				for (int iM = 0; iM < nMat; iM++) yl[iM] = ry[iM][i - 1][j][fi] / ro[i - 1][j][fi];
				templ = temp[i - 1][j][fi];
				pil = pidin[i - 1][j][fi];
				rl = ro[i - 1][j][fi];
				ul = ru[i - 1][j][fi] / ro[i - 1][j][fi];
				vl = rv[i - 1][j][fi] / ro[i - 1][j][fi];
				wl = rw[i - 1][j][fi] / ro[i - 1][j][fi];
				hl = rh[i - 1][j][fi] / ro[i - 1][j][fi];



				for (int iM = 0; iM < nMat; iM++) yr[iM] = ry[iM][i][j][fi] / ro[i][j][fi];
				tempr = temp[i][j][fi];
				pir = pidin[i][j][fi];
				rr = ro[i][j][fi];
				ur = ru[i][j][fi] / ro[i][j][fi];
				vr = rv[i][j][fi] / ro[i][j][fi];
				wr = rw[i][j][fi] / ro[i][j][fi];
				hr = rh[i][j][fi] / ro[i][j][fi];

				//LAX-Friedrichs flux
				double alpha = _MAX_(sqrt(ul * ul + vl * vl + wl * wl), sqrt(ur * ur + vr * vr + wr * wr));

				fluxx_ru[i][j][fi] = 0.5 * ((rr * ur * ur + rl * ul * ul) - alpha * (rr * ur - rl * ul));
				fluxx_rv[i][j][fi] = 0.5 * ((rr * ur * vr + rl * ul * vl) - alpha * (rr * vr - rl * vl));
				fluxx_rw[i][j][fi] = 0.5 * ((rr * ur * wr + rl * ul * wl) - alpha * (rr * wr - rl * wl));
				fluxx_rh[i][j][fi] = 0.5 * ((rr * ur * hr + rl * ul * hl) - alpha * (rr * hr - rl * hl));
				for (int iM = 0; iM < nMat; iM++) {
					fluxx_ry[iM][i][j][fi] = 0.5 * ((rr * ur * yr[iM] + rl * ul * yl[iM]) - alpha * (rr * yr[iM] - rl * yl[iM]));
				}

				// расчёт вязкости
				// тут мы считаем r слева и справа)))))
		/*		double r_yl = D_MIN[1] + (j - lo[1]) * DX[1];
				double r_yr = D_MIN[1] + (j - lo[1] + 1) * DX[1];
				double r_y = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];

				//считаем узлы разностной схемы для скоростей по z
				t_ul = ru[i - 1][j][fi] / ro[i - 1][j][fi], t_ur = ru[i][j][fi] / ro[i][j][fi], t_vl = rv[i - 1][j][fi] / ro[i - 1][j][fi], t_vr = rv[i][j][fi] / ro[i][j][fi];
				t_wl = rw[i - 1][j][fi] / ro[i - 1][j][fi];
				t_wr = rw[i][j][fi] / ro[i][j][fi];

				double ul_r = 0.25*(ru[i-1][j][fi]/ro[i-1][j][fi]+ru[i][j][fi]/ro[i][j][fi]+ru[i-1][j-1][fi]/ro[i-1][j-1][fi]+ru[i][j-1][fi]/ro[i][j-1][fi]);
				double ur_r = 0.25*(ru[i-1][j][fi]/ro[i-1][j][fi]+ru[i][j][fi]/ro[i][j][fi]+ru[i-1][j+1][fi]/ro[i-1][j+1][fi]+ru[i][j+1][fi]/ro[i][j+1][fi]);
				double wl_fi = 0.25*(rw[i-1][j][fi]/ro[i-1][j][fi]+rw[i][j][fi]/ro[i][j][fi]+rw[i-1][j][fi-1]/ro[i-1][j][fi-1]+rw[i][j][fi-1]/ro[i][j][fi-1]);
				double wr_fi = 0.25*(rw[i-1][j][fi]/ro[i-1][j][fi]+rw[i][j][fi]/ro[i][j][fi]+rw[i-1][j][fi+1]/ro[i-1][j][fi+1]+rw[i][j][fi+1]/ro[i][j][fi+1]);
				double vl_r = 0.25*(rv[i-1][j][fi]/ro[i-1][j][fi]+rv[i][j][fi]/ro[i][j][fi]+rv[i-1][j-1][fi]/ro[i-1][j-1][fi]+rv[i][j-1][fi]/ro[i][j-1][fi]);
				double vr_r = 0.25*(rv[i-1][j][fi]/ro[i-1][j][fi]+rv[i][j][fi]/ro[i][j][fi]+rv[i-1][j+1][fi]/ro[i-1][j+1][fi]+rv[i][j+1][fi]/ro[i][j+1][fi]);
				double ul_fi = 0.25*(ru[i-1][j][fi]/ro[i-1][j][fi]+ru[i][j][fi]/ro[i][j][fi]+ru[i-1][j][fi-1]/ro[i-1][j][fi-1]+ru[i][j][fi-1]/ro[i][j][fi-1]);
				double ur_fi = 0.25*(ru[i-1][j][fi]/ro[i-1][j][fi]+ru[i][j][fi]/ro[i][j][fi]+ru[i-1][j][fi+1]/ro[i-1][j][fi+1]+ru[i][j][fi+1]/ro[i][j][fi+1]);


				tau_xx = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * (2 * (t_ur - t_ul) / DX[0] - 2 * ((t_ur - t_ul) / DX[0] + (vr_r * r_yr  - vl_r * r_yl) / (DX[1] * r_y) + (wr_fi - wl_fi) / (DX[2] * r_y)) / 3);
				tau_xy = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * ((ur_r - ul_r) / DX[1] + (t_vr - t_vl) / DX[0]);
				tau_xz = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * ((t_wr - t_wl) / DX[0] + (ur_fi - ul_fi) / (DX[2] * r_y));

				fluxx_ru[i][j][fi] -= tau_xx;
				fluxx_rv[i][j][fi] -= tau_xy;
				fluxx_rw[i][j][fi] -= tau_xz;

				//Поток тепла
				QF = 0.5 * (Calc_KP_Nv(yr, tempr) + Calc_KP_Nv(yl, templ)) * (tempr - templ) / DX[0];
				fluxx_rh[i][j][fi] -= QF;*/

			}
		}
	}
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1] + 1; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				for (int iM = 0; iM < nMat; iM++) yl[iM] = ry[iM][i][j - 1][fi] / ro[i][j - 1][fi];
				templ = temp[i][j - 1][fi];
				pil = pidin[i][j - 1][fi];
				rl = ro[i][j - 1][fi];
				ul = ru[i][j - 1][fi] / ro[i][j - 1][fi];
				vl = rv[i][j - 1][fi] / ro[i][j - 1][fi];
				wl = rw[i][j - 1][fi] / ro[i][j - 1][fi];
				hl = rh[i][j - 1][fi] / ro[i][j - 1][fi];

				for (int iM = 0; iM < nMat; iM++) yr[iM] = ry[iM][i][j][fi] / ro[i][j][fi];
				tempr = temp[i][j][fi];
				pir = pidin[i][j][fi];
				rr = ro[i][j][fi];
				ur = ru[i][j][fi] / ro[i][j][fi];
				vr = rv[i][j][fi] / ro[i][j][fi];
				wr = rw[i][j][fi] / ro[i][j][fi];
				hr = rh[i][j][fi] / ro[i][j][fi];

				double r_y = D_MIN[1] + (j - lo[1]) * DX[1];
				double r_yl = D_MIN[1] + (j - lo[1] - 0.5) * DX[1];
				double r_yr = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];

				//LAX-Friedrichs flux 

				double alpha = _MAX_(sqrt(ul * ul + vl * vl + wl * wl), sqrt(ur * ur + vr * vr + wr * wr));

				fluxy_ru[i][j][fi] = 0.5 * r_y * ((rr * vr * ur + rl * vl * ul) - alpha * (rr * ur - rl * ul));
				fluxy_rv[i][j][fi] = 0.5 * r_y * ((rr * vr * vr + rl * vl * vl) - alpha * (rr * vr - rl * vl));
				fluxy_rw[i][j][fi] = 0.5 * r_y * ((rr * vr * wr + rl * vl * wl) - alpha * (rr * wr - rl * wl));
				fluxy_rh[i][j][fi] = 0.5 * r_y * ((rr * vr * hr + rl * vl * hl) - alpha * (rr * hr - rl * hl));
				for (int iM = 0; iM < nMat; iM++) {
					fluxy_ry[iM][i][j][fi] = 0.5 * r_y * ((rr * vr * yr[iM] + rl * vl * yl[iM]) - alpha * (rr * yr[iM] - rl * yl[iM]));
				}

				//вязкость
				//считаем узлы разностной схемы для скоростей по r
			/*	t_ul = ru[i][j - 1][fi] / ro[i][j - 1][fi], t_ur = ru[i][j][fi] / ro[i][j][fi], t_vl = rv[i][j - 1][fi] / ro[i][j - 1][fi], t_vr = rv[i][j][fi] / ro[i][j][fi];
				t_wl = rw[i][j - 1][fi] / ro[i][j - 1][fi];
				t_wr = rw[i][j][fi] / ro[i][j][fi];

				double ul_z = 0.25*(ru[i][j-1][fi]/ro[i][j-1][fi]+ru[i][j][fi]/ro[i][j][fi]+ru[i-1][j-1][fi]/ro[i-1][j-1][fi]+ru[i-1][j][fi]/ro[i-1][j][fi]);
				double ur_z = 0.25*(ru[i][j-1][fi]/ro[i][j-1][fi]+ru[i][j][fi]/ro[i][j][fi]+ru[i+1][j-1][fi]/ro[i+1][j-1][fi]+ru[i+1][j][fi]/ro[i+1][j][fi]);

				double vl_z = 0.25*(rv[i][j-1][fi]/ro[i][j-1][fi]+rv[i][j][fi]/ro[i][j][fi]+rv[i-1][j-1][fi]/ro[i-1][j-1][fi]+rv[i-1][j][fi]/ro[i-1][j][fi]);
				double vr_z = 0.25*(rv[i][j-1][fi]/ro[i][j-1][fi]+rv[i][j][fi]/ro[i][j][fi]+rv[i+1][j-1][fi]/ro[i+1][j-1][fi]+rv[i+1][j][fi]/ro[i+1][j][fi]);

				double wl_fi = 0.25*(rw[i][j-1][fi]/ro[i][j-1][fi]+rw[i][j][fi]/ro[i][j][fi]+rw[i][j-1][fi-1]/ro[i][j-1][fi-1]+rw[i][j][fi-1]/ro[i][j][fi-1]);
				double wr_fi = 0.25*(rw[i][j-1][fi]/ro[i][j-1][fi]+rw[i][j][fi]/ro[i][j][fi]+rw[i][j-1][fi+1]/ro[i][j-1][fi+1]+rw[i][j][fi+1]/ro[i][j][fi+1]);

				double vl_fi = 0.25*(rv[i][j-1][fi]/ro[i][j-1][fi]+rv[i][j][fi]/ro[i][j][fi]+rv[i][j-1][fi-1]/ro[i][j-1][fi-1]+rv[i][j][fi-1]/ro[i][j][fi-1]);
				double vr_fi = 0.25*(rv[i][j-1][fi]/ro[i][j-1][fi]+rv[i][j][fi]/ro[i][j][fi]+rv[i][j-1][fi+1]/ro[i][j-1][fi+1]+rv[i][j][fi+1]/ro[i][j][fi+1]);


				if (j != lo[1]) {

					tau_xy = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * ((t_ur - t_ul) / DX[1] + (vr_z - vl_z) / DX[0]);
					tau_yy = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * (2.0 * (t_vr - t_vl) / DX[1] - 2 * ((ur_z - ul_z) / DX[0] + (r_yr * t_vr - r_yl * t_vl) / (r_y * DX[1]) + (wr_fi - wl_fi) / (DX[2] * r_y)) / 3);
					tau_yz = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * ((t_wr - t_wl) / DX[1] + (vr_fi - vl_fi) / (r_y * DX[2]) - 0.5 * (t_wr + t_wl) / r_y);

					fluxy_ru[i][j][fi] -= r_y * tau_xy;
					fluxy_rv[i][j][fi] -= r_y * tau_yy;
					fluxy_rw[i][j][fi] -= r_y * tau_yz;
				}

				QF = 0.5 * r_y * (Calc_KP_Nv(yr, tempr) + Calc_KP_Nv(yl, templ)) * (tempr - templ) / DX[1];
				fluxy_rh[i][j][fi] -= QF;*/

			}
		}
	}
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2] + 1; fi++) {
				for (int iM = 0; iM < nMat; iM++) yl[iM] = ry[iM][i][j][fi - 1] / ro[i][j][fi - 1];
				templ = temp[i][j][fi - 1];
				pil = pidin[i][j][fi - 1];
				rl = ro[i][j][fi - 1];
				ul = ru[i][j][fi - 1] / ro[i][j][fi - 1];
				vl = rv[i][j][fi - 1] / ro[i][j][fi - 1];
				wl = rw[i][j][fi - 1] / ro[i][j][fi - 1];
				hl = rh[i][j][fi - 1] / ro[i][j][fi - 1];

				for (int iM = 0; iM < nMat; iM++) yr[iM] = ry[iM][i][j][fi] / ro[i][j][fi];
				tempr = temp[i][j][fi];
				pir = pidin[i][j][fi];
				rr = ro[i][j][fi];
				ur = ru[i][j][fi] / ro[i][j][fi];
				vr = rv[i][j][fi] / ro[i][j][fi];
				wr = rw[i][j][fi] / ro[i][j][fi];
				hr = rh[i][j][fi] / ro[i][j][fi];

				//LAX-Friedrichs flux 
				double alpha = _MAX_(sqrt(ul * ul + vl * vl + wl * wl), sqrt(ur * ur + vr * vr + wr * wr));

				fluxz_ru[i][j][fi] = 0.5 * ((rr * wr * ur + rl * wl * ul) - alpha * (rr * ur - rl * ul));
				fluxz_rv[i][j][fi] = 0.5 * ((rr * wr * vr + rl * wl * vl) - alpha * (rr * vr - rl * vl));
				fluxz_rw[i][j][fi] = 0.5 * ((rr * wr * wr + rl * wl * wl) - alpha * (rr * wr - rl * wl));
				fluxz_rh[i][j][fi] = 0.5 * ((rr * wr * hr + rl * wl * hl) - alpha * (rr * hr - rl * hl));
				for (int iM = 0; iM < nMat; iM++) {
					fluxz_ry[iM][i][j][fi] = 0.5 * ((rr * wr * yr[iM] + rl * wl * yl[iM]) - alpha * (rr * yr[iM] - rl * yl[iM]));
				}

				double r_y = D_MIN[1] + (j - lo[1]) * DX[1];
				double r_yl = D_MIN[1] + (j - lo[1] - 0.5) * DX[1];
				double r_yr = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];

				//вязкость


			/*	//считаем узлы разностной схемы для скоростей по fi
				t_ul = ru[i][j][fi - 1] / ro[i][j][fi - 1], t_ur = ru[i][j][fi] / ro[i][j][fi], t_vl = rv[i][j][fi - 1] / ro[i][j][fi - 1], t_vr = rv[i][j][fi] / ro[i][j][fi];
				t_wl = rw[i][j][fi - 1] / ro[i][j][fi - 1], t_wr = rw[i][j][fi] / ro[i][j][fi];

				double wl_z = 0.;//0.25*(rw[i][j][fi-1]/ro[i][j][fi-1]+rw[i][j][fi]/ro[i][j][fi]+rw[i-1][j][fi-1]/ro[i-1][j][fi-1]+rw[i-1][j][fi]/ro[i-1][j][fi]);
				double wr_z = 0.;//0.25*(rw[i][j][fi-1]/ro[i][j][fi-1]+rw[i][j][fi]/ro[i][j][fi]+rw[i+1][j][fi-1]/ro[i+1][j][fi-1]+rw[i+1][j][fi]/ro[i+1][j][fi]);

				double wl_r = 0.;//0.25*(rw[i][j][fi-1]/ro[i][j][fi-1]+rw[i][j][fi]/ro[i][j][fi]+rw[i][j-1][fi-1]/ro[i][j-1][fi-1]+rw[i][j-1][fi]/ro[i][j-1][fi]);
				double wr_r = 0.25*(rw[i][j][fi-1]/ro[i][j][fi-1]+rw[i][j][fi]/ro[i][j][fi]+rw[i][j+1][fi-1]/ro[i][j+1][fi-1]+rw[i][j+1][fi]/ro[i][j+1][fi]);

				double ul_z = 0.;//0.25*(ru[i][j][fi-1]/ro[i][j][fi-1]+ru[i][j][fi]/ro[i][j][fi]+ru[i-1][j][fi-1]/ro[i-1][j][fi-1]+ru[i-1][j][fi]/ro[i-1][j][fi]);
				double ur_z = 0.;//0.25*(ru[i][j][fi-1]/ro[i][j][fi-1]+ru[i][j][fi]/ro[i][j][fi]+ru[i+1][j][fi-1]/ro[i+1][j][fi-1]+ru[i+1][j][fi]/ro[i+1][j][fi]);

				double vl_r = 0.;//0.25*(rv[i][j][fi-1]/ro[i][j][fi-1]+rv[i][j][fi]/ro[i][j][fi]+rv[i][j-1][fi-1]/ro[i][j-1][fi-1]+rv[i][j-1][fi]/ro[i][j-1][fi]);
				double vr_r = 0.;//0.25*(rv[i][j][fi-1]/ro[i][j][fi-1]+rv[i][j][fi]/ro[i][j][fi]+rv[i][j+1][fi-1]/ro[i][j+1][fi-1]+rv[i][j+1][fi]/ro[i][j+1][fi]);


				if (j != lo[1]) {
					tau_xz = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * ((wr_z - wl_z) / DX[0] + (t_ur - t_ul) / (r_y * DX[2]));
					tau_yz = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * ((wr_r - wl_r) / DX[1] + (t_vr - t_vl) / (r_y * DX[2]) - 0.5 * (t_wr + t_wl) / r_y);
					tau_zz = 0.5 * (Calc_ML_Nv(yl, templ) + Calc_ML_Nv(yr, tempr)) * 2.0 * ((t_wr - t_wl) / (r_y * DX[2]) + 0.5 * (t_vr + t_vl) / r_y - ((ur_z - ul_z) / DX[0] + (r_yr * vr_r - r_yl * vl_r) / (DX[1] * r_y) + (t_wr - t_wl) / (DX[2] * r_y)) / 3.0);

					fluxz_ru[i][j][fi] -= tau_xz;
					fluxz_rv[i][j][fi] -= tau_yz;
					fluxz_rw[i][j][fi] -= tau_zz;
				}

				//Поток тепла
				QF = 0.5 * (Calc_KP_Nv(yr, tempr) + Calc_KP_Nv(yl, templ)) * (tempr - templ) / (r_yr * DX[2]);
				fluxz_rh[i][j][fi] -= QF;*/
			}
		}
	}

}

double Calc_KP_Nv(double* y, double tmp)
{
	double fKP = 0.0;
	double* KPm = new double[nMat];
	double fM_ = 0.0;
	double KP = 0.0;

	KPm[0] = (0.0) * tmp * tmp + (0.0002) * tmp + (-0.0272);
	KPm[1] = (1.07e-7) * tmp * tmp + (5.41e-5) * tmp + (-4.72e-3);

	for (int iM = 0; iM < nMat; iM++)
	{
		double fCK_M = y[iM] / Mm[iM];

		fM_ += fCK_M;
		//if (iM < OsnV) {
		fKP += fCK_M * KPm[iM];
		//}
		//else {
		//	//fKP += fCK_M * KP0[iM];
		//}
	}

	KP = fKP / fM_;

	//KP=KPm[0];
	delete[] KPm;

	return KP;
}
void calc_new_fields()
{
	double tmp, gamma_, r_, h_;
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				double r_y = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];
				for (int iM = 0; iM < nMat; iM++) y[iM] = ry[iM][i][j][fi] / ro[i][j][fi];
				rh[i][j][fi] -= ((fluxx_rh[i + 1][j][fi] - fluxx_rh[i][j][fi]) / DX[0] + (fluxy_rh[i][j + 1][fi] - fluxy_rh[i][j][fi]) / (DX[1] * r_y) + (fluxz_rh[i][j][fi + 1] - fluxz_rh[i][j][fi]) / (DX[2] * r_y)) * TAU;
				for (int iM = 0; iM < nMat; iM++) {
					ry[iM][i][j][fi] -= ((fluxx_ry[iM][i + 1][j][fi] - fluxx_ry[iM][i][j][fi]) / DX[0] + (fluxy_ry[iM][i][j + 1][fi] - fluxy_ry[iM][i][j][fi]) / (DX[1] * r_y) + (fluxz_ry[iM][i][j][fi + 1] - fluxz_ry[iM][i][j][fi]) / (DX[2] * r_y)) * TAU;
				}
			}
		}
	}
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				double r_y = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];
				double r_yl = D_MIN[1] + (j - lo[1]) * DX[1];
				double r_yr = D_MIN[1] + (j - lo[1] + 1) * DX[1];
				double rr = ro[i][j][fi];
				double ur = rv[i][j][fi] / rr;
				double ufi = rw[i][j][fi] / rr;
				double wr_fi = rw[i][j][fi + 1] / ro[i][j][fi + 1];
				double wl_fi = rw[i][j][fi - 1] / ro[i][j][fi - 1];
				double vv = rv[i][j][fi] / ro[i][j][fi];
				double ur_z = ru[i + 1][j][fi] / ro[i + 1][j][fi];
				double ul_z = ru[i - 1][j][fi] / ro[i - 1][j][fi];
				double vr_r = rv[i][j + 1][fi] / ro[i][j + 1][fi];
				double vl_r = rv[i][j - 1][fi] / ro[i][j - 1][fi];
				double wr_r = rw[i][j + 1][fi] / ro[i][j + 1][fi];
				double wl_r = rw[i][j - 1][fi] / ro[i][j - 1][fi];
				double vr_fi = rv[i][j][fi + 1] / ro[i][j][fi + 1];
				double vl_fi = rv[i][j][fi - 1] / ro[i][j][fi - 1];
				for (int iM = 0; iM < nMat; iM++) y[iM] = ry[iM][i][j][fi] / ro[i][j][fi];

				double tau_fifi = 0.;//Calc_ML_Nv(y, temp[i][j][fi]) * 2.0 * ((wr_fi - wl_fi) / (2*r_y * DX[2]) + vv/ r_y - ((ur_z - ul_z) / (2*DX[0]) + (r_yr * vr_r - r_yl * vl_r) / (2*DX[1] * r_y) + (wr_fi - wl_fi) / (2*DX[2] * r_y)) / 3.0);
				double tau_rfi = 0.;//Calc_ML_Nv(y, temp[i][j][fi])*((wr_r - wl_r) / (2*DX[1])+(vr_fi - vl_fi) / (2*r_y * DX[2])-ufi/r_y);

				ru[i][j][fi] -= ((fluxx_ru[i + 1][j][fi] - fluxx_ru[i][j][fi]) / DX[0] + (fluxy_ru[i][j + 1][fi] - fluxy_ru[i][j][fi]) / (DX[1] * r_y) + (fluxz_ru[i][j][fi + 1] - fluxz_ru[i][j][fi]) / (DX[2] * r_y)) * TAU;
				rv[i][j][fi] -= ((fluxx_rv[i + 1][j][fi] - fluxx_rv[i][j][fi]) / DX[0] + (fluxy_rv[i][j + 1][fi] - fluxy_rv[i][j][fi]) / (DX[1] * r_y) + (fluxz_rv[i][j][fi + 1] - fluxz_rv[i][j][fi]) / (DX[2] * r_y)) * TAU;
				rv[i][j][fi] += TAU * (rr * ufi * ufi - tau_fifi) / r_y;
				rw[i][j][fi] -= ((fluxx_rw[i + 1][j][fi] - fluxx_rw[i][j][fi]) / DX[0] + (fluxy_rw[i][j + 1][fi] - fluxy_rw[i][j][fi]) / (DX[1] * r_y) + (fluxz_rw[i][j][fi + 1] - fluxz_rw[i][j][fi]) / (DX[2] * r_y)) * TAU;
				rw[i][j][fi] -= TAU * (rr * ur * ufi - tau_rfi) / r_y;
			}
		}
	}

	//u*
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				u[i][j][fi] = ru[i][j][fi] / ro[i][j][fi];
				v[i][j][fi] = rv[i][j][fi] / ro[i][j][fi];//printf("!!!");	
				w[i][j][fi] = rw[i][j][fi] / ro[i][j][fi];
			}
		}
	}

	for (int i = lo[0] - 1; i <= hi[0] + 1; i++) {
		for (int j = lo[1] - 1; j <= hi[1] + 1; j++) {
			for (int fi = lo[2] - 1; fi <= hi[2] + 1; fi++) {
				ro_1[i][j][fi] = ro[i][j][fi];
			}
		}
	}
	//обновила поле плотности
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				ro[i][j][fi] = 0.0;
				for (int iM = 0; iM < nMat; iM++) {
					ro[i][j][fi] += ry[iM][i][j][fi];
					ry_1[iM][i][j][fi] = ry[iM][i][j][fi];
				}
			}
		}
	}

	//обновила поле температуры
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				for (int iM = 0; iM < nMat; iM++) y[iM] = ry[iM][i][j][fi] / ro[i][j][fi];
				urs_mixt(y, rh[i][j][fi] / ro[i][j][fi], 0.0, temp[i][j][fi], 1,
					tmp, gamma_, r_, h_);
				temp[i][j][fi] = tmp;
				temp_1[i][j][fi] = temp[i][j][fi];
			}
		}
	}
}

void calc_new_pi()
{
	calc_s();
	calc_pi();
	bnd_cond_p();
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				dpx[i][j][fi] = pidin[i + 1][j][fi] - pidin[i - 1][j][fi];
			}
		}
	}

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				dpy[i][j][fi] = pidin[i][j + 1][fi] - pidin[i][j - 1][fi];
			}
		}
	}

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				dpz[i][j][fi] = pidin[i][j][fi + 1] - pidin[i][j][fi - 1];
			}
		}
	}

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int fi = lo[2]; fi <= hi[2]; fi++) {
				double r_y = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];
				u[i][j][fi] -= (TAU / ro_1[i][j][fi]) * (dpx[i][j][fi]) / (2.0 * DX[0]);
				ru[i][j][fi] = ro[i][j][fi] * u[i][j][fi];
				v[i][j][fi] -= (TAU / ro_1[i][j][fi]) * (dpy[i][j][fi]) / (2.0 * DX[1]);
				rv[i][j][fi] = ro[i][j][fi] * v[i][j][fi];
				w[i][j][fi] -= (TAU / ro_1[i][j][fi]) * (dpz[i][j][fi]) / (2.0 * DX[2] * r_y);
				rw[i][j][fi] = ro[i][j][fi] * w[i][j][fi];
			}
		}
	}
}


void calc_one_time_step()
{
	bnd_exch();
	bnd_cond();
	calc_fluxes();
	calc_new_fields();
	bnd_exch();
	bnd_cond();
	calc_new_pi();
}


double Calc_CP(int iM, double fT)
{
	double fTP = fT, fCP = k_TCP[iM][0];
	for (int i = 1; i < n_TCP; i++) {
		fCP += fTP * k_TCP[iM][i];
		fTP *= fT;
	}
	return fCP;
}
double Calc_CP_(int iM, double fT)
{
	int n_TCP_ = n_TCP - 1;
	double* k_TCP_ = new double[n_TCP_];
	for (int i = 0; i < n_TCP_; i++) k_TCP_[i] = (i + 1) * k_TCP[iM][i + 1];

	double fTP = fT, fCP = k_TCP_[0];
	for (int i = 1; i < n_TCP_; i++) {
		fCP += fTP * k_TCP_[i];
		fTP *= fT;
	}
	delete[] k_TCP_;
	return fCP;

}


//уравнение состояния для многокомпонентного идеального газа
void urs_mixt(double* y, double h, double p, double fT, int flag,
	double& tmp, double& gamma, double& r, double& h_)
{
	double Cv, Cp, e;
	double fM = 0.0; // fM = SUM( Yi / Mi )
	tmp = fT;
	double tmp1 = fT;
	for (int iM = 0; iM < nMat; iM++) fM += y[iM] / Mm[iM];

	// вычисляем Т итерациями по Ньютону
	if (flag == 1) {
		Newton_Method(h, y, tmp1,
			tmp);
	}

	double fCP = 0.0;
	for (int iM = 0; iM < nMat; iM++)  fCP += y[iM] * Calc_CP(iM, tmp);

	Cp = fCP;
	Cv = Cp - gR * fM;
	gamma = Cp / Cv;

	switch (flag)
	{
	case 2:
		calc_h(tmp,
			hii);
		h_ = 0.0;
		for (int iM = 0; iM < nMat; iM++) h_ += y[iM] * hii[iM];
		e = h_ - gR * tmp * fM;
		r = p0 / (gR * tmp * fM);
		break;
	case 3:
		calc_h(tmp,
			hii);
		h_ = 0.0;
		for (int iM = 0; iM < nMat; iM++) h_ += y[iM] * hii[iM];
		e = h_ - gR * tmp * fM;
		r = p / (gR * tmp * fM);
		break;

	}

}


void Newton_Method(double e, double* y, double tmp1,
	double& tmp)
{
	double fT = tmp1;		// начальное приближение для температуры
	double fE = e;		// энтальпия

	double fM = 0.0; // fM = SUM( Yi / Mi )
	for (int i = 0; i < nMat; i++) fM += y[i] / Mm[i];
	double fR_M = gR * fM;
	fR_M = 0.0;

	for (int ic = 0; ic < 100; ic++)
	{
		double fCP = 0.0;
		double fCP_ = 0.0;
		calc_h(fT,
			hir);
		for (int i = 0; i < nMat; i++)
		{
			fCP += y[i] * hir[i];
			fCP_ += y[i] * Calc_CP(i, fT);
		}

		double fFT = fCP - fE;
		double fFT_ = fCP_;

		double fTg = fT - fFT / fFT_;
		if ((fTg - fT) * (fTg - fT) < 1.0e-10) { fT = fTg; break; }
		fT = fTg;
	}
	tmp = fT;
}


void calc_s()
{

	double templ, tempr, rr, rl, Cp, pil, pir;
	// Расчёт по z
	/*for (int i = lo[0]; i <= hi[0] + 1; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int k = lo[2]; k <= hi[2]; k++) {
				for (int iM = 0; iM < nMat; iM++) yl[iM] = ry[iM][i - 1][j][k] / ro[i - 1][j][k];
				templ = temp[i - 1][j][k];
				rl = ro[i - 1][j][k];
				pil = pidin[i - 1][j][k];

				for (int iM = 0; iM < nMat; iM++) yr[iM] = ry[iM][i][j][k] / ro[i][j][k];

				tempr = temp[i][j][k];
				rr = ro[i][j][k];
				pir = pidin[i][j][k];

				QFx[i][j][k] = 0.5 * (Calc_KP_Nv(yr, tempr) + Calc_KP_Nv(yl, templ)) * (tempr - templ) / DX[0];

			}
		}
	}

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1] + 1; j++) {
			for (int k = lo[2]; k <= hi[2]; k++) {
				for (int iM = 0; iM < nMat; iM++) yl[iM] = ry[iM][i][j - 1][k] / ro[i][j - 1][k];
				templ = temp[i][j - 1][k];
				rl = ro[i][j - 1][k];
				pil = pidin[i][j - 1][k];

				for (int iM = 0; iM < nMat; iM++) yr[iM] = ry[iM][i][j][k] / ro[i][j][k];

				tempr = temp[i][j][k];
				rr = ro[i][j][k];
				pir = pidin[i][j][k];


				double r_y = D_MIN[1] + (j - lo[1]) * DX[1];
				double r_yl = D_MIN[1] + (j - lo[1] - 0.5) * DX[1];
				double r_yr = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];

				QFy[i][j][k] = 0.5 * r_y * (Calc_KP_Nv(yr, tempr) + Calc_KP_Nv(yl, templ)) * (tempr - templ) / DX[1];


			}
		}
	}

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int k = lo[2]; k <= hi[2] + 1; k++) {
				for (int iM = 0; iM < nMat; iM++) yl[iM] = ry[iM][i][j][k - 1] / ro[i][j][k - 1];
				templ = temp[i][j][k - 1];
				rl = ro[i][j][k - 1];
				pil = pidin[i][j][k - 1];

				for (int iM = 0; iM < nMat; iM++) yr[iM] = ry[iM][i][j][k] / ro[i][j][k];

				tempr = temp[i][j][k];
				rr = ro[i][j][k];
				pir = pidin[i][j][k];


				double r_y = D_MIN[1] + (j - lo[1]) * DX[1];
				double r_yl = D_MIN[1] + (j - lo[1] - 0.5) * DX[1];
				double r_yr = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];

				QFz[i][j][k] = 0.5 * (Calc_KP_Nv(yr, tempr) + Calc_KP_Nv(yl, templ)) * (tempr - templ) / (r_yr * DX[2]);

			}
		}
	}*/
	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int k = lo[2]; k <= hi[2]; k++) {
				S[i][j][k] = 0.0;
			}
		}
	}

	/*for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int k = lo[2]; k <= hi[2]; k++) {
				for (int iM = 0; iM < nMat; iM++) y[iM] = ry[iM][i][j][k] / ro[i][j][k];
				double fM = 0.0; //средняя молекулярная масса смеси
				for (int iM = 0; iM < nMat; iM++) fM += y[iM] / Mm[iM];
				fM = 1.0 / fM;
				Cp = 0.0; //теплоемкость смеси
				for (int iM = 0; iM < nMat; iM++)  Cp += y[iM] * Calc_CP(iM, temp[i][j][k]);
				double Cv = Cp - gR / fM;//так надо!!!
				double gamma = Cp / Cv;
				double sumYh = 0.0;
				double sumDF = 0.0;
				double sumR = 0.0;
				double r_y = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];
				S[i][j][k] = (1.0 / (ro[i][j][k] * Cp * temp[i][j][k])) * ((QFx[i + 1][j][k] - QFx[i][j][k]) / DX[0] + (QFy[i][j + 1][k] - QFy[i][j][k]) / (DX[1] * r_y) + (QFz[i][j][k + 1] - QFz[i][j][k]) / (DX[2] * r_y));
			}
		}
	}*/


	//нахожу du/dx (до замены граничных условий)
	//нахожу dpix (до замены граничных условий)

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int k = lo[2]; k <= hi[2]; k++) {
				du[i][j][k] = (ru[i + 1][j][k] / ro_1[i + 1][j][k] - ru[i - 1][j][k] / ro_1[i - 1][j][k]) / (2.0 * DX[0]);
				dpx[i][j][k] = pidin[i - 1][j][k] + pidin[i + 1][j][k];
			}
		}
	}

	//нахожу dv/dy (после замены граничных условий)
	//нахожу dpiy (после замены граничных условий)


	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int k = lo[2]; k <= hi[2]; k++) {
				dpy[i][j][k] = pidin[i][j - 1][k] + pidin[i][j + 1][k];
				dpy1[i][j][k] = pidin[i][j + 1][k] - pidin[i][j - 1][k];

				double r_y = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];
				double r_yr = D_MIN[1] + (j - lo[1] + 1) * DX[1];
				double r_yl = D_MIN[1] + (j - lo[1]) * DX[1];
				dv[i][j][k] = (r_yr * (rv[i][j + 1][k] / ro_1[i][j + 1][k] + rv[i][j][k] / ro_1[i][j][k]) - r_yl * (rv[i][j][k] / ro_1[i][j][k] + rv[i][j - 1][k] / ro_1[i][j - 1][k])) / (2.0 * DX[1] * r_y);
			}
		}
	}

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int k = lo[2]; k <= hi[2]; k++) {
				dpz[i][j][k] = pidin[i][j][k - 1] + pidin[i][j][k + 1];

				double r_y = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];
				dw[i][j][k] = (rw[i][j][k + 1] / ro_1[i][j][k + 1] - rw[i][j][k - 1] / ro_1[i][j][k - 1]) / (2.0 * DX[2] * r_y);
			}
		}
	}
}

void calc_pi()
{

	for (int i = lo[0]; i <= hi[0]; i++) {
		for (int j = lo[1]; j <= hi[1]; j++) {
			for (int k = lo[2]; k <= hi[2]; k++) {
				S[i][j][k] = (ro_1[i][j][k] / TAU) * (du[i][j][k] + dv[i][j][k] + dw[i][j][k] - S[i][j][k]);
				//printf("%f\n",ro_1[i][j][k]);
				//S[i][j][k] = 0.0;
			}
		}
	}
	for (int i = 0; i < TOTAL_NX; i++) {
		for (int j = 0; j < TOTAL_NY; j++) {
			for (int k = 0; k < TOTAL_NZ; k++) {
				pidin_1[i][j][k] = pidin[i][j][k];
			}
		}
	}
	double pid, dm, dmax;
	int kkk = 0;
	do {
		kkk++;
		dmax = 0.0; // максимальное изменение значений u
		for (int i = lo[0]; i <= hi[0]; i++) {
			for (int j = lo[1]; j <= hi[1]; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					double r_y = D_MIN[1] + (j - lo[1] + 0.5) * DX[1];
					pid = pidin_1[i][j][k];
					//pidin_1[i][j][k] = (DX[1] * DX[1] * (dpx[i][j][k]) + DX[0] * DX[0] * (dpy[i][j][k]) - DX[0] * DX[0] * DX[1] * DX[1] * S[i][j][k] + dpy1[i][j][k] * DX[0] * DX[0] * DX[1] / (2 * r_y)) / (2.0 * (DX[0] * DX[0] + DX[1] * DX[1]));
					pidin_1[i][j][k] = (dpy[i][j][k] / (DX[1] * DX[1]) + dpy1[i][j][k] / (2. * r_y * DX[1]) + dpz[i][j][k] / (r_y * r_y * DX[2] * DX[2]) + dpx[i][j][k] / (DX[0] * DX[0]) - S[i][j][k]) / (2. * (1 / (DX[1] * DX[1]) + 1 / (DX[2] * DX[2] * r_y * r_y) + 1 / (DX[0] * DX[0])));
					dm = fabs(pidin_1[i][j][k] - pid);
					if (dmax < dm) dmax = dm;
				}
			}
		}
		exchange_fld(pidin_1);
		MPI_Allreduce(&dmax, &dm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		dmax = dm;
		for (int i = lo[0]; i <= hi[0]; i++) {
			for (int j = lo[1]; j <= hi[1]; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					dpx[i][j][k] = pidin_1[i - 1][j][k] + pidin_1[i + 1][j][k];
				}
			}
		}

		for (int i = lo[0]; i <= hi[0]; i++) {
			for (int j = lo[1]; j <= hi[1]; j++) {
				for (int k = lo[2]; k <= hi[2]; k++) {
					dpy[i][j][k] = pidin_1[i][j - 1][k] + pidin_1[i][j + 1][k];
					dpy1[i][j][k] = pidin_1[i][j + 1][k] - pidin_1[i][j - 1][k];
					dpz[i][j][k] = pidin_1[i][j][k - 1] + pidin_1[i][j][k + 1];
				}
			}
		}

	} //while (kkk < 4);

	while (dmax > 1.e-4);

	for (int i = 0; i < TOTAL_NX; i++) {
		for (int j = 0; j < TOTAL_NY; j++) {
			for (int k = 0; k < TOTAL_NZ; k++) {
				pidin[i][j][k] = pidin_1[i][j][k];
			}
		}
	}

}

void calc_h(double tmp,
	double hir[nMat])
{
	double t1, t2;
	for (int iM = 0; iM < nMat; iM++) {
		hir[iM] = h0[iM];
		for (int i = 0; i < n_TCP; i++) {
			t2 = exp((i + 1) * log(tmp)) / (i + 1);
			t1 = exp((i + 1) * log(TRef)) / (i + 1);
			hir[iM] += k_TCP[iM][i] * (t2 - t1);
		}
	}
}


double Calc_ML_Nv(double* y, double tmp)
{
	double fKP = 0.0;
	double* KPm = new double[nMat];
	double fM_ = 0.0;
	double KP = 0.0;

	KPm[0] = (-9.85e-12) * tmp * tmp + (3.61e-8) * tmp + (1.31e-6);
	KPm[1] = (-9.45e-12) * tmp * tmp + (3.67e-8) * tmp + (0.22e-6);


	for (int iM = 0; iM < nMat; iM++)
	{
		double fCK_M = y[iM] / Mm[iM];
		fM_ += fCK_M;
		//	if (iM < OsnV) {
		fKP += fCK_M * KPm[iM];
		//}
		/*else {
			fKP += fCK_M * ML0[iM];
		}*/
	}

	KP = fKP / fM_;

	delete[] KPm;

	return KP;
}
