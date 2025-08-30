// VTK_writer.cpp
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <stdexcept>
#include <iomanip>
const double Pi = 3.14;
// структура для хранения точки
struct DecartCoord {
    double x, y, z, value;
};

// создаём 3D массив
// NOTE: здесь мы выделяем второй индекс с размером (nfi + 1) — последний узел по φ равен 2π
DecartCoord*** createDecartCoords(double*** uinRes, int nr, int nfi, int nz, double hfi, double hr, double hz) {
    if (nr <= 0 || nfi <= 0 || nz <= 0) return nullptr;

    // выделяем память: coords[i][j][k], где j = 0..nfi (включительно) => nfi+1 узлов по углу
    DecartCoord*** coords = new DecartCoord * *[nr];
    for (int i = 0; i < nr; ++i) {
        coords[i] = new DecartCoord * [nfi + 1];
        for (int j = 0; j <= nfi; ++j) {
            coords[i][j] = new DecartCoord[nz];
        }
    }

    // заполняем: для j == nfi ставим phi = 2*pi и value = value при j == 0 (периодичность)
    for (int i = 0; i < nr; ++i) {
        double r = i * hr;
        for (int j = 0; j <= nfi; ++j) {
            double fi = (j == nfi) ? 2.0 * Pi : j * hfi;
            for (int k = 0; k < nz; ++k) {
                double z = k * hz;
                coords[i][j][k].x = r * cos(fi);
                coords[i][j][k].y = r * sin(fi);
                coords[i][j][k].z = z;

                // для последнего узла по углу ставим значение из j==0 (периодичность)
                if (j == nfi) {
                    coords[i][j][k].value = uinRes[i][0][k];
                }
                else {
                    coords[i][j][k].value = uinRes[i][j][k];
                }
            }
        }
    }

    return coords;
}

// освобождение памяти, соответствующей createDecartCoords
void freeDecartCoords(DecartCoord*** coords, int nr, int nfi, int nz) {
    if (!coords) return;
    for (int i = 0; i < nr; ++i) {
        if (!coords[i]) continue;
        for (int j = 0; j <= nfi; ++j) {
            delete[] coords[i][j];
        }
        delete[] coords[i];
    }
    delete[] coords;
}

// запись в VTK Structured Grid
// Теперь мы явно пишем DIMENSIONS nr (X) , (nfi+1) (Y), nz (Z)
// И записываем точки в порядке: Z -> Y -> X (VTK expected order)
void writeVTK(DecartCoord*** decartCoords, int nr, int nfi, int nz, const std::string& filename) {
    if (!decartCoords) throw std::runtime_error("decartCoords is null");

    int ny = nfi + 1; // добавили последний узел по φ
    int totalPoints = nr * ny * nz;

    std::ofstream file(filename + ".vtk");
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи");
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "Cylindrical to Cartesian grid\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    file << "DIMENSIONS " << nr << " " << ny << " " << nz << "\n";
    file << "POINTS " << totalPoints << " float\n";

    file << std::setprecision(9);

    // VTK ожидает порядок вложенности: Z, Y, X (то есть самый быстрый индекс - X)
    for (int k = 0; k < nz; ++k) {         // Z
        for (int j = 0; j < ny; ++j) {     // Y (φ: 0..nfi inclusive)
            for (int i = 0; i < nr; ++i) { // X (r)
                const DecartCoord& p = decartCoords[i][j][k];
                file << p.x << " " << p.y << " " << p.z << "\n";
            }
        }
    }

    // поле
    file << "\nPOINT_DATA " << totalPoints << "\n";
    file << "SCALARS scalars float 1\n";
    file << "LOOKUP_TABLE default\n";

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nr; ++i) {
                file << decartCoords[i][j][k].value << "\n";
            }
        }
    }

    file.close();
}
