#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <locale>
#include <random>
#include <string>
#include <stdexcept>
#include <sys/stat.h>
using namespace std;
int ProcNum;
int ProcRank;
int Size;
double* A, * B, * C;

bool FileExists(const string& path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

void PrintMatrix(double* M) {
    for (int i = 0; i < Size; i++) {
        cout << endl;
        for (int j = 0; j < Size; j++) {
            cout << M[i * Size + j];
        }
    }
}

void MatrixGeneration(double* M, int& Size) {
    random_device rd;
    mt19937 g(rd());
    uniform_real_distribution<double> double_dist(-10.0, 10.0);
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            M[i * Size + j] = double_dist(g);
        }
    }
}

void WriteCSV(const string& filename, const double* M) {
    ofstream out(filename);
    if (!out) {
        throw runtime_error("Unable to open file for writing");
    }

    // Фиксируем формат чисел: точка как разделитель, 14 знаков после запятой
    out << fixed << setprecision(15);
    // устанавливаем локаль, чтобы у дробей всегда был '.'
    out.imbue(locale::classic());
    for (int i = 0; i < Size; i++) {
        out << M[i * Size + 0]; // первое значение в строке
        for (int j = 1; j < Size; j++) {
            out << ',' << M[i * Size + j]; // остальные с запятой
        }

        out << endl; // перевод строки после строки
    }
}

// Функция для преобразования строки в double
double StringToDouble(const string& str) {
    istringstream iss(str);
    double value;
    if (!(iss >> value)) {
        // Обработка ошибки преобразования
        throw runtime_error("Invalid double value in CSV file");
    }

    return value;
}

// Функция для чтения CSV-файла в двумерный массив
void ReadCSV(const string& filename, double*& M) {
    ifstream in(filename);
    if (!in) {
        throw runtime_error("Unable to open file for reading");
    }

    string line;
    int i = 0;
    while (i < Size && getline(in, line)) {
        stringstream ss(line);
        string cell;
        int j = 0;
        while (j < Size && getline(ss, cell, ',')) {
            M[i * Size + j] = StringToDouble(cell);
            j++;
        }

        i++;
    }

    in.close();
}

bool InitProcess(double*& A, double*& B, double*& C, int& Size, ofstream& out, int argc, char* argv[]) {
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    bool ReadFile = false;
    if (ProcRank == 0) {
        Size = 200;
        if (argc > 1) {
            try {
                Size = stoi(argv[1]);
            }
            catch (const exception& e) {
                // обработка ошибки
                Size = 200;
            }
        }

        if (argc > 2) {
            ReadFile = true;
        }

        cout << "Number of process: " << ProcNum << endl;
        out << "Number of process: " << ProcNum << endl;
        cout << "Matrix size: " << Size << endl;
        out << "Matrix size: " << Size << endl;
        if (Size < ProcNum) {
            cout << "Matrix size is less than the number of process!" << endl;
            return false;
        }
        else if (Size % ProcNum != 0) {
            cout << "Matrix size should be dividable by the number of process!" << endl;
            return false;
        }
    }

    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (ProcRank == 0) {
        A = new double[Size * Size];
        B = new double[Size * Size];
        C = new double[Size * Size];
        string fn;
        fn = "MatrixA.csv";
        if (ReadFile && FileExists(fn)) {
            cout << "Чтение массива A из файла" << endl;
            ReadCSV(fn, A);
        }
        else {
            cout << "Генерация массива A случайными значениями" << endl;
            MatrixGeneration(A, Size);
            cout << "Сохранение массива A в файл" << endl;
            WriteCSV(fn, A);
        }

        fn = "MatrixB.csv";
        if (ReadFile && FileExists(fn)) {
            cout << "Чтение массива B из файла" << endl;
            ReadCSV(fn, B);
        }
        else {
            cout << "Генерация массива B случайными значениями" << endl;
            MatrixGeneration(B, Size);
            cout << "Сохранение массива B в файл" << endl;
            WriteCSV(fn, B);
        }
    }

    return true;
}

void Flip(double*& B, int dim) {
    double temp = 0.0;
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            temp = B[i * dim + j];
            B[i * dim + j] = B[j * dim + i];
            B[j * dim + i] = temp;
        }
    }
}

void MatrixMultiplicationMPI(double*& A, double*& B, double*& C, int& Size) {
    int dim = Size;
    int i, j, k, p, ind;
    double temp;
    MPI_Status Status;
    int ProcPartSize = dim / ProcNum;
    int ProcPartElem = ProcPartSize * dim;
    double* bufA = new double[dim * ProcPartSize];
    double* bufB = new double[dim * ProcPartSize];
    double* bufC = new double[dim * ProcPartSize];
    int ProcPart = dim / ProcNum;
    int part = ProcPart * dim;
    if (ProcRank == 0) {
        Flip(B, Size);
    }

    MPI_Scatter(A, part, MPI_DOUBLE, bufA, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, part, MPI_DOUBLE, bufB, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    temp = 0.0;
    for (i = 0; i < ProcPartSize; i++) {
        for (j = 0; j < ProcPartSize; j++) {
            for (k = 0; k < dim; k++) {
                temp += bufA[i * dim + k] * bufB[j * dim + k];
            }

            bufC[i * dim + j + ProcPartSize * ProcRank] = temp;
            temp = 0.0;
        }
    }

    int NextProc;
    int PrevProc;
    for (p = 1; p < ProcNum; p++) {
        NextProc = ProcRank + 1;
        if (ProcRank == ProcNum - 1) {
            NextProc = 0;
        }

        PrevProc = ProcRank - 1;
        if (ProcRank == 0) {
            PrevProc = ProcNum - 1;
        }

        MPI_Sendrecv_replace(bufB, part, MPI_DOUBLE, NextProc, 0, PrevProc, 0, MPI_COMM_WORLD, &Status);
        temp = 0.0;
        for (i = 0; i < ProcPartSize; i++) {
            for (j = 0; j < ProcPartSize; j++) {
                for (k = 0; k < dim; k++) {
                    temp += bufA[i * dim + k] * bufB[j * dim + k];
                }

                if (ProcRank - p >= 0) {
                    ind = ProcRank - p;
                }
                else {
                    ind = ProcNum + ProcRank - p;
                }

                bufC[i * dim + j + ind * ProcPartSize] = temp;
                temp = 0.0;
            }
        }
    }

    MPI_Gather(bufC, ProcPartElem, MPI_DOUBLE, C, ProcPartElem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[]bufA;
    delete[]bufB;
    delete[]bufC;
}

int main(int argc, char* argv[])
{
    ofstream out("Duration.txt", ios::app);
    if (!out.is_open()) {
        throw runtime_error("Unable to open file for writing");
        return 1;
    }

    MPI_Init(&argc, &argv);
    if (InitProcess(A, B, C, Size, out, argc, argv)) {
        double beg = MPI_Wtime();
        MatrixMultiplicationMPI(A, B, C, Size);
        double end = MPI_Wtime();
        double parallel = end - beg;
        if (ProcRank == 0) {
            cout << "Number of process: " << ProcNum << endl << "Time of execution - Parallel calculation: " << 1000 * parallel << " ms" << endl;
            out << "Number of process: " << ProcNum << endl << "Time of execution - Parallel calculation: " << 1000 * parallel << " ms" << endl;

            string fn = "MatrixC.csv";
            cout << "Сохранение массива C в файл" << endl;
            WriteCSV(fn, C);
        }

        delete[]A;
        delete[]B;
        delete[]C;
    }

    out.close();
    MPI_Finalize();
}
