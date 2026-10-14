#include <filesystem>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <random>
using namespace std;
const int MATRIX_SIZE = 2000;
const int NUM_THREADS = 12;

void matrix_generation(vector<double>& M) {
    random_device rd;
    mt19937 g(rd());
    uniform_real_distribution<double> double_dist(-10.0, 10.0);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            M[i * MATRIX_SIZE + j] = double_dist(g);
        }
    }
}

void write_csv(const string& filename, const vector<double>& M) {
    ofstream out(filename);
    if (!out) {
        throw runtime_error("Unable to open file for writing");
    }

    // Фиксируем формат чисел: точка как разделитель, 14 знаков после запятой
    out << fixed << setprecision(15);
    // устанавливаем локаль, чтобы у дробей всегда был '.'
    out.imbue(locale::classic());
    for (int i = 0; i < MATRIX_SIZE; i++) {
        out << M[i * MATRIX_SIZE + 0]; // первое значение в строке
        for (int j = 1; j < MATRIX_SIZE; j++) {
            out << ',' << M[i * MATRIX_SIZE + j]; // остальные с запятой
        }

        out << endl; // перевод строки после строки
    }
}

// Функция для преобразования строки в double
double stringToDouble(const string& str) {
    istringstream iss(str);
    double value;
    if (!(iss >> value)) {
        // Обработка ошибки преобразования
        throw runtime_error("Invalid double value in CSV file");
    }

    return value;
}

// Функция для чтения CSV-файла в двумерный массив
void read_csv(const string& filename, vector<double>& M) {
    ifstream in(filename);
    if (!in) {
        throw runtime_error("Unable to open file for reading");
    }

    string line;
    int i = 0;
    while (getline(in, line) || i < MATRIX_SIZE) {
        stringstream ss(line);
        string cell;
        int j = 0;
        while (getline(ss, cell, ',') || j < MATRIX_SIZE) {
            M[i * MATRIX_SIZE + j] = stringToDouble(cell);
            j++;
        }

        i++;
    }

    in.close();
}

void matrix_multiplication(const vector<double>& A, const vector<double>& B, vector<double>& C) {
#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            double sum = 0.0;
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                sum += A[i * MATRIX_SIZE + k] * B[k * MATRIX_SIZE + j];
            }

            C[i * MATRIX_SIZE + j] = sum;
        }
    }
}

int main()
{
    string fn;
    vector<double> A(MATRIX_SIZE * MATRIX_SIZE, 0);
    vector<double> B(MATRIX_SIZE * MATRIX_SIZE, 0);
    vector<double> C(MATRIX_SIZE * MATRIX_SIZE, 0);

    fn = "MatrixA.csv";
    if (filesystem::exists((filesystem::path)fn)) {
        cout << "Чтение массива A из файла" << endl;
        read_csv(fn, A);
    }
    else {
        cout << "Генерация массива A случайными значениями" << endl;
        matrix_generation(A);
        cout << "Сохранение массива A в файл" << endl;
        write_csv(fn, A);
    }

    fn = "MatrixB.csv";
    if (filesystem::exists((filesystem::path)fn)) {
        cout << "Чтение массива B из файла" << endl;
        read_csv(fn, B);
    }
    else {
        cout << "Генерация массива B случайными значениями" << endl;
        matrix_generation(B);
        cout << "Сохранение массива B в файл" << endl;
        write_csv(fn, B);
    }

    ofstream out("Duration.txt");
    if (!out) {
        throw runtime_error("Unable to open file for writing");
    }

#ifdef _OPENMP
    cout << "OpenMP Version: " << _OPENMP / 100 << " (" << _OPENMP % 100 << ")" << endl;
    out << "OpenMP Version: " << _OPENMP / 100 << " (" << _OPENMP % 100 << ")" << endl;
#else
    cout << "Sequential Version" << endl;
    out << "Sequential Version" << endl;
#endif
    cout
        << "Processors: " << omp_get_num_procs()
        << ", Max threads: " << omp_get_max_threads()
        << endl;
    out
        << "Processors: " << omp_get_num_procs()
        << ", Max threads: " << omp_get_max_threads()
        << endl;

    int cnt = 10;
    double sum_duration = 0.0;
    cout
        << "Вычисление произведения матриц A и B (размер " << MATRIX_SIZE << "X" << MATRIX_SIZE
        << ", количество потоков " << NUM_THREADS
        << ", количество попыток " << cnt << "):"
        << endl;
    out
        << "Вычисление произведения матриц A и B (размер " << MATRIX_SIZE << "X" << MATRIX_SIZE
        << ", количество потоков " << NUM_THREADS
        << ", количество попыток " << cnt << "):"
        << endl;
    for (int i = 0; i < cnt; i++) {
        // Записываем время начала
        auto start = omp_get_wtime();

        // Вызываем метод
        matrix_multiplication(A, B, C);

        // Записываем время окончания
        auto end = omp_get_wtime();
        // Вычисляем длительность
        double duration = 1000.0 * (end - start);
        sum_duration += duration;
        // Выводим результат
        cout << "Время выполнения попытки №" << i + 1 << ": " << duration << " мс" << endl;
        out << "Время выполнения попытки №" << i + 1 << ": " << duration << " мс" << endl;
    }

    cout << "-------------------------------------" << endl;
    out << "-------------------------------------" << endl;
    cout << "Среднее время выполнения: " << sum_duration / cnt << " мс" << endl;
    out << "Среднее время выполнения: " << sum_duration / cnt << " мс" << endl;

    fn = "MatrixC.csv";
    cout << "Сохранение массива C в файл" << endl;
    write_csv(fn, C);

    return 0;
}
