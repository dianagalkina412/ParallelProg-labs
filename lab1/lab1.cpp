#include <iostream>
#include <fstream>
#include <filesystem>
#include <random> 
using namespace std;
const int MATRIX_SIZE = 25;

void matrix_generation(double M[MATRIX_SIZE][MATRIX_SIZE]) {
    random_device rd;
    mt19937 g(rd());
    uniform_real_distribution<double> double_dist(-10.0, 10.0);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            M[i][j] = double_dist(g);
        }
    }
}

void write_csv(const string& filename, const double M[MATRIX_SIZE][MATRIX_SIZE]) {
    ofstream out(filename);
    if (!out) {
        throw runtime_error("Unable to open file for writing");
    }

    // Фиксируем формат чисел: точка как разделитель, 14 знаков после запятой
    out << fixed << setprecision(15);
    // устанавливаем локаль, чтобы у дробей всегда был '.'
    out.imbue(locale::classic());
    for (int i = 0; i < MATRIX_SIZE; i++) {
        out << M[i][0]; // первое значение в строке
        for (int j = 1; j < MATRIX_SIZE; j++) {
            out << ',' << M[i][j]; // остальные с запятой
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
void read_csv(const string& filename, double M[MATRIX_SIZE][MATRIX_SIZE]) {
    ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Unable to open file for reading");
    }

    string line;
    int i = 0;
    while (getline(in, line) || i < MATRIX_SIZE) {
        stringstream ss(line);
        string cell;
        int j = 0;
        while (getline(ss, cell, ',') || j < MATRIX_SIZE) {
            M[i][j] = stringToDouble(cell);
            j++;
        }

        i++;
    }

    in.close();
}

void matrix_multiplication(const double A[MATRIX_SIZE][MATRIX_SIZE], const double B[MATRIX_SIZE][MATRIX_SIZE], double C[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            C[i][j] = 0; // Начальное значение элемента
            for (int k = 0; k < MATRIX_SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main()
{
    string fn;
    double A[MATRIX_SIZE][MATRIX_SIZE];
    double B[MATRIX_SIZE][MATRIX_SIZE];
    double C[MATRIX_SIZE][MATRIX_SIZE];

    fn = "MatrixA.csv";
    if (filesystem::exists((filesystem::path)fn)) {
        read_csv(fn, A);
    }
    else {
        matrix_generation(A);
        write_csv(fn, A);
    }

    fn = "MatrixB.csv";
    if (filesystem::exists((filesystem::path)fn)) {
        read_csv(fn, B);
    }
    else {
        matrix_generation(B);
        write_csv(fn, B);
    }

    ofstream out("Duration.txt");
    if (!out) {
        throw runtime_error("Unable to open file for writing");
    }

    int cnt = 10;
    double sum_duration = 0.0;
    for (int i = 0; i < cnt; i++) {
        // Записываем время начала
        auto start = chrono::steady_clock::now();

        // Вызываем метод
        matrix_multiplication(A, B, C);

        // Записываем время окончания
        auto end = chrono::steady_clock::now();
        // Вычисляем длительность
        chrono::duration<double> duration = 1000.0 * (end - start);
        sum_duration += duration.count();
        // Выводим результат
        cout << "Время выполнения " << i + 1 << ": " << duration.count() << " мс" << endl;
        out << "Время выполнения " << i + 1 << ": " << duration.count() << " мс" << endl;
    }

    cout << "-------------------------------------" << endl;
    out << "-------------------------------------" << endl;
    cout << "Среднее время выполнения: " << sum_duration / cnt << " мс" << endl;
    out << "Среднее время выполнения: " << sum_duration / cnt << " мс" << endl;

    fn = "MatrixC.csv";
    write_csv(fn, C);

    return 0;
}
