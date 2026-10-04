import pandas as pd
import numpy as np
matrix_size = 200

def fill_matrix(filename):
    # Читаем CSV в DataFrame и преобразуем в массив NumPy
    m = pd.read_csv(filename, header=None).values.reshape(matrix_size, matrix_size).astype(float)
    return m

def main():
    a = fill_matrix("MatrixA.csv")
    b = fill_matrix("MatrixB.csv")
    c = fill_matrix("MatrixC.csv")
    ab = a @ b
    # Количество совпадений
    count_equal = np.sum(c == ab)
    print("Количество совпадающих элементов:", count_equal)
    # Сравнение с учётом погрешности
    arrays_close = np.allclose(c, ab, atol=1e-8)
    print("Матрицы близки (с допуском):", arrays_close)

if __name__ == '__main__':
    main()
