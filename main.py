import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, x_matrix, y_matrix):
        """
        x = [[1 x11 x12 ... x1p]
             [1 x21 x12 ... x1p]
             [.. .. ... ... ...]
             [1 xn1 xn2 ... xnp]]
        y = [y1 y2 ... yn] (это столбец)
        """
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.b_vector = self.__calc_b_vector()  # вектор с найденными коэффициентами
        self.y_fit = self.__fit()

    def __calc_b_vector(self):
        x_transpose = np.transpose(self.x_matrix)
        return np.linalg.inv(x_transpose.dot(self.x_matrix)).dot(x_transpose).dot(self.y_matrix)

    def __fit(self):
        pass


def normalize_column(column):
    """
    Используется мин макс нормализация результат приниамет значение от 0 до 100
    """
    if column[0] == 'Yes' or column[0] == 'No':
        to_numbers_func = np.vectorize(lambda x: 100.0 if x == 'Yes' else 0.0)
        column = to_numbers_func(column)
    column_min = column.min()
    column_max = column.max()
    normalize_func = np.vectorize(lambda x: (x - column_min) / (column_max - column_min) * 100)
    return normalize_func(column)


def normalize_data(data):
    num_rows = len(data)
    num_columns = len(data.columns)
    matrix = np.zeros((num_rows, num_columns))
    for num_column, column in enumerate(data.columns):
        matrix[:, num_column] = normalize_column(data[column].to_numpy())
    return matrix


def first(normalized_data):
    num_rows = len(normalized_data)
    x_matrix = np.ones((num_rows, 6))
    x_matrix[:, 1] = normalized_data[:, 0]
    x_matrix[:, 2] = normalized_data[:, 1]
    x_matrix[:, 3] = normalized_data[:, 2]
    x_matrix[:, 4] = normalized_data[:, 3]
    x_matrix[:, 5] = normalized_data[:, 4]
    y_matrix = normalized_data[:, 5]
    print(x_matrix)
    print('=======================')
    print(y_matrix)
    print('=======================')
    linear_regression = LinearRegression(x_matrix, y_matrix)
    print(linear_regression.b_vector)


def main():
    plt.rcParams['figure.figsize'] = (12.0, 9.0)
    # Preprocessing Input data
    data = pd.read_csv('Student_Performance.csv')
    print(normalize_data(data))
    print('=======================')
    normalized_data = normalize_data(data)
    first(normalized_data)


if __name__ == '__main__':
    main()
