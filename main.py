import pandas as pd
import numpy as np


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
        self.epsilon = self.__calc_epsilon()  # вектор остатков
        self.r_square = self.__calc_r_square()  # коэффициент детерминации

    def __calc_b_vector(self):
        x_transpose = np.transpose(self.x_matrix)
        return np.linalg.inv(x_transpose.dot(self.x_matrix)).dot(x_transpose).dot(self.y_matrix)

    def __fit(self):
        y_fit = np.zeros(len(self.y_matrix))
        for num_row, x_row in enumerate(self.x_matrix):
            y_fit[num_row] = sum([x_value * self.b_vector[i] for i, x_value in enumerate(x_row)])
        return y_fit

    def __calc_epsilon(self):
        epsilon = np.zeros(len(self.y_matrix))
        for i, y_value, y_value_fit in zip(range(len(self.y_matrix)), self.y_matrix, self.y_fit):
            epsilon[i] = y_value - y_value_fit
        return epsilon

    def __calc_r_square(self):
        return 1 - (self.epsilon.var() / self.y_matrix.var())


def normalize_column(column):
    """
    Используется мин макс нормализация результат принимает значение от 0 до 100
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
    linear_regression = LinearRegression(x_matrix, y_matrix)
    print("Модель №1")
    print("Для поиска значения индекса производительности")
    print(f"Вектор с найденными коэффициентами:\n{linear_regression.b_vector}")
    print(f"Найденные значения y:\n{linear_regression.y_fit}\n")
    print(f"Найденные значения остатков:\n{linear_regression.epsilon}\n")
    print(f"Найденные значения коэффициента детерминации:\n{linear_regression.r_square}\n")


def second(normalized_data):
    num_rows = len(normalized_data)
    x_matrix = np.ones((num_rows, 6))
    x_matrix[:, 1] = normalized_data[:, 1]
    x_matrix[:, 2] = normalized_data[:, 2]
    x_matrix[:, 3] = normalized_data[:, 3]
    x_matrix[:, 4] = normalized_data[:, 4]
    x_matrix[:, 5] = normalized_data[:, 5]
    y_matrix = normalized_data[:, 0]
    linear_regression = LinearRegression(x_matrix, y_matrix)
    print("Модель №2")
    print("Для поиска значения индекса производительности")
    print(f"Вектор с найденными коэффициентами:\n{linear_regression.b_vector}")
    print(f"Найденные значения y:\n{linear_regression.y_fit}\n")
    print(f"Найденные значения остатков:\n{linear_regression.epsilon}\n")
    print(f"Найденные значения коэффициента детерминации:\n{linear_regression.r_square}\n")


def third(normalized_data):
    num_rows = len(normalized_data)
    x_matrix = np.ones((num_rows, 6))
    x_matrix[:, 1] = normalized_data[:, 0]
    x_matrix[:, 2] = normalized_data[:, 2]
    x_matrix[:, 3] = normalized_data[:, 3]
    x_matrix[:, 4] = normalized_data[:, 4]
    x_matrix[:, 5] = normalized_data[:, 5]
    y_matrix = normalized_data[:, 1]
    linear_regression = LinearRegression(x_matrix, y_matrix)
    print("Модель №3")
    print("Для поиска значения индекса производительности")
    print(f"Вектор с найденными коэффициентами:\n{linear_regression.b_vector}")
    print(f"Найденные значения y:\n{linear_regression.y_fit}\n")
    print(f"Найденные значения остатков:\n{linear_regression.epsilon}\n")
    print(f"Найденные значения коэффициента детерминации:\n{linear_regression.r_square}\n")


def visualize_info_column(column, num_column, name_column):
    print(f"Информация о столбце №{num_column}: {name_column}")
    print(f"Среднее значение: {column.mean()}")
    print(f"Стандартное отклонение: {column.std()}")
    print(f"Минимальное значение: {column.min()}")
    print(f"Максимальное значение: {column.max()}")
    print(f"Первый квантиль: {np.quantile(column, 0)}")
    print(f"Второй квантиль: {np.quantile(column, 0.25)}")
    print(f"Третий квантиль: {np.quantile(column, 0.50)}")
    print(f"Четвертый квантиль: {np.quantile(column, 0.75)}")
    print(f"Пятый квантиль: {np.quantile(column, 1)}")
    print()


def visualize_info_dataset(data):
    num_rows = len(data)
    num_columns = len(data.columns)
    print("Информация о датасете")
    print("Содержит столбцы: Часы обучения; Предыдущая оценка; Дополнительная активность; Часы сна; Образцы вопросов, отработанных на практике; Индекс производительности")
    print(f"Количество строк: {num_rows}")
    print(f"Количество столбцов: {num_columns}\n")
    visualize_info_column(data.iloc[:, 0], 1, "Часы обучения")
    visualize_info_column(data.iloc[:, 1], 2, "Предыдущая оценка")
    print(f"Информация о столбце №3: Дополнительная активность")
    print("Значения принимают либо \"Yes\" либо \"No\"\n")
    visualize_info_column(data.iloc[:, 3], 4, "Часы сна")
    visualize_info_column(data.iloc[:, 4], 5, "Образцы вопросов, отработанных на практике")
    visualize_info_column(data.iloc[:, 5], 6, "Индекс производительности")
    print()


def main():
    data = pd.read_csv('Student_Performance.csv')
    visualize_info_dataset(data)
    normalized_data = normalize_data(data)
    first(normalized_data)
    second(normalized_data)
    third(normalized_data)


if __name__ == '__main__':
    main()
