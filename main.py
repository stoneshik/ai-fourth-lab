import pandas as pd
import numpy as np


class LinearRegression:
    def __init__(self, x_matrix, y_matrix, param_names):
        """
        x = [[1 x11 x12 ... x1p]
             [1 x21 x12 ... x1p]
             [.. .. ... ... ...]
             [1 xn1 xn2 ... xnp]]
        y = [y1 y2 ... yn] (это столбец)
        """
        self.x_matrix = x_matrix
        self.y_matrix = y_matrix
        self.param_names = param_names
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


class FirstModel:
    def __init__(self, normalized_data):
        self.normalized_data = normalized_data
        self.find_values = []
        self.name_model = "Модель для индекса производительности"

    def __first(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 3))
        x_matrix[:, 1] = self.normalized_data[:, 0]
        x_matrix[:, 2] = self.normalized_data[:, 1]
        y_matrix = self.normalized_data[:, 5]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Часы обучения; Предыдущая оценка;'
        )
        self.find_values.append(linear_regression)

    def __second(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 4))
        x_matrix[:, 1] = self.normalized_data[:, 0]
        x_matrix[:, 2] = self.normalized_data[:, 1]
        x_matrix[:, 3] = self.normalized_data[:, 2]
        y_matrix = self.normalized_data[:, 5]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Часы обучения; Предыдущая оценка; Дополнительная активность;'
        )
        self.find_values.append(linear_regression)

    def __third(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 6))
        x_matrix[:, 1] = self.normalized_data[:, 0]
        x_matrix[:, 2] = self.normalized_data[:, 1]
        x_matrix[:, 3] = self.normalized_data[:, 2]
        x_matrix[:, 4] = self.normalized_data[:, 3]
        x_matrix[:, 5] = self.normalized_data[:, 4]
        y_matrix = self.normalized_data[:, 5]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Часы обучения; Предыдущая оценка; Дополнительная активность; Часы сна; Образцы вопросов, отработанных на практике;'
        )
        self.find_values.append(linear_regression)

    def find_best(self):
        self.__first()
        self.__second()
        self.__third()
        max_find_value = 0
        max_find_value_obj = None
        for find_value in self.find_values:
            if find_value.r_square > max_find_value:
                max_find_value = find_value.r_square
                max_find_value_obj = find_value
        return max_find_value_obj


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


class NormalizeData:
    @classmethod
    def __normalize_column(cls, column):
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

    @classmethod
    def normalize_data(cls, data):
        num_rows = len(data)
        num_columns = len(data.columns)
        matrix = np.zeros((num_rows, num_columns))
        for num_column, column in enumerate(data.columns):
            matrix[:, num_column] = cls.__normalize_column(data[column].to_numpy())
        return matrix


class VisualizeData:
    @classmethod
    def __visualize_info_column(cls, column, num_column, name_column):
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

    @classmethod
    def visualize_info_dataset(cls, data):
        num_rows = len(data)
        num_columns = len(data.columns)
        print("Информация о датасете")
        print("Содержит столбцы: Часы обучения; Предыдущая оценка; Дополнительная активность; Часы сна; Образцы вопросов, отработанных на практике; Индекс производительности")
        print(f"Количество строк: {num_rows}")
        print(f"Количество столбцов: {num_columns}\n")
        cls.__visualize_info_column(data.iloc[:, 0], 1, "Часы обучения")
        cls.__visualize_info_column(data.iloc[:, 1], 2, "Предыдущая оценка")
        print(f"Информация о столбце №3: Дополнительная активность")
        print("Значения принимают либо \"Yes\" либо \"No\"\n")
        cls.__visualize_info_column(data.iloc[:, 3], 4, "Часы сна")
        cls.__visualize_info_column(data.iloc[:, 4], 5, "Образцы вопросов, отработанных на практике")
        cls.__visualize_info_column(data.iloc[:, 5], 6, "Индекс производительности")
        print()


def task(task_class, normalized_data):
    task_obj = task_class(normalized_data)
    print(task_obj.name_model)
    find_best = task_obj.find_best()
    for i in task_obj.find_values:
        print(f"Параметры для модели: {i.param_names}")
        print(f"Вектор с найденными коэффициентами: {i.b_vector}")
        print(f"Значение коэффициента детерминации: {i.r_square}\n")
    print(f"||Лучшие параметры для модели: {find_best.param_names}")
    print(f"Вектор с найденными коэффициентами: {find_best.b_vector}")
    print(f"Значение коэффициента детерминации: {find_best.r_square}")


def main():
    data = pd.read_csv('Student_Performance.csv')
    VisualizeData.visualize_info_dataset(data)
    normalized_data = NormalizeData.normalize_data(data)
    task(FirstModel, normalized_data)
    #second(normalized_data)
    #third(normalized_data)


if __name__ == '__main__':
    main()
