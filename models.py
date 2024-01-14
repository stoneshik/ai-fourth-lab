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
        self.name_model = "Модель для значения индекса производительности"

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


class SecondModel:
    def __init__(self, normalized_data):
        self.normalized_data = normalized_data
        self.find_values = []
        self.name_model = "Модель для значения часов обучения"

    def __first(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 3))
        x_matrix[:, 1] = self.normalized_data[:, 1]
        x_matrix[:, 2] = self.normalized_data[:, 5]
        y_matrix = self.normalized_data[:, 0]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Предыдущая оценка; Индекс производительности'
        )
        self.find_values.append(linear_regression)

    def __second(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 5))
        x_matrix[:, 1] = self.normalized_data[:, 1]
        x_matrix[:, 2] = self.normalized_data[:, 3]
        x_matrix[:, 3] = self.normalized_data[:, 4]
        x_matrix[:, 4] = self.normalized_data[:, 5]
        y_matrix = self.normalized_data[:, 0]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Предыдущая оценка; Часы сна; Образцы вопросов, отработанных на практике; Индекс производительности'
        )
        self.find_values.append(linear_regression)

    def __third(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 6))
        x_matrix[:, 1] = self.normalized_data[:, 1]
        x_matrix[:, 2] = self.normalized_data[:, 2]
        x_matrix[:, 3] = self.normalized_data[:, 3]
        x_matrix[:, 4] = self.normalized_data[:, 4]
        x_matrix[:, 5] = self.normalized_data[:, 5]
        y_matrix = self.normalized_data[:, 0]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Предыдущая оценка; Дополнительная активность; Часы сна; Образцы вопросов, отработанных на практике; Индекс производительности'
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


class ThirdModel:
    def __init__(self, normalized_data):
        self.normalized_data = normalized_data
        self.find_values = []
        self.name_model = "Модель для значения предыдущей оценки"

    def __first(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 3))
        x_matrix[:, 1] = self.normalized_data[:, 0]
        x_matrix[:, 2] = self.normalized_data[:, 3]
        y_matrix = self.normalized_data[:, 1]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Часы обучения; Часы сна;'
        )
        self.find_values.append(linear_regression)

    def __second(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 4))
        x_matrix[:, 1] = self.normalized_data[:, 0]
        x_matrix[:, 2] = self.normalized_data[:, 2]
        x_matrix[:, 3] = self.normalized_data[:, 3]
        y_matrix = self.normalized_data[:, 1]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Часы обучения; Дополнительная активность; Часы сна;'
        )
        self.find_values.append(linear_regression)

    def __third(self):
        num_rows = len(self.normalized_data)
        x_matrix = np.ones((num_rows, 6))
        x_matrix[:, 1] = self.normalized_data[:, 0]
        x_matrix[:, 2] = self.normalized_data[:, 2]
        x_matrix[:, 3] = self.normalized_data[:, 3]
        x_matrix[:, 4] = self.normalized_data[:, 4]
        x_matrix[:, 5] = self.normalized_data[:, 5]
        y_matrix = self.normalized_data[:, 1]
        linear_regression = LinearRegression(
            x_matrix,
            y_matrix,
            'Часы обучения; Дополнительная активность; Часы сна; Образцы вопросов, отработанных на практике; Индекс производительности'
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
