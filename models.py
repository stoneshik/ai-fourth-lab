import numpy as np


class LinearRegression:
    def __init__(self, x_train, y_train, x_test, y_test, param_names):
        """
        x = [[1 x11 x12 ... x1p]
             [1 x21 x12 ... x1p]
             [.. .. ... ... ...]
             [1 xn1 xn2 ... xnp]]
        y = [y1 y2 ... yn] (это столбец)
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.param_names = param_names
        self.b_vector = self.__calc_b_vector()  # вектор с найденными коэффициентами
        self.y_fit = self.__fit()
        self.epsilon = self.__calc_epsilon()  # вектор остатков
        self.r_square = self.__calc_r_square()  # коэффициент детерминации

    def __calc_b_vector(self):
        x_transpose = np.transpose(self.x_train)
        return np.linalg.inv(x_transpose.dot(self.x_train)).dot(x_transpose).dot(self.y_train)

    def __fit(self):
        y_fit = np.zeros(len(self.y_test))
        for num_row, x_row in enumerate(self.x_test):
            y_fit[num_row] = sum([x_value * self.b_vector[i] for i, x_value in enumerate(x_row)])
        return y_fit

    def __calc_epsilon(self):
        epsilon = np.zeros(len(self.y_test))
        for i, y_value, y_value_fit in zip(range(len(self.y_test)), self.y_test, self.y_fit):
            epsilon[i] = abs(y_value - y_value_fit)
        return epsilon

    def __calc_r_square(self):
        return 1 - (self.epsilon.var() / self.y_test.var())


class FirstModel:
    def __init__(self, normalized_train_data, normalized_test_data):
        self.normalized_train_data = normalized_train_data
        self.normalized_test_data = normalized_test_data
        self.find_values = []
        self.name_model = "Модель для значения индекса производительности"

    def __first(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 3))
        x_train[:, 1] = self.normalized_train_data[:, 0]
        x_train[:, 2] = self.normalized_train_data[:, 1]
        y_train = self.normalized_train_data[:, 5]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 3))
        x_test[:, 1] = self.normalized_test_data[:, 0]
        x_test[:, 2] = self.normalized_test_data[:, 1]
        y_test = self.normalized_test_data[:, 5]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
            'Часы обучения; Предыдущая оценка;'
        )
        self.find_values.append(linear_regression)

    def __second(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 4))
        x_train[:, 1] = self.normalized_train_data[:, 0]
        x_train[:, 2] = self.normalized_train_data[:, 1]
        x_train[:, 3] = self.normalized_train_data[:, 2]
        y_train = self.normalized_train_data[:, 5]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 4))
        x_test[:, 1] = self.normalized_test_data[:, 0]
        x_test[:, 2] = self.normalized_test_data[:, 1]
        x_test[:, 3] = self.normalized_test_data[:, 2]
        y_test = self.normalized_test_data[:, 5]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
            'Часы обучения; Предыдущая оценка; Дополнительная активность;'
        )
        self.find_values.append(linear_regression)

    def __third(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 6))
        x_train[:, 1] = self.normalized_train_data[:, 0]
        x_train[:, 2] = self.normalized_train_data[:, 1]
        x_train[:, 3] = self.normalized_train_data[:, 2]
        x_train[:, 4] = self.normalized_train_data[:, 3]
        x_train[:, 5] = self.normalized_train_data[:, 4]
        y_train = self.normalized_train_data[:, 5]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 6))
        x_test[:, 1] = self.normalized_test_data[:, 0]
        x_test[:, 2] = self.normalized_test_data[:, 1]
        x_test[:, 3] = self.normalized_test_data[:, 2]
        x_test[:, 4] = self.normalized_test_data[:, 3]
        x_test[:, 5] = self.normalized_test_data[:, 4]
        y_test = self.normalized_test_data[:, 5]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
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
    def __init__(self, normalized_train_data, normalized_test_data):
        self.normalized_train_data = normalized_train_data
        self.normalized_test_data = normalized_test_data
        self.find_values = []
        self.name_model = "Модель для значения часов обучения"

    def __first(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 3))
        x_train[:, 1] = self.normalized_train_data[:, 1]
        x_train[:, 2] = self.normalized_train_data[:, 5]
        y_train = self.normalized_train_data[:, 0]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 3))
        x_test[:, 1] = self.normalized_test_data[:, 1]
        x_test[:, 2] = self.normalized_test_data[:, 5]
        y_test = self.normalized_test_data[:, 0]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
            'Предыдущая оценка; Индекс производительности'
        )
        self.find_values.append(linear_regression)

    def __second(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 5))
        x_train[:, 1] = self.normalized_train_data[:, 1]
        x_train[:, 2] = self.normalized_train_data[:, 3]
        x_train[:, 3] = self.normalized_train_data[:, 4]
        x_train[:, 4] = self.normalized_train_data[:, 5]
        y_train = self.normalized_train_data[:, 0]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 5))
        x_test[:, 1] = self.normalized_test_data[:, 1]
        x_test[:, 2] = self.normalized_test_data[:, 3]
        x_test[:, 3] = self.normalized_test_data[:, 4]
        x_test[:, 4] = self.normalized_test_data[:, 5]
        y_test = self.normalized_test_data[:, 0]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
            'Предыдущая оценка; Часы сна; Образцы вопросов, отработанных на практике; Индекс производительности'
        )
        self.find_values.append(linear_regression)

    def __third(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 6))
        x_train[:, 1] = self.normalized_train_data[:, 1]
        x_train[:, 2] = self.normalized_train_data[:, 2]
        x_train[:, 3] = self.normalized_train_data[:, 3]
        x_train[:, 4] = self.normalized_train_data[:, 4]
        x_train[:, 5] = self.normalized_train_data[:, 5]
        y_train = self.normalized_train_data[:, 0]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 6))
        x_test[:, 1] = self.normalized_test_data[:, 1]
        x_test[:, 2] = self.normalized_test_data[:, 2]
        x_test[:, 3] = self.normalized_test_data[:, 3]
        x_test[:, 4] = self.normalized_test_data[:, 4]
        x_test[:, 5] = self.normalized_test_data[:, 5]
        y_test = self.normalized_test_data[:, 0]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
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
    def __init__(self, normalized_train_data, normalized_test_data):
        self.normalized_train_data = normalized_train_data
        self.normalized_test_data = normalized_test_data
        self.find_values = []
        self.name_model = "Модель для значения предыдущей оценки"

    def __first(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 3))
        x_train[:, 1] = self.normalized_train_data[:, 0]
        x_train[:, 2] = self.normalized_train_data[:, 3]
        y_train = self.normalized_train_data[:, 1]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 3))
        x_test[:, 1] = self.normalized_test_data[:, 0]
        x_test[:, 2] = self.normalized_test_data[:, 3]
        y_test = self.normalized_test_data[:, 1]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
            'Часы обучения; Часы сна;'
        )
        self.find_values.append(linear_regression)

    def __second(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 4))
        x_train[:, 1] = self.normalized_train_data[:, 0]
        x_train[:, 2] = self.normalized_train_data[:, 2]
        x_train[:, 3] = self.normalized_train_data[:, 3]
        y_train = self.normalized_train_data[:, 1]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 4))
        x_test[:, 1] = self.normalized_test_data[:, 0]
        x_test[:, 2] = self.normalized_test_data[:, 2]
        x_test[:, 3] = self.normalized_test_data[:, 3]
        y_test = self.normalized_test_data[:, 1]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
            'Часы обучения; Дополнительная активность; Часы сна;'
        )
        self.find_values.append(linear_regression)

    def __third(self):
        num_rows_train = len(self.normalized_train_data)
        x_train = np.ones((num_rows_train, 6))
        x_train[:, 1] = self.normalized_train_data[:, 0]
        x_train[:, 2] = self.normalized_train_data[:, 2]
        x_train[:, 3] = self.normalized_train_data[:, 3]
        x_train[:, 4] = self.normalized_train_data[:, 4]
        x_train[:, 5] = self.normalized_train_data[:, 5]
        y_train = self.normalized_train_data[:, 1]

        num_rows_test = len(self.normalized_test_data)
        x_test = np.ones((num_rows_test, 6))
        x_test[:, 1] = self.normalized_test_data[:, 0]
        x_test[:, 2] = self.normalized_test_data[:, 2]
        x_test[:, 3] = self.normalized_test_data[:, 3]
        x_test[:, 4] = self.normalized_test_data[:, 4]
        x_test[:, 5] = self.normalized_test_data[:, 5]
        y_test = self.normalized_test_data[:, 1]

        linear_regression = LinearRegression(
            x_train,
            y_train,
            x_test,
            y_test,
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
