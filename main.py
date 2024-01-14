import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    y = ax + b
    Y = mX + c
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = 0
        self.c = 0

    def build_model(self):
        """
        Строим модель
        """
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)

        num = 0
        den = 0
        for i in range(len(self.x)):
            num += (self.x[i] - x_mean) * (self.y[i] - y_mean)
            den += (self.x[i] - x_mean) ** 2
        self.m = num / den
        self.c = y_mean - self.m * x_mean
        print(self.m, self.c)

    def make_predictions(self):
        """
        Делаем предсказания
        """
        y_pred = self.m * self.x + self.c
        plt.scatter(self.x, self.y, color='royalblue')  # actual
        plt.plot([min(self.x), max(self.x)], [min(y_pred), max(y_pred)], color='red')  # predicted
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


def first_case(data):
    x = data.iloc[:, 1]
    y = data.iloc[:, 5]
    model = LinearRegression(x, y)
    model.build_model()
    model.make_predictions()


def second_case(data):
    x = data.iloc[:, 0]
    y = data.iloc[:, 3]
    model = LinearRegression(x, y)
    model.build_model()
    model.make_predictions()


def third_case(data):
    x = data.iloc[:, 0]
    y = data.iloc[:, 4]
    model = LinearRegression(x, y)
    model.build_model()
    model.make_predictions()


def main():
    plt.rcParams['figure.figsize'] = (12.0, 9.0)
    # Preprocessing Input data
    data = pd.read_csv('Student_Performance.csv')
    print(data)
    first_case(data)
    second_case(data)
    third_case(data)


if __name__ == '__main__':
    main()

