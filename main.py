import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models import FirstModel, SecondModel, ThirdModel


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
        plt.hist(column, bins=15, edgecolor='black')
        plt.title(name_column)
        plt.show()

    @classmethod
    def __visualize_info_column_activities(cls, column, num_column, name_column):
        print(f"Информация о столбце №{num_column}: {name_column}")
        print("Значения принимают либо \"Yes\" либо \"No\"\n")
        bool_count_yes = (column == 'Yes')
        bool_count_no = (column == 'No')
        count_yes = np.count_nonzero(bool_count_yes)
        count_no = np.count_nonzero(bool_count_no)
        fig, ax = plt.subplots()
        ax.pie([count_yes, count_no], labels=[f'Да ({count_yes})', f'Нет ({count_no})'], autopct='%1.1f%%')
        plt.title(name_column)
        plt.show()

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
        cls.__visualize_info_column_activities(data.iloc[:, 2], 3, "Дополнительная активность")
        cls.__visualize_info_column(data.iloc[:, 3], 4, "Часы сна")
        cls.__visualize_info_column(data.iloc[:, 4], 5, "Образцы вопросов, отработанных на практике")
        cls.__visualize_info_column(data.iloc[:, 5], 6, "Индекс производительности")
        print()


def task(task_class, normalized_data):
    task_obj = task_class(normalized_data)
    print("==============================================================")
    print(task_obj.name_model)
    find_best = task_obj.find_best()
    for i in task_obj.find_values:
        print(f"Признаки для модели: {i.param_names}")
        print(f"Вектор с найденными коэффициентами: {i.b_vector}")
        print(f"Вектор с найденными коэффициентами:\n{i.b_vector}")
        print(f"Найденные значения y:\n{i.y_fit}")
        print(f"Найденные значения остатков:\n{i.epsilon}")
        print(f"Значение коэффициента детерминации: {i.r_square}\n")
    print(f"||Лучшие признаки для модели: {find_best.param_names}")
    print(f"Вектор с найденными коэффициентами: {find_best.b_vector}")
    print(f"Значение коэффициента детерминации: {find_best.r_square}")
    print("==============================================================\n\n")


def main():
    data = pd.read_csv('Student_Performance.csv')
    VisualizeData.visualize_info_dataset(data)
    normalized_data = NormalizeData.normalize_data(data)
    task(FirstModel, normalized_data)
    task(SecondModel, normalized_data)
    task(ThirdModel, normalized_data)


if __name__ == '__main__':
    main()
