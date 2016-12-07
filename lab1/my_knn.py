import csv
import random
import math
import operator
from collections import defaultdict


class MyKNN:

    def __init__(self, n_neighbors=5):
        self.train_inputs = []
        self.train_outputs = []
        self.K = n_neighbors

    def fit(self, inputs, outputs):
        """
        "Обучает" классификатор.
        По факту тупо запоминает значения, чтобы использовать в дальнейшем.
        :param inputs:
        :param outputs:
        :return:
        """
        self.train_inputs = inputs
        self.train_outputs = outputs

    def predict(self, inputs):
        """
        Классифицирует входные данные
        :param inputs:
        :return:
        """
        res = []
        for x in range(len(inputs)):
            neighbors = self.__get_neighbors(inputs[x])
            result = self.select_neighbor(neighbors)
            res.append(result)
        return res

    def __get_neighbors(self, test_object):
        """
        Возвращает список блажайших К соседей
        :param test_object:
        :return:
        """
        distances = []
        length = len(test_object)
        for i in range(len(self.train_inputs)):
            dist = self.dist(test_object, self.train_inputs[i], length)
            distances.append((self.train_outputs[i], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for i in range(self.K):
            neighbors.append(distances[i][0])
        return neighbors

    def dist(self, object_a, object_b, d):
        """
        Находит евклидово расстояние в D-мерном пространстве
        :param object_a:
        :param object_b:
        :param d:
        :return:
        """
        distance = 0
        for i in range(d):
            distance += pow((object_a[i] - object_b[i]), 2)
        return math.sqrt(distance)

    def select_neighbor(self, neighbors):
        """
        Выбирает класс соседей
        :param neighbors:
        :return:
        """
        res_dict = defaultdict(int)
        for x in neighbors:
            res_dict[x] += 1
        res = sorted(res_dict.items(), key=operator.itemgetter(1), reverse=True)
        return res[0][0]

    def score(self, inputs, outputs):
        """
        Классифицирует входные данные и возвращает посчитанную точность по известным выходным данным.
        :param inputs:
        :param outputs:
        :return:
        """
        predicted = self.predict(inputs)
        correct = 0
        for i in range(len(outputs)):
            if outputs[i] == predicted[i]:
                correct += 1
        return correct/float(len(outputs))

    def __str__(self):
        return 'MyKNN with N_neighbors={}'.format(self.K)