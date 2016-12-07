import math
from collections import defaultdict


class MyGaussianNaiveBayes:

    def __init__(self):
        self.__summaries = None

    def fit(self, inputs, outputs):
        """
        Обучает классификатор по входным inputs и выходным outputs данным.
        :param inputs:
        :param outputs:
        :return:
        """
        class_dict = self.__combine_by_class(inputs, outputs)
        self.__summaries = {}
        for k, v in class_dict.items():
            self.__summaries[k] = self.__calc_params(v)

    def __combine_by_class(self, inputs, outputs):
        """
        Разделяет обучающую выборку по классам таким образом,
        чтобы можно было получить все элементы, принадлежащие определенному классу.
        :param inputs:
        :param outputs:
        :return: dict[class, elems]
        """
        res = defaultdict(list)
        for i in range(len(inputs)):
            res[outputs[i]].append(inputs[i])
        return res

    def __calc_params(self, dataset):
        """
        :param dataset:
        :return: Среднее значение и среднеквадратичное отклонение для каждого "свойства" входных данных
        """
        return [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]

    def mean(self, mas):
        """
        :param mas:
        :return: среднее значение массива mas
        """
        return sum(mas) / float(len(mas))

    def stdev(self, mas):
        """
        :param numbers:
        :return: среднеквадратичное отклонение в массиве mas
        """
        avg = self.mean(mas)
        variance = sum([pow(x-avg, 2) for x in mas])/float(len(mas)-1)
        return math.sqrt(variance)


    def predict(self, inputs):
        """
        Классифицирует входные данные
        :param inputs:
        :return:
        """
        predictions = []
        for i in range(len(inputs)):
            result = self.__predict_single(inputs[i])
            predictions.append(result)
        return predictions

    def __predict_single(self, input_data):
        """
        Классифицирует один объект.
        :param input_data:
        :return:
        """
        probabilities = self.__calc_allclass_probabilities(input_data)
        className, res = None, -1
        for classValue, probability in probabilities.items():
            if className is None or probability > res:
                res = probability
                className = classValue
        return className

    def __calc_allclass_probabilities(self, input_data):
        """
        Считает вероятности принадлежности объекта ко всем классам
        :param input_data:
        :return:
        """
        res = {}
        for className, classValues in self.__summaries.items():
            res[className] = 1
            for i in range(len(classValues)):
                mean, stdev = classValues[i]
                x = input_data[i]
                res[className] *= self.__calc_single_probability(x, mean, stdev)
        return res


    def __calc_single_probability(self, x, mean, stdev):
        """
        Апостериорная вероятность принадлежности объекта к определенному классу (с предрасчитанными средний и отклонением).
        Магия по формуле.
        :param x:
        :param mean:
        :param stdev:
        :return:
        """
        exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

    def score(self, inputs, outputs):
        """
        Классифицирует входные данные, сравнивает с выходными и возвращает точность классификации.
        :param inputs:
        :param outputs:
        :return:
        """
        predicted = self.predict(inputs)
        correct = 0
        for i in range(len(inputs)):
            if outputs[i] == predicted[i]:
                correct += 1
        return correct / float(len(inputs))

    def __str__(self):
        return 'MyGaussianNaiveBayes'
