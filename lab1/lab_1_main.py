from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from data_prepocessing import load_dataset
from lab1.my_knn import MyKNN
from lab1.my_naive_bayes import MyGaussianNaiveBayes


def main():
    # готовим датасет
    inputs_train, outputs_train, inputs_test, outputs_test = load_dataset(split_ratio=.4, normalize=True)
    # готовим классификаторы
    classificators = [
        MyGaussianNaiveBayes(),
        GaussianNB(),
        MyKNN(n_neighbors=75),
        KNeighborsClassifier(n_neighbors=75)
    ]
    # оцениваем работу каждого классификатора
    for clf in classificators:
        clf.fit(inputs_train, outputs_train)
        print(clf, '\nAccuracy: ', clf.score(inputs_test, outputs_test), '\n--------------')

main()
