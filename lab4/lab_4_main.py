import numpy as np
from sklearn.svm import SVC
from data_prepocessing import load_dataset


# Чтение датасета
inputs_train, outputs_train, inputs_test, outputs_test = load_dataset(split_ratio=.4, normalize=True)
#  Варианты
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
penalties = np.arange(.5, 10, 0.5)
degrees = np.arange(1, 10, 1)
gammas = np.arange(0.05, 1, 0.05)
coefs0 = np.arange(0, 1, 0.1)
probabilities = [False, True]
shrinkings = [False, True]
tols = [1e-1, 1e-2, 1e-3, 1e-4]
#
for value in kernel_types:
    clf = SVC(kernel=value)
    clf.fit(inputs_train, outputs_train)
    accuracy = clf.score(inputs_test, outputs_test)
    print('kernel_types: {}, accuracy: {:.9f}'.format(value, accuracy))
print('-------------')
#
for value in penalties:
    clf = SVC(C=value)
    clf.fit(inputs_train, outputs_train)
    accuracy = clf.score(inputs_test, outputs_test)
    print('penalties: {}, accuracy: {:.9f}'.format(value, accuracy))
print('-------------')
#
for value in degrees:
    clf = SVC(kernel='poly', degree=value)
    clf.fit(inputs_train, outputs_train)
    accuracy = clf.score(inputs_test, outputs_test)
    print('degrees: {}, accuracy: {:.9f}'.format(value, accuracy))
print('-------------')
#
for value in gammas:
    clf = SVC(kernel='poly', gamma=value)
    clf.fit(inputs_train, outputs_train)
    accuracy = clf.score(inputs_test, outputs_test)
    print('gammas: {}, accuracy: {:.9f}'.format(value, accuracy))
print('-------------')
#
for value in coefs0:
    clf = SVC(kernel='poly', coef0=value)
    clf.fit(inputs_train, outputs_train)
    accuracy = clf.score(inputs_test, outputs_test)
    print('coefs0: {}, accuracy: {:.9f}'.format(value, accuracy))
print('-------------')
#
for value in probabilities:
    clf = SVC(probability=value)
    clf.fit(inputs_train, outputs_train)
    accuracy = clf.score(inputs_test, outputs_test)
    print('probabilities: {}, accuracy: {:.9f}'.format(value, accuracy))
print('-------------')
#
for value in shrinkings:
    clf = SVC(shrinking=value)
    clf.fit(inputs_train, outputs_train)
    accuracy = clf.score(inputs_test, outputs_test)
    print('shrinkings: {}, accuracy: {:.9f}'.format(value, accuracy))
print('-------------')
#
for value in tols:
    clf = SVC(tol=value)
    clf.fit(inputs_train, outputs_train)
    accuracy = clf.score(inputs_test, outputs_test)
    print('tols: {}, accuracy: {:.9f}'.format(value, accuracy))
print('-------------')
#
# best params run
clf = SVC(C=3.5, kernel='poly', degree=1, gamma=0.15, coef0=0.6)
clf.fit(inputs_train, outputs_train)
accuracy = clf.score(inputs_test, outputs_test)
print('-------------')
print('Best params SVM')
print('{}'.format(clf))
print('Accuracy: {:.9f}'.format(accuracy))
