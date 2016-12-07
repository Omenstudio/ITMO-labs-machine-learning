import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn import metrics


# Чтение датасета
dataset_full = np.loadtxt(open("../pima-indians-diabetes.data.csv", "r"), delimiter=",", skiprows=0, dtype=np.float64)
inputs = dataset_full[:, :-1]
outputs = dataset_full[:, -1]
outputs = outputs.astype(np.int64, copy=False)

# Визуализация двух признаков
plt.figure(figsize=(10, 8))
for class_number, marker, color in zip((0, 1), ('x', 'o'), ('blue', 'red')):
    # Вычисление коэффициента корреляции Пирсона
    R = pearsonr(inputs[:, 1][outputs == class_number], inputs[:, 3][outputs == class_number])
    # Отображение класса на графике
    plt.scatter(x=inputs[:, 1][outputs == class_number],
                y=inputs[:, 3][outputs == class_number],
                marker=marker,
                color=color,
                alpha=0.7,
                label='class {:}, R={:.4f}'.format(class_number, R[0])
                )
corel_coef = pearsonr(inputs[:, 1], inputs[:, 3])[0]
plt.title('Pima Indians Diabetes, K={}'.format(corel_coef))
plt.xlabel('Plasma glucose concentration')
plt.ylabel('Triceps skin fold thickness')
plt.legend(loc='upper right')
plt.show()
print('Correlation coef (Pearson)={}'.format(corel_coef))

# Разбиение выборки и нормализация
inputs_train, inputs_test, outputs_train, outputs_test \
    = train_test_split(inputs, outputs, test_size=0.30, random_state=123)
std_scale = preprocessing.StandardScaler().fit(inputs_train)
inputs_train = std_scale.transform(inputs_train)
inputs_test = std_scale.transform(inputs_test)

# Визуализация двух признаков для разных выборок
f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))
for a, inp, outp in zip(ax, (inputs_train, inputs_test), (outputs_train, outputs_test)):
    for class_number, marker, color in zip((0, 1), ('x', 'o'), ('blue', 'red')):
        a.scatter(x=inp[:, 1][outp == class_number],
                  y=inp[:, 3][outp == class_number],
                  marker=marker,
                  color=color,
                  alpha=0.7,
                  label='class {}: {:.2%}'.format(class_number, list(outp).count(class_number) / outp.shape[0])
                  )
    a.legend(loc='upper right')
ax[0].set_title('Training Dataset')
ax[1].set_title('Test Dataset')
f.text(0.5, 0.04, 'Plasma glucose concentration', ha='center', va='center')
f.text(0.08, 0.5, 'Triceps skin fold thickness', ha='center', va='center', rotation='vertical')
plt.show()

# Разбиение LDA и визуализация
sklearn_lda = LDA(n_components=2)  # TODO: почему-то возвращает не два значимых вектора, а один!! print(sklearn_transf.shape) = (537, 1)
sklearn_transf = sklearn_lda.fit(inputs_train, outputs_train).transform(inputs_train)
# print(sklearn_transf.shape)
plt.figure(figsize=(10, 8))
for label, marker, color in zip((0, 1), ('x', 'o'), ('blue', 'red')):
    plt.scatter(x=sklearn_transf[:, 0][outputs_train == label],
                y=sklearn_transf[:, 0][outputs_train == label],
                marker=marker,
                color=color,
                alpha=0.7,
                label='class {}'.format(label)
                )
plt.xlabel('vector 1')
plt.ylabel('vector 1')
plt.legend()
plt.title('Most significant singular vectors after linear transformation via LDA')
plt.show()

# ТЕСТ LDA и QDA
print('------------')
classificators = [LDA(), QDA()]
for clf in classificators:
    clf.fit(inputs_train, outputs_train)
    pred_train = clf.predict(inputs_train)
    pred_test = clf.predict(inputs_test)
    print(clf)
    print('Точность на обучающем наборе {:.2%}'.format(metrics.accuracy_score(outputs_train, pred_train)))
    print('Точность на тестовом наборе {:.2%}'.format(metrics.accuracy_score(outputs_test, pred_test)))
    print('------------')


