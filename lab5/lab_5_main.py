from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from data_prepocessing import load_dataset


def print_score(model, inputs, outputs):
    print('Model: {}'.format(model))
    print('CrossValidation results:')
    #
    scorings = ['accuracy', 'neg_log_loss', 'roc_auc']
    for scoring in scorings:
        results = cross_val_score(model, inputs, outputs, cv=KFold(n_splits=10), scoring=scoring)
        print('{}: {:.3f} ({:.3f})'.format(scoring, results.mean(), results.std()))
    #
    X_train, X_test, Y_train, Y_test = train_test_split(inputs, outputs, test_size=0.4)
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    #
    print('ConfusionMatrix:\n', confusion_matrix(Y_test, predicted))
    print('ClassificationReport:\n', classification_report(Y_test, predicted))
    #
    print('--------------')


def main():
    inputs, outputs, _, _ = load_dataset()
    print_score(GaussianNB(), inputs, outputs)
    print_score(KNeighborsClassifier(n_neighbors=75), inputs, outputs)


main()
