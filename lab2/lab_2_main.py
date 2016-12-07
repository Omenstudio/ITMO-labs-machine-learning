from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import tree

from data_prepocessing import load_dataset


def calc_state(split_ratio):
    res = [split_ratio]
    inputs_train, outputs_train, inputs_test, outputs_test = load_dataset(split_ratio=split_ratio, normalize=True)
    classificators = [
        tree.DecisionTreeClassifier(),
        RandomForestClassifier()
    ]
    for clf in classificators:
        clf.fit(inputs_train, outputs_train)
        res.append(clf.score(inputs_test, outputs_test))
    return res


def main():
    split_ratio = .5
    print('ratio', 'DecisionTree', 'RandomForest')
    while split_ratio < .9:
        ans = calc_state(split_ratio)
        print('{:.1f}\t\t{:.5f}\t\t{:.5f}'.format(ans[0], ans[1], ans[2]))
        split_ratio += .1


main()
