import random
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_dataset(split_ratio=None, normalize=True):
    dataset_full = np.loadtxt(open("../pima-indians-diabetes.data.csv", "r"), delimiter=",", skiprows=0, dtype=np.float64)
    inputs = dataset_full[:, :-1]
    outputs = dataset_full[:, -1]
    outputs = outputs.astype(np.int64, copy=False)
    if split_ratio is None:
        return inputs, outputs, None, None
    inputs_train, inputs_test, outputs_train, outputs_test \
        = train_test_split(inputs, outputs, test_size=split_ratio, random_state=random.randint(0, 1000))
    if normalize:
        std_scale = preprocessing.StandardScaler().fit(inputs_train)
        inputs_train = std_scale.transform(inputs_train)
        inputs_test = std_scale.transform(inputs_test)
    return inputs_train, outputs_train, inputs_test, outputs_test

