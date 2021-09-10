from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from lightgbm.sklearn import LGBMClassifier
from bonsai.base.c45tree import C45Tree
import seaborn as sns
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from typing import List, Any
from dataclasses import dataclass
from typing import Union
from numba import njit, prange, jit
# @njit(fastmath=True)
# @profile
# def gini_impurity(X: np.ndarray):
#     # result = 0
#     # classes = np.unique(X[:, -1])
#     classes = [0, 1]
#     # for c in classes:
#     #     result += ((X[:, -1]==c).sum()/X.shape[0])**2
#     n_samples = X.shape[0]
#     n_negative = (X[:, -1] == 0).sum()
#     n_positive = n_samples - n_negative
#     result = (n_negative/n_samples)**2 + (n_positive/n_samples)**2
#     return 1 - result

def gini_impurity(x, w=None):
    # The rest of the code requires numpy arrays.
    # x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        if cumx[-1] == 0:
            return 0
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# @njit(fastmath=True)
# @profile
def gini_index(Xs: List[np.ndarray]):
#     total_size = np.sum(np.array([X.shape[0] for X in Xs]))
    total_size = 0
    for X in Xs:
        total_size += X.shape[0]
    if total_size == 0:
        return 1
    result = 0
    for X in Xs:
        result += X.shape[0]/total_size * gini_impurity(X[:, -1])
    return result

@dataclass
class Leaf:
    value: int
    n_samples: int

@dataclass
class Tree:
    left: Union['Tree', Leaf]
    right: Union['Tree', Leaf]
    split_feature: int
    split_value: float
    n_samples: int

# @njit
# @profile
def split_data(X: np.ndarray, feature_index: int, split_value: float):
    X_feature = X[:, feature_index]
    is_na = np.isnan(X_feature)
    less_than_split = (X_feature < split_value)
    left = X[less_than_split | is_na]
    right = X[~less_than_split | is_na]
    return left, right

# @njit
# @profile
def generate_split_points(X: np.ndarray, feature_index: int):
    splits = np.sort(X[:, feature_index])
    splits = splits[~np.isnan(splits)]
    splits = np.unique(splits)
    splits = (splits[:-1] + splits[1:])/2
    return set(splits)

# @njit(fastmath=True)
# @profile
def get_split(X: np.ndarray):
    best_gini, best_feature, best_split, best_groups = np.inf, None, None, None
    for feature_index in range(X.shape[1]-1):
        for split_value in generate_split_points(X, feature_index):
            groups = split_data(X, feature_index, split_value)
            gini = gini_index(groups)
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_index
                best_split = split_value
                best_groups = groups
    current_gini = gini_impurity(X)
    if current_gini < best_gini:
        return 0, 0, 0, (None, None)
    return best_feature, best_split, best_gini, best_groups

# @njit
def most_frequent(x):
    return np.argmax(np.bincount(x.astype(np.int32)))


# @jit()
# @profile
def decision_tree(X: np.ndarray, last_gini=None):
    best_feature, best_split, best_gini, (left, right) = get_split(X)
    if left is None or last_gini == best_gini:
        return Leaf(most_frequent(X[:, -1]), X.shape[0])

    if len(np.unique(left[:, -1])) == 1:
        left_tree = Leaf(most_frequent(left[:, -1]), left.shape[0])
    elif len(np.unique(left[:, -1])) == 0:
        left_tree = Leaf(most_frequent(X[:, -1]), 1)
    else:
        left_tree = decision_tree(left, last_gini=best_gini)

    if len(np.unique(right[:, -1])) == 1:
        right_tree = Leaf(most_frequent(right[:, -1]), right.shape[0])
    elif len(np.unique(right[:, -1])) == 0:
        right_tree = Leaf(most_frequent(X[:, -1]), 1)
    else:
        right_tree = decision_tree(right, last_gini=best_gini)

    return Tree(left_tree, right_tree, best_feature, best_split, X.shape[0])

# @profile
def predict(tree: Union[Tree, Leaf], sample: np.ndarray):
    if isinstance(tree, Leaf):
        return tree.value

    if sample[tree.split_feature] < tree.split_value:
        return predict(tree.left, sample)
    elif np.isnan(sample[tree.split_feature]):
        total_samples = tree.left.n_samples + tree.right.n_samples
        return (predict(tree.left, sample) * tree.left.n_samples + predict(tree.right,
                                                                           sample) * tree.right.n_samples) / total_samples

    return predict(tree.right, sample)


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=0):
        pass

    def fit(self, data, labels):
        self.tree = decision_tree(np.concatenate([data, labels[:, None]], axis=1))

    def predict(self, data):
        return np.array([np.round(predict(self.tree, data[i, :])) for i in range(data.shape[0])])


def main():
    # X, y = load_breast_cancer(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    # X_train[y_train == 1, 4] = np.nan
    # X_test[y_test == 0, 4] = np.nan

    df = pd.read_pickle('data/processed/ictcf.pkl')
    df = df.sample(frac=0.4)

    df_train = df[df.Meta.Metadata_Hospital == 'Union']
    df_test = df[df.Meta.Metadata_Hospital != 'Union']

    import numpy as np
    import re
    def prepare(X):
        for col in X.select_dtypes(include=['category']):
            X[col] = X[col].cat.codes
            X.loc[X[col] < 0, col] = np.nan

        X.columns = [re.sub(r'[^0-9a-zA-Z_-]+ ', '', '_'.join(col).strip()) for col in X.columns.values]
        return X

    X_train, y_train = df_train.Input, df_train.Target.Metadata_PCR
    X_test, y_test = df_test.Input, df_test.Target.Metadata_PCR
    X_train = prepare(X_train)
    X_test = prepare(X_test)

    for i in range(1):
        dt = DecisionTree()
        dt.fit(X_train.values, (y_train == 'Positive').values)
        print(dt.score(X_test.values, (y_test == 'Positive').values))

if __name__ == '__main__':
    main()