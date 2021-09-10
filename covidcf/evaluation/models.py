import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone as estimator_clone
from sklearn.metrics import roc_curve, precision_recall_curve

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer, SimpleImputer


class MultiOutputClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self._estimators = []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for col in y.columns:
            estimator = estimator_clone(self.base_estimator)
            estimator.fit(X, y[col])
            self._estimators.append(estimator)

    def predict(self, X):
        result = np.stack(
            [np.argmax(pred, axis=1).astype(bool) for pred in self.predict_proba(X)]).T
        return result

    def predict_proba(self, X):
        preds_list = [estimator.predict_proba(X) for estimator in self._estimators]
        return preds_list

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


class ShapZeroingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.background = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.background = X
        self.base_estimator.fit(X, y)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        # try:
        #     explainer = shap.TreeExplainer(self.base_estimator)
        # except:
        #     explainer = shap.KernelExplainer(self.base_estimator.predict_proba, X)
        try:
            explainer = shap.Explainer(self.base_estimator)
        except:
            masker = shap.maskers.Independent(data=self.background)
            explainer = shap.Explainer(self.base_estimator, masker=masker)
        shap_values = explainer(X)
        if shap_values.base_values.ndim == 2:
            preds = inv_logit(
                shap_values.base_values[:, 1].astype(np.float) + ((~np.isnan(np.array(X).astype(np.float))) * shap_values.values[:, :, 1].astype(np.float)).sum(axis=1))
        else:
            preds = inv_logit(
                shap_values.base_values.astype(np.float) + ((~np.isnan(np.array(X).astype(np.float))) * shap_values.values.astype(np.float)).sum(axis=1))
        return np.stack([1 - preds, preds]).T


# class TestTimeImputingClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, base_estimator, imputer, n_repeat):
#         self.base_estimator = base_estimator
#         self.imputer = imputer
#         self.n_repeat = n_repeat
#
#     def _impute_data(self, X, y, should_fit):
#         X_rep = []
#         if y is not None:
#             y_rep = []
#
#         if should_fit:
#             self.imputer.fit(X)
#
#         for i in range(self.n_repeat):
#             X_rep.append(self.imputer.transform(X))
#             if y is not None:
#                 y_rep.append(y)
#
#         X_rep = pd.concat([pd.DataFrame(x) for x in X_rep])
#         if y is not None:
#             y_rep = pd.concat([pd.Series(y) for y in y_rep])
#             return X_rep, y_rep
#         else:
#             return X_rep
#
#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
#         X_rep, y_rep = self._impute_data(X, y, should_fit=True)
#         self.base_estimator.fit(X_rep, y_rep)
#
#     def predict(self, X):
#         return np.argmax(self.predict_proba(X), axis=1)
#
#     def predict_proba(self, X):
#
#         X_rep = self._impute_data(X, None, should_fit=False)
#         preds = self.base_estimator.predict_proba(X_rep)
#         n_preds = X.shape[0]
#         #         print(np.array([preds[n_preds*i:n_preds*(i+1)] for i in range(self.n_repeat)]).shape)
#         return np.mean([preds[n_preds * i:n_preds * (i + 1)] for i in range(self.n_repeat)], axis=0)
#
#     def shap_values(self, X):
#         X_rep = self._impute_data(X, None, should_fit=False)
#         n_preds = X.shape[0]
#         explainer = shap.TreeExplainer(self.base_estimator)
#         return np.mean([explainer(X_rep[n_preds * i:n_preds * (i + 1)]).values for i in range(self.n_repeat)], axis=0)

from sklearn.base import clone


class TestTimeImputingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, imputer, n_repeat, ensemble=True, impute_in_predict=True):
        self.base_estimator = base_estimator
        self.imputer = imputer
        self.n_repeat = n_repeat
        self.ensemble = ensemble
        self.impute_in_predict = impute_in_predict

        if ensemble:
            self._estimators = []

    def _impute_data(self, X, y, should_fit):
        X_rep = []
        if y is not None:
            y_rep = []

        if should_fit:
            self.imputer.fit(X)

        for i in range(self.n_repeat):
            X_rep.append(self.imputer.transform(X))
            if y is not None:
                y_rep.append(y)

        X_rep = pd.concat([pd.DataFrame(x) for x in X_rep])
        if y is not None:
            y_rep = pd.concat([pd.Series(y) for y in y_rep])
            return X_rep, y_rep
        else:
            return X_rep

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if isinstance(self.imputer, IterativeImputer):
            self.imputer.sample_posterior = True
        elif isinstance(self.imputer, ColumnImputer):
            self.predict_imputer = SimpleImputer(strategy='mean').fit(X, y)
            if isinstance(X, pd.DataFrame):
                self.cols_to_use = np.where(~X.isna().all())[0]
                X = X.iloc[:, self.cols_to_use]
            else:
                self.cols_to_use = np.where(~np.isnan(X.astype(np.float)).all(axis=0))[0]
                X = X[:, self.cols_to_use]
        X_rep, y_rep = self._impute_data(X, y, should_fit=True)
        for col in X_rep.select_dtypes(include=[np.object]):
            X_rep[col] = X_rep[col].astype(np.float)
        if self.ensemble:
            n_samples = X.shape[0]
            self._estimators = []
            for i in range(self.n_repeat):
                X_i, y_i = X_rep[n_samples * i:n_samples * (i + 1)], y_rep[n_samples * i:n_samples * (i + 1)]
                estimator = clone(self.base_estimator)
                estimator.fit(X_i, y_i)
                self._estimators.append(estimator)
        else:
            self.base_estimator.fit(X_rep, y_rep)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if isinstance(self.imputer, IterativeImputer):
            self.imputer.sample_posterior = False
            return self.base_estimator.predict_proba(self.imputer.transform(X))
        elif isinstance(self.imputer, ColumnImputer):
            # if isinstance(X, pd.DataFrame):
            #     X = X.iloc[:, self.cols_to_use]
            # else:
            #     X = X[:, self.cols_to_use]
            if not self.ensemble:
                return self.base_estimator.predict_proba(self.predict_imputer.transform(X))
            else:
                X_imputed = self.predict_imputer.transform(X)
                return np.mean([est.predict_proba(X_imputed) for est in self._estimators], axis=0)

        if self.impute_in_predict:
            X_rep = self._impute_data(X, None, should_fit=False)

        #         print(np.array([preds[n_preds*i:n_preds*(i+1)] for i in range(self.n_repeat)]).shape)
        if self.ensemble:
            n_samples = X.shape[0]
            if self.impute_in_predict:
                return np.mean([est.predict_proba(X_rep[n_samples * i:n_samples * (i + 1)]) for i, est in
                                enumerate(self._estimators)], axis=0)
            else:
                return np.mean([est.predict_proba(X) for i, est in enumerate(self._estimators)], axis=0)
        else:
            preds = self.base_estimator.predict_proba(X_rep)
            n_preds = X.shape[0]
            return np.mean([preds[n_preds * i:n_preds * (i + 1)] for i in range(self.n_repeat)], axis=0)

    def shap_values(self, X):
        if not isinstance(self.imputer, ColumnImputer):
            raise ValueError('oop cant handle this')

        try:
            X = self.predict_imputer.transform(X)
        except ValueError:
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, self.cols_to_use]
            else:
                X = X[:, self.cols_to_use]
            X = self.predict_imputer.transform(X)

        if self.ensemble:
            shap_values = [shap.Explainer(est)(X) for est in self._estimators]
            return np.mean([s.values for s in shap_values], axis=0), np.mean([s.base_values for s in shap_values],
                                                                             axis=0), self.cols_to_use
        else:
            explainer = shap.Explainer(self.base_estimator)(X)
            return explainer.values, explainer.base_values, self.cols_to_use
        # X_rep = self._impute_data(X, None, should_fit=False)
        # n_preds = X.shape[0]
        # explainer = shap.TreeExplainer(self.base_estimator)
        # shap_values = [explainer(X_rep[n_preds * i:n_preds * (i + 1)]) for i in range(self.n_repeat)]
        # return np.mean([s.values for s in shap_values], axis=0), np.mean([s.base_values for s in shap_values], axis=0)

    def base_values(self, X):
        X_rep = self._impute_data(X, None, should_fit=False)
        n_preds = X.shape[0]
        explainer = shap.TreeExplainer(self.base_estimator)
        return np.mean([explainer(X_rep[n_preds * i:n_preds * (i + 1)]).base_values for i in range(self.n_repeat)],
                       axis=0)


class ColumnImputer(TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.options = {col: X[col].dropna().values for col in X.columns}
        else:
            self.options = {i: X[:, i].astype(np.float)[~np.isnan(X.astype(np.float)[:, i])] for i in range(X.shape[1])}
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if len(self.options[col]) > 0:
                    X.loc[X[col].isna(), col] = np.random.choice(self.options[col], size=X[col].isna().sum())
                else:
                    X.loc[X[col].isna(), col] = 0
        else:
            for i in range(X.shape[1]):
                if len(self.options[i]) > 0:
                    X[np.isnan(X.astype(np.float)[:, i]), i] = np.random.choice(self.options[i], size=len(X.astype(np.float)[np.isnan(X.astype(np.float)[:, i]), i]))
                    assert np.isnan(X.astype(np.float)[:, i]).sum() == 0
                else:
                    X[np.isnan(X.astype(np.float)[:, i]), i] = 0

        assert not np.any(np.isnan(np.array(X).astype(np.float)))
        return X


class BestThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, thresh_type='f1'):
        self.base_estimator = base_estimator
        self.thresh_type = thresh_type

    def fit(self, X, y):
        # self.base_estimator.fit(X, y)
        if self.thresh_type == 'roc':
            fpr, tpr, thresholds = roc_curve(y, self.base_estimator.predict_proba(X)[:, 1])
            gmeans = np.sqrt(tpr * (1 - fpr))
            self.threshold = thresholds[np.argmax(gmeans)]
        elif self.thresh_type == 'f1':
            # TODO: Is this correct??
            precision, recall, thresholds = precision_recall_curve(get_cat_codes(y), self.base_estimator.predict_proba(X)[:, 1])
            fscore = (2 * precision * recall) / (precision + recall)
            self.threshold = thresholds[np.argmax(fscore)]
        return self

    def predict(self, X):
        return self.base_estimator.predict_proba(X)[:, 1] > self.threshold

    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)


def get_cat_codes(x: pd.Series):
    if x.dtype.name == 'category':
        return x.cat.codes
    else:
        return x.values