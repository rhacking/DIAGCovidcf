def generate_config():
    from sklearn.model_selection import PredefinedSplit, StratifiedKFold

    from evaluation.evaluation import Config
    import pandas as pd
    import os
    from data.base import DATA_PROCESSED_PATH
    from data.ictcf import get_ictcf_features
    import numpy as np

    from lightgbm.sklearn import LGBMClassifier
    from sklearn.feature_selection import chi2
    from sklearn.svm import SVC
    from sklearn.feature_selection import RFE, RFECV, SelectFromModel
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from util import categorical_to_int
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier

    df = pd.read_pickle(os.path.join(DATA_PROCESSED_PATH, 'ictcf.pkl'))
    features = get_ictcf_features(df)
    categorical_to_int(df)

    n_input = len(features.input_features)

    # processors = {
    #     'none': None,
    #     'rfe.5': RFE(LGBMClassifier(n_jobs=-1)),
    #     'rfe.1': RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=int(0.1*n_input)),
    #     'rfe.2': RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=int(0.2*n_input)),
    #     'rfe.3': RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=int(0.3*n_input)),
    #     'rfe.4': RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=int(0.4*n_input)),
    #     'rfe.6': RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=int(0.6*n_input)),
    #     'rfe.7': RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=int(0.7*n_input)),
    #     'rfe.8': RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=int(0.8*n_input)),
    #     'rfe.0': RFE(LGBMClassifier(n_jobs=-1), n_features_to_select=int(0.9*n_input)),
    #     'sfm': SelectFromModel(LGBMClassifier(n_jobs=-1)),
    # }
    #
    # estimators = {name: make_pipeline(SimpleImputer(strategy='mean'), processor, LGBMClassifier(n_jobs=-1)) for name, processor in processors.items()}
    # estimators.update({
    #     'svc': make_pipeline(SimpleImputer(strategy='mean'), SVC()),
    #     'LR': make_pipeline(SimpleImputer(strategy='mean'), LogisticRegression())
    # })

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import StackingClassifier
    from sklearn.pipeline import Pipeline
    from skopt import BayesSearchCV

    def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,
                             n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                             n_points=1, iid=True, refit=True, cv=None, verbose=0,
                             pre_dispatch='2*n_jobs', random_state=None,
                             error_score='raise', return_train_score=False):
        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        self.fit_params = fit_params

        super(BayesSearchCV, self).__init__(
            estimator=estimator, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)

    BayesSearchCV.__init__ = bayes_search_CV_init

    clf1 = LogisticRegression(random_state=1, max_iter=800)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    clf4 = LGBMClassifier(n_jobs=-1)
    clf5 = LGBMClassifier(n_jobs=-1, boosting_type='dart')
    clf6 = LGBMClassifier(n_jobs=-1, boosting_type='goss')

    model = Pipeline([
        ('classifier',
         (LGBMClassifier(n_jobs=-1, num_leaves=40, n_estimators=200, learning_rate=0.04, boosting_type='gbdt')))
    ])
    bayes_search = BayesSearchCV(model, {
        'classifier__n_estimators': (10, 600),
        'classifier__boosting_type': ['dart', 'gbdt', 'goss'],
        'classifier__num_leaves': (3, 233),
        'classifier__reg_alpha': (1e-9, 1e2, 'log-uniform'),
        'classifier__reg_lambda': (1e-9, 1e3, 'log-uniform'),
        'classifier__learning_rate': (1e-7, 5e-1, 'log-uniform'),
        'classifier__subsample': (0.1, 1.0),
        'classifier__subsample_freq': (0, 12),
        'classifier__min_child_samples': (2, 80),
        'classifier__max_depth': (1, 30),
        'classifier__max_bin': [31, 63, 127, 255, 300],
        'classifier__extra_trees': [False, True],
        'classifier__colsample_bytree': (0.1, 1.0)
    }, scoring='accuracy', n_iter=50)

    estimators = {
        'lgbm': LGBMClassifier(n_jobs=-1),
        'voting': make_pipeline(SimpleImputer(), VotingClassifier(estimators=[
            ('clf1', clf1),
            ('clf2', clf2),
            ('clf3', clf3),
            ('clf4', clf4),
            ('clf5', clf5),
            ('clf6', clf6),
        ], voting='soft')),
        'stack': make_pipeline(SimpleImputer(), StackingClassifier(estimators=[
            ('clf1', clf1),
            ('clf2', clf2),
            ('clf3', clf3),
            ('clf4', clf4),
            ('clf5', clf5),
            ('clf6', clf6),
        ])),
        'stack_lgbm': make_pipeline(SimpleImputer(), StackingClassifier(estimators=[
            ('clf1', clf1),
            ('clf2', clf2),
            ('clf3', clf3),
            ('clf4', clf4),
            ('clf5', clf5),
            ('clf6', clf6),
        ], final_estimator=LGBMClassifier(n_jobs=-1))),
        'bayes_search': bayes_search
    }


    X, y = df[features.input_features], df[features.output_feature]
    validator = PredefinedSplit(test_fold=(df.Metadata_Hospital == 'Union').astype(np.int32))

    return Config('test',
                  estimators=estimators,
                  metrics=['roc_auc', 'accuracy'],
                  validator=validator,
                  X=X, y=y)
