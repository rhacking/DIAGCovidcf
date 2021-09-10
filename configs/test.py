def generate_config():
    from sklearn.model_selection import RepeatedStratifiedKFold

    from data.ictcf import load_ictcf
    from evaluation.evaluation import Config

    from lightgbm.sklearn import LGBMClassifier
    estimators = {
        'gbdt': LGBMClassifier(n_jobs=-1),
        'dart': LGBMClassifier(n_jobs=-1, boosting_type='dart')
    }

    validator = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
    X, y = load_ictcf()

    return Config('test',
                  estimators=estimators,
                  metrics=['roc_auc', 'accuracy'],
                  validator=validator,
                  X=X, y=y)
