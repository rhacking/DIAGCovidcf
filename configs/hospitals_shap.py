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
    from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectKBest
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from util import categorical_to_int
    from probatus.feature_elimination import ShapRFECV
    from sklearn.ensemble import VotingClassifier
    processors = {
        'none': LGBMClassifier(n_jobs=-1),
        'srfe3': ShapRFECV(LGBMClassifier(n_jobs=-1), min_features_to_select=3),
        'srfe5': ShapRFECV(LGBMClassifier(n_jobs=-1), min_features_to_select=5),
        'srfe7': ShapRFECV(LGBMClassifier(n_jobs=-1), min_features_to_select=7),
        'srfe10': ShapRFECV(LGBMClassifier(n_jobs=-1), min_features_to_select=10),
        'srfe15': ShapRFECV(LGBMClassifier(n_jobs=-1), min_features_to_select=15),
        'srfe20': ShapRFECV(LGBMClassifier(n_jobs=-1), min_features_to_select=20),
        'srfe25': ShapRFECV(LGBMClassifier(n_jobs=-1), min_features_to_select=25),
        'srfe60': ShapRFECV(LGBMClassifier(n_jobs=-1), min_features_to_select=60),
        # 'sfm': SelectFromModel(LGBMClassifier(n_jobs=-1)),
    }

    # estimators = {name: make_pipeline(SimpleImputer(strategy='mean'), processor, LGBMClassifier(n_jobs=-1)) for name, processor in processors.items()}

    df = pd.read_pickle(os.path.join(DATA_PROCESSED_PATH, 'ictcf.pkl'))
    features = get_ictcf_features(df)
    categorical_to_int(df)
    X, y = df[features.input_features], df[features.output_feature]
    validator = PredefinedSplit(test_fold=(df.Metadata_Hospital == 'Union').astype(np.int32))

    return Config('hospitals_shap',
                  estimators=processors,
                  metrics=['roc_auc', 'accuracy'],
                  validator=validator,
                  X=X, y=y)
