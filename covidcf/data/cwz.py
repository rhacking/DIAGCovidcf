import os
import ast

import click
import pandas as pd
import numpy as np

from . import ensure_data_dirs
from .base import DATA_RAW_PATH, DATA_PROCESSED_PATH


@click.command()
def process_cwz():
    ensure_data_dirs()

    RAW_CWZ_PATH = os.path.join(DATA_RAW_PATH, 'data_steven_anon.csv')
    RAW_OVERVIEW_PATH = os.path.join(DATA_RAW_PATH, 'overview_20210329.csv')

    RAW_CWZ_VIS_PATH = os.path.join(DATA_RAW_PATH, 'cwz_preds (5).csv')
    RAW_VIS_TRAIN = os.path.join(DATA_RAW_PATH,
                                 "predictions__train_changed['test_csv_version', 'copy_data_to_local']to['12_05_2020_2', True]_features.p.csv")
    RAW_VIS_VAL = os.path.join(DATA_RAW_PATH,
                               "predictions__val_changed['test_csv_version', 'copy_data_to_local']to['12_05_2020_2', True]_features.p.csv")
    RAW_VIS_TEST = os.path.join(DATA_RAW_PATH,
                                "predictions__test_changed['test_csv_version', 'copy_data_to_local']to['12_05_2020_2', True]_features.p.csv")

    df = pd.read_csv(RAW_CWZ_PATH)


    def load_visual_features(path, cwz):
        visual_features = pd.read_csv(path)
        visual_features.features = visual_features.features.str.replace(r'[^\[] +', ', ', regex=True).apply(
            ast.literal_eval)
        vf = visual_features.features.apply(pd.Series)
        vf.columns = vf.columns.map('vis_feature_{}'.format)
        visual_features = pd.concat([visual_features[['scan_id']], vf], axis=1)
        if cwz:
            visual_features.set_index(['scan_id'], inplace=True)
        else:
            visual_features['patientprimarymrn'] = pd.to_numeric(
                visual_features.scan_id.str.extract(r'(\d+)_st\d+').iloc[:, 0])
            visual_features['study'] = visual_features.scan_id.str.extract(r'\d+_(st\d+)').iloc[:, 0]
            visual_features.drop(columns=['scan_id'], inplace=True)
        return visual_features

    visual_cwz = load_visual_features(RAW_CWZ_VIS_PATH, True)

    visual_rumc_train = load_visual_features(RAW_VIS_TRAIN, False)
    visual_rumc_val = load_visual_features(RAW_VIS_VAL, False)
    visual_rumc_test = load_visual_features(RAW_VIS_TEST, False)
    visual_rumc = pd.concat([visual_rumc_train, visual_rumc_val, visual_rumc_test])

    df['scan_id'] = pd.to_numeric(df.record_id.str.split('-', expand=True).iloc[:, 2])
    df.loc[df.scan_id.isna(), 'scan_id'] = pd.to_numeric(df[df.scan_id.isna()].record_id)
    visual_rumc['scan_id'] = visual_rumc.patientprimarymrn.astype(str)
    visual = pd.concat([visual_cwz.reset_index(), visual_rumc])
    visual.scan_id = pd.to_numeric(visual.scan_id)

    df_overview = pd.read_csv(RAW_OVERVIEW_PATH)
    df_overview.StudyDate = pd.to_datetime(df_overview.StudyDate, format='%Y%m%d')
    df_overview['StudyId'] = df_overview.StudyPath.str.extract(r'\d+/(.+)')
    df_overview['PatientID'] = df_overview.StudyPath.str.extract(r'(\d+)/.+')

    study_dates = df_overview.groupby(['PatientID', 'StudyId']).StudyDate.mean(numeric_only=False)

    df['study'] = np.nan
    df.diagn_CT_date = pd.to_datetime(df.diagn_CT_date)
    for (patient, study), date in study_dates.iteritems():
        df.loc[(df.record_id == patient) &
               ((date - df.diagn_CT_date) <= pd.Timedelta(days=2)) &
               ((date - df.diagn_CT_date) >= pd.Timedelta(days=-2)), 'study'] = study

    df = df.merge(visual, left_on=['scan_id', 'study'], right_on=['scan_id', 'study'], how='left')
    df.patientprimarymrn = df.record_id
    df = df.set_index(['patientprimarymrn', 'study'])

    visual_features = df.columns[df.columns.str.startswith('vis')]
    clinical_features = (['dem_pat_age_at_inclusion', 'dem_pat_gender', 'dem_pat_BMI']
                         + list(df.columns[df.columns.str.startswith('med_hist')].values)
                         + list(df.columns[df.columns.str.startswith('adm') & ~df.columns.isin(
                ['adm_date', 'adm_dept', 'adm_hosp_date'])].values))
    target_features = ['COVID_and_prob_and_pos', 'COVID_and_prob', 'COVID_diagnosis']

    df = df[list(visual_features) + clinical_features + target_features]

    def get_nested_col(col):
        if col in clinical_features:
            return 'Input', 'Clinical', col
        elif col in visual_features:
            return 'Input', 'Visual', col
        elif col in target_features:
            return 'Target', col, ''
        else:
            raise ValueError(f"Can't handle {col}")

    new_cols = [get_nested_col(col) for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(new_cols)

    df = df.rename(mapper={'COVID_and_prob_and_pos': 'diagnosis'}, level=1, axis='columns')

    cwz_indices = df.index.get_level_values(0).str.contains('CWZ')
    df.loc[~cwz_indices, ('Meta', 'hospital', '')] = 'rumc'
    df.loc[cwz_indices, ('Meta', 'hospital', '')] = 'cwz'

    for col in df.Input.Clinical.select_dtypes(include=['int64', 'int32']):
        if col == 'dem_pat_age_at_inclusion':
            continue

        df['Input', 'Clinical', col] = df['Input', 'Clinical', col].astype('category')

    # Save data
    df.to_pickle(os.path.join(DATA_PROCESSED_PATH, 'cwz.pkl'))
