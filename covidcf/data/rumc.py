import os
from typing import Tuple

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

from covidcf.config.dataset import DatasetTarget
from covidcf.data.base import DATA_RAW_PATH, DATA_PROCESSED_PATH, load_visual_features


def load_rumc(target: DatasetTarget) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_pickle(os.path.join(DATA_PROCESSED_PATH, 'rumc.pkl'))
    return df, df[['Input']], df.Target[target]


@click.command()
def process_rumc():
    # Declare paths
    RAW_RUMC_PATH = os.path.join(DATA_RAW_PATH, 'covid_ct_CF_anon_with_date_shifts.csv')
    RAW_OVERVIEW_PATH = os.path.join(DATA_RAW_PATH, 'overview_20210329.csv')
    RAW_DIAGNOSIS_PATH = os.path.join(DATA_RAW_PATH, 'covid_ct_diagnosis_anon.csv')
    SPLIT_PATH = os.path.join(DATA_PROCESSED_PATH, 'split.json')

    RAW_VIS_TRAIN = os.path.join(DATA_RAW_PATH,
                                 "predictions__train_changed['test_csv_version', 'copy_data_to_local']to['12_05_2020_2', True]_features.p.csv")
    RAW_VIS_VAL = os.path.join(DATA_RAW_PATH,
                               "predictions__val_changed['test_csv_version', 'copy_data_to_local']to['12_05_2020_2', True]_features.p.csv")
    RAW_VIS_TEST = os.path.join(DATA_RAW_PATH,
                                "predictions__test_changed['test_csv_version', 'copy_data_to_local']to['12_05_2020_2', True]_features.p.csv")

    # Load raw data
    df_rumc = pd.read_csv(RAW_RUMC_PATH, index_col=0)
    df_overview = pd.read_csv(RAW_OVERVIEW_PATH)
    df_diagnosis = pd.read_csv(RAW_DIAGNOSIS_PATH)

    # Deel with <, >, <=, >= values
    indices = df_rumc.labresultsvalue.str.match(r'[<>]=?(\d*\.?\d*)').fillna(False)
    df_rumc.loc[indices, 'labresultsvalue'] = df_rumc[indices].labresultsvalue.str.extract(r'[<>]=?(\d*\.?\d*)').values

    # Remove + from <number>+
    indices = df_rumc.labresultsvalue.str.match(r'\d+\+').fillna(False)
    df_rumc.loc[indices, 'labresultsvalue'] = df_rumc[indices].labresultsvalue.str.extract(r'(\d+)\+').values

    # Change 'Positief' and 'Negatief' to 1 and 0 repsectively
    # df_rumc.loc[df_rumc.labresultsvalue == 'Positief', 'labresultsvalue'] = 1
    # df_rumc.loc[df_rumc.labresultsvalue == 'Negatief', 'labresultsvalue'] = 0

    # Change sequence of plusses to length of string
    indices = df_rumc.labresultsvalue.str.match(r'\++$').fillna(False)
    df_rumc.loc[indices, 'labresultsvalue'] = df_rumc[indices].labresultsvalue.str.len()

    # Change sequence of minusses to length of string
    indices = df_rumc.labresultsvalue.str.match(r'\-+$').fillna(False)
    df_rumc.loc[indices, 'labresultsvalue'] = -df_rumc[indices].labresultsvalue.str.len()

    # Remove 'Afname' variables
    df_rumc.drop(df_rumc[df_rumc.labresultsname == 'Afname'].index, inplace=True)

    # Deal with log values
    indices = df_rumc.labresultsvalue.str.match(r'log\([<>]\d+\)').fillna(False)
    log_args = df_rumc[indices].labresultsvalue.str.extract(r'log\((.+?)\)').iloc[:, 0]
    df_rumc.loc[indices, 'labresultsvalue'] = np.log(log_args.str.replace('<', '').str.replace('>', '').astype(int))

    # Deal with log ranges
    indices = df_rumc.labresultsvalue.str.match(r'log\(\d+\-\d+\)').fillna(False)
    log_args = df_rumc[indices].labresultsvalue.str.extract(r'log\((\d+)\-(\d+)\)')
    df_rumc.loc[indices, 'labresultsvalue'] = np.log((log_args[0].astype(int) + log_args[1].astype(int)) / 2)

    string_value_counts = df_rumc.labresultsvalue[
        pd.to_numeric(df_rumc.labresultsvalue, errors='coerce').isnull()].value_counts()

    # Remove missing values
    indices = df_rumc.labresultsvalue.str.lower().isin(
        ['niet ontvangen', 'niet te bepalen', 'geannuleerd', 'mislukt', 'vervalt', 'foutief geprikt',
         'te weinig materiaal', 'zie bijlage', 'zie opmerking in het bijlage rapport', 'ontvangen', 'vervalt',
         'handdif_uitgevoerd', 'foutief aangemeld', 'zie toelichting', 'dubieus', 'niet aantoonbaar',
         'niet te berekenen', 'gestold', 'hemolytisch', 'nabepaling niet honoreerbaar'])
    # Remove string values that occur less than 12 times
    indices |= df_rumc.labresultsvalue.isin(string_value_counts[string_value_counts < 12].index.values)
    # Remove metadata values
    indices |= df_rumc.labresultsname.isin(
        ['Status', 'Status 2', 'Afname', 'Product code', 'Product code 2', 'EIN nr.', 'EIN nr. 2', 'Expiratie datum'])
    # Remove whitespace
    indices |= df_rumc.labresultsvalue.str.match(r'\s*$').fillna(False)
    # Remove N codes
    indices |= df_rumc.labresultsvalue.str.match(r'N\d+$').fillna(False)
    # Remove pathologists
    indices |= df_rumc.labresultsvalue.str.startswith('Patholoog').fillna(False)
    # Remove date values
    indices |= df_rumc.labresultsvalue.str.match(r'\d+/\d+/\d+$').fillna(False)
    indices |= df_rumc.labresultsvalue.str.match(r'\d+\-\d+\-\d+$').fillna(False)

    # Set all these to nan
    df_rumc.loc[indices, 'labresultsvalue'] = np.nan

    # Set string null values to 0
    df_rumc.loc[df_rumc.labresultsvalue.isin(['Normaal', 'geen']), 'labresultsvalue'] = 0

    # Parse study dates and extract study id
    df_overview.StudyDate = pd.to_datetime(df_overview.StudyDate, format='%Y%m%d')
    df_overview['StudyId'] = df_overview.StudyPath.str.extract(r'\d+/(.+)')
    df_overview['PatientID'] = df_overview.StudyPath.str.extract(r'(\d+)/.+').astype(np.int64)

    # Compute study dates
    study_dates = df_overview.groupby(['PatientID', 'StudyId']).StudyDate.mean(numeric_only=False)

    # Parse lab value dates
    df_rumc.labresultsresultdate = pd.to_datetime(df_rumc.labresultsresultdate)

    # Determine which study each lab value corresponds to
    # df_rumc['study'] = np.nan
    # is_pcr = df_rumc.labresultscommonname.str.lower().str.contains(
    #     'cov') & df_rumc.labresultscommonname.str.lower().str.contains('pcr')
    # df_rumc['date_diff'] = np.inf
    # for (patient, study), date in tqdm(study_dates.iteritems()):
    #     is_patient = (df_rumc.patientprimarymrn == patient)
    #     date_diff = (date - df_rumc[is_patient].labresultsresultdate)
    #     non_pcr_range = ((date_diff <= pd.Timedelta(days=3)) &
    #                      (date_diff >= pd.Timedelta(days=-3)))
    #     pcr_range = ((date_diff <= pd.Timedelta(days=14)) &
    #                  (date_diff >= pd.Timedelta(days=-14)))
    #
    #     # new_diffs = date_diff / pd.to_timedelta(1, unit='D')
    #     new_diffs = date_diff.dt.total_seconds()
    #     is_closer = (new_diffs.abs() < df_rumc.loc[is_patient, 'date_diff'].abs())
    #     df_rumc.loc[is_closer & is_patient & (
    #             (is_pcr & pcr_range) | (~is_pcr | non_pcr_range)), 'study'] = study

    df_rumc['study'] = np.nan
    is_pcr = df_rumc.labresultscommonname.str.lower().str.contains(
        'cov') & df_rumc.labresultscommonname.str.lower().str.contains('pcr')
    for (patient, study), date in tqdm(list(study_dates.iteritems())):
        date_diff = (date - df_rumc.labresultsresultdate)
        non_pcr_range = ((date_diff <= pd.Timedelta(days=4)) &
                         (date_diff >= pd.Timedelta(days=-4)))
        pcr_range = ((date_diff <= pd.Timedelta(days=14)) &
                     (date_diff >= pd.Timedelta(days=-14)))

        df_rumc.loc[(df_rumc.patientprimarymrn == patient) & (
                (is_pcr & pcr_range) | (~is_pcr | non_pcr_range)), 'study'] = study

    # For lab values that do not correspond directly to a single study, add them for each study
    # studies_per_patient = study_dates.index.to_frame()
    # studies_per_patient.rename(columns={'PatientID': 'patientprimarymrn', 'StudyId': 'study'}, inplace=True)
    # df_rumc_no_study = df_rumc[df_rumc.study.isna()].copy()
    # df_rumc_no_study.drop(columns=['study'], inplace=True)
    # df_rumc_no_study_repeated = df_rumc_no_study.merge(studies_per_patient, on='patientprimarymrn')
    #
    # # Merge lab values that correspond directly to a study and those that don't
    # df_rumc_full = pd.concat([df_rumc[~df_rumc.study.isna()], df_rumc_no_study_repeated], ignore_index=True)

    # Bring table into wide form
    df_pivot = pd.pivot_table(df_rumc[~df_rumc.study.isna()], index=['patientprimarymrn', 'study'],
                              columns=['labresultscommonname'],
                              values='labresultsvalue', aggfunc=lambda x: x.iloc[0])

    # Load visual features
    vis_train = load_visual_features(RAW_VIS_TRAIN)
    vis_val = load_visual_features(RAW_VIS_VAL)
    vis_test = load_visual_features(RAW_VIS_TEST)

    # Concatenate visual features
    visual_features = pd.concat([vis_train, vis_val, vis_test])

    import re
    cat_in_num_vals = [
        (col,
         [x for x in df_pivot[col].value_counts().index.values if re.match(r'[+-]?([0-9]*[.])?[0-9]+', str(x)) is None])
        for col in df_pivot.select_dtypes(include='category').iloc[:, np.where(
            [df_pivot[col].str.match(r'[+-]?([0-9]*[.])?[0-9]+').any() for col in
             df_pivot.select_dtypes(include='category').columns])[0]].columns]

    assert len(cat_in_num_vals) == 0

    # Join lab and visual features
    df_pivot = df_pivot.join(visual_features, how='outer')

    # Add sex and age
    df_pivot_reindexed = df_pivot.copy()
    df_pivot_reindexed['study'] = df_pivot_reindexed.index.get_level_values(1)
    df_pivot_reindexed.index = df_pivot.index.get_level_values(0)
    df_pivot_reindexed['age'] = df_rumc.groupby(['patientprimarymrn']).patientage.mean()
    df_pivot_reindexed['sex'] = df_rumc.groupby(['patientprimarymrn']).patientsex.first()
    df_pivot = df_pivot_reindexed.set_index(['study'], append=True)

    # Compress PCR to single column
    pcr = df_pivot.loc[:,
          df_pivot.columns.str.lower().str.contains('cov') & df_pivot.columns.str.lower().str.contains('pcr')].apply(
        lambda x: x[x.first_valid_index()] if x.first_valid_index() is not None else np.nan, axis=1)
    df_pivot.drop(columns=df_pivot.columns[
        df_pivot.columns.str.lower().str.contains('cov') & (
                df_pivot.columns.str.lower().str.contains('pcr') | df_pivot.columns.str.lower().str.contains(
            'ig'))],
                  inplace=True)
    df_pivot['pcr'] = pcr

    # Load split data
    import json
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    # Convert to dataframes
    train_split = pd.DataFrame(split['train'])
    train_split['part'] = 'train'
    val_split = pd.DataFrame(split['val'])
    val_split['part'] = 'val'
    test_split = pd.DataFrame(split['test'])
    test_split['part'] = 'test'

    # Join to other data
    df_split = pd.concat([train_split, val_split, test_split])
    df_split.rename(columns={'patientid': 'patientprimarymrn', 'y': 'corads'}, inplace=True)
    df_split.drop(columns=['x'], inplace=True)
    df_split.set_index(['patientprimarymrn', 'study'], inplace=True)

    df_pivot = df_pivot.join(df_split)

    # Process diagnosis data
    df_diagnosis.diagnosisstartdate = pd.to_datetime(df_diagnosis.diagnosisstartdate)
    df_diagnosis['study'] = np.nan
    for (patient, study), date in study_dates.iteritems():
        df_diagnosis.loc[(df_diagnosis.patientprimarymrn == patient) &
                         ((date - df_diagnosis.diagnosisstartdate) <= pd.Timedelta(days=14)) &
                         ((date - df_diagnosis.diagnosisstartdate) >= pd.Timedelta(days=-14)), 'study'] = study

    df_diagnosis.drop(df_diagnosis[df_diagnosis.study.isna()].index, inplace=True)
    df_diagnosis.set_index(['patientprimarymrn', 'study'], inplace=True)
    s_diagnosis = df_diagnosis[['diagnosishospitaldiagnosis', 'diagnosisemergencydepartmentdiagnosis']].mean(axis=1)
    s_diagnosis = s_diagnosis[~s_diagnosis.index.duplicated(keep='first')]
    df_pivot['diagnosis'] = s_diagnosis

    non_visual_features = list(df_pivot.columns[~df_pivot.columns.isin(
        ['pcr', 'corads', 'diagnosis', 'part']) & ~df_pivot.columns.str.startswith('vis_')])
    visual_features = list(df_pivot.columns[df_pivot.columns.str.startswith('vis_')])
    target_features = ['pcr', 'corads', 'diagnosis']
    meta_features = ['part']

    def get_nested_col(col):
        if col in non_visual_features:
            return 'Input', 'Clinical', col
        elif col in visual_features:
            return 'Input', 'Visual', col
        elif col in target_features:
            return 'Target', col, ''
        elif col in meta_features:
            return 'Meta', col, ''

    new_cols = [get_nested_col(col) for col in df_pivot.columns]
    df_pivot.columns = pd.MultiIndex.from_tuples(new_cols)

    # Fix dtypes
    for col in df_pivot.columns:
        try:
            df_pivot[col] = pd.to_numeric(df_pivot[col], errors='raise')
        except ValueError:
            df_pivot[col] = df_pivot[col].astype('category')
            # num_cats = df_pivot[col].cat.categories[df_pivot[col].cat.categories.str.match(r'\d+', na=False)]
            # df_pivot[col].cat.rename_categories({cat: 'v' + cat for cat in num_cats}, inplace=True)
            # df_pivot[col].cat.add_categories(['Missing'], inplace=True)
            # df_pivot[col].fillna('Missing', inplace=True)

    # Save data
    df_pivot.to_pickle(os.path.join(DATA_PROCESSED_PATH, 'rumc.pkl'))
