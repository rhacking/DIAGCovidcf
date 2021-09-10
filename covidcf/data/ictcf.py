import os
import pickle
import re
from typing import Tuple
import ast

import click
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from .base import ensure_data_dirs, DATA_RAW_PATH, DATA_PROCESSED_PATH


def load_ictcf() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_pickle(os.path.join(DATA_PROCESSED_PATH, 'ictcf.pkl'))
    return df, df[['Input']], df.Target


@click.command()
def process_ictcf():
    ensure_data_dirs()
    RAW_ICTCF_PATH = os.path.join(DATA_RAW_PATH, 'patient.txt')

    RAW_VIS_TRAIN = os.path.join(DATA_RAW_PATH, 'predictions__train_ens_features.p.csv')
    RAW_VIS_VAL = os.path.join(DATA_RAW_PATH, 'predictions__val_ens_features.p.csv')
    RAW_VIS_TEST = os.path.join(DATA_RAW_PATH, 'predictions__test_ens_features2.p.csv')

    if not os.path.exists(RAW_ICTCF_PATH):
        print('patient.txt missing, downloading...')
        res = requests.get("http://ictcf.biocuckoo.cn/patient/patient.txt")
        text = res.text
        with open(RAW_ICTCF_PATH, 'wb') as f:
            f.write(res.content)
    else:
        with open(RAW_ICTCF_PATH, 'r', encoding='utf-8') as f:
            text = f.read()

    raw_data = [row.split("\t") for row in text.split("\n")]

    patients = {}
    print('Processing patient data...')
    for i in tqdm(range(len(raw_data) - 1)):
        features = get_patient_features(get_patient(raw_data, i))
        patients[i] = features

    df = pd.concat(patients)
    df["Patient"] = df.index.get_level_values(0)

    ranges = pd.concat([df.groupby('Abbreviation').NormalMin.mean(), df.groupby('Abbreviation').NormalMax.mean()],
                       axis=1)
    ranges.index = 'Value_' + ranges.index

    # Normalize values
    df.Value = (df.Value - df.NormalMin) / (df.NormalMax - df.NormalMin)

    mapping = {}
    for name in df.Name.unique():
        mapping[f'Value_{df[df.Name == name].Abbreviation.values[0]}'] = f'{name} Value'
        mapping[f'Change_{df[df.Name == name].Abbreviation.values[0]}'] = f'{name} Change'

    df = df.pivot(index="Patient", columns="Abbreviation", values=["Value", "Change"])
    patient_meta = []
    print('Processing patient metadata...')
    for patient in tqdm(range(len(raw_data) - 1)):
        patient_meta.append(raw_data[patient][-1].split(r"&basic_info&&ct&")[0].split("$"))
        patient_meta[-1][0] = patient_meta[-1][0].split('&')[2]

    print('Processing data...')

    meta_df = pd.DataFrame(patient_meta)
    metadata_names = ["Hospital", "Age", "Sex", "Temperature", "Diseases", "PCR", "CT", "Morbidity", "Mortality"]
    meta_df.columns = pd.MultiIndex.from_product([["Metadata"], metadata_names])
    df = df.join(meta_df, how="right")
    df['Metadata', "PCR"] = df['Metadata', "PCR"].str.replace("Negative; Positive (Confirmed later)", "Positive",
                                                              regex=False)

    df.columns = df.columns.to_flat_index()
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]

    diseases = [d.strip().lower() for d in df.Metadata_Diseases[df.Metadata_Diseases != "No"].str.split(",").sum()]
    disease_counts = pd.Series(diseases).value_counts()

    for disease in disease_counts[disease_counts >= 50].index.values:
        df[f'Disease_{disease}'] = df.Metadata_Diseases.str.lower().str.contains(disease)

    df['Disease'] = df.Metadata_Diseases != 'No'

    ranges.to_pickle(os.path.join(DATA_PROCESSED_PATH, 'ranges.pkl'))
    with open(os.path.join(DATA_PROCESSED_PATH, 'mapping.pkl'), 'wb') as f:
        pickle.dump(mapping, f)

    def load_visual_features(path):
        visual_features = pd.read_csv(path)
        visual_features.features = visual_features.features.str.replace(r'[^\[] +', ', ', regex=True).apply(
            ast.literal_eval)
        vf = visual_features.features.apply(pd.Series)
        vf.columns = vf.columns.map('vis_feature_{}'.format)
        visual_features = pd.concat([visual_features[['scan_id']], vf], axis=1)
        visual_features = visual_features.rename(columns={'scan_id': 'patient_id'})
        visual_features.patient_id = pd.to_numeric(visual_features.patient_id.str.extract('Patient (\d+)').iloc[:, 0])
        return visual_features

    vis_train = load_visual_features(RAW_VIS_TRAIN)
    vis_val = load_visual_features(RAW_VIS_VAL)
    vis_test = load_visual_features(RAW_VIS_TEST)

    visual_features = pd.concat([vis_train, vis_val, vis_test], ignore_index=True)
    visual_features['patient_id'] = visual_features['patient_id'].astype(np.int)
    df['patient_id'] = df.index + 1

    df = df.merge(visual_features, how='left', on='patient_id')
    df.drop(columns=['patient_id', 'Metadata_Diseases'], inplace=True, errors='ignore')
    df.drop(columns=df.columns[df.columns.str.startswith('Change_')], inplace=True)

    categorical_features = ["Metadata_Sex"] + [col for col in df.columns if col.startswith("Disease")]
    numeric_features = [col for col in df.columns if col.startswith("Value_")] + ['Metadata_Age',
                                                                                  'Metadata_Temperature']
    non_visual_features = categorical_features + numeric_features
    vis_features = df.columns[df.columns.str.startswith('vis_')]
    target_features = ['Metadata_PCR', 'Metadata_Mortality', 'Metadata_Morbidity', 'Metadata_CT']
    meta_features = ['Metadata_Hospital']

    for cat_col in categorical_features + target_features:
        df[cat_col] = pd.Categorical(df[cat_col])

    for non_cat_col in numeric_features:
        df[non_cat_col] = pd.to_numeric(df[non_cat_col], errors='coerce')

    def get_nested_col(col):
        if col in non_visual_features:
            return 'Input', 'Clinical', col
        elif col in vis_features:
            return 'Input', 'Visual', col
        elif col in target_features:
            return 'Target', col, ''
        elif col in meta_features:
            return 'Meta', col, ''
        else:
            raise ValueError(f"Can't handle {col}")

    new_cols = [get_nested_col(col) for col in df.columns]
    df.columns = pd.MultiIndex.from_tuples(new_cols)

    df = df.rename(mapper={'Metadata_PCR': 'pcr'}, level=1, axis='columns')
    df = df.rename(mapper={'Metadata_Hospital': 'hospital'}, level=1, axis='columns')
    df['Meta', 'cohort', ''] = ''
    df.loc[df.index < 1170, ('Meta', 'cohort', '')] = 'c1'
    df.loc[df.index >= 1170, ('Meta', 'cohort', '')] = 'c2'

    df.drop(df[df.Target.Metadata_Morbidity == 'Suspected'].index, inplace=True)

    df.to_pickle(os.path.join(DATA_PROCESSED_PATH, 'ictcf.pkl'))


def get_patient(raw_data, pid: int):
    return raw_data[pid]


def get_patient_features(patient_data):
    DOWN_ARROW = "#8595;"
    UP_ARROW = "#8593;"
    split_tables = patient_data[-1]
    basic_info = split_tables.split("Basic Information")[1]
    found = re.findall(
        r"\$(?P<name>[^\$]+)@(?P<abbr>[^\$]+)@(?P<value>[^\$&]+)&?(?P<change>[^\$]+)?@(?P<normal>[^\$]+)",
        basic_info)

    df = pd.DataFrame(found, columns=["Name", "Abbreviation", "Value", "Change", "Normal"])
    # TODO: Check if UP and DOWN arrow ar correct
    df.Change = df.Change.str.replace(DOWN_ARROW, "-1").replace(UP_ARROW, "1").replace("", "0")
    pd.Change = pd.to_numeric(df.Change)
    df.Normal = df.Normal.str.replace("<", "0-")

    # !!!!
    df.Value = df.Value.str.replace("<", "")
    df.Value = df.Value.str.replace(">", "")
    df[["NormalMin", "NormalMax"]] = df.Normal.str.extract(r"([\d\.]+)\-([\d\.]+).+")
    df.drop(columns=['Normal'], inplace=True)

    df.Value = pd.to_numeric(df.Value)
    df.NormalMin = pd.to_numeric(df.NormalMin)
    df.NormalMax = pd.to_numeric(df.NormalMax)

    return df


# def get_ictcf_features(df: pd.DataFrame) -> Features:
#     categorical_features = ["Metadata_Sex"] + [col for col in df.columns if col.startswith("Disease_")]
#     numeric_features = [col for col in df.columns if col.startswith("Value_")] + ['Metadata_Age',
#                                                                                   'Metadata_Temperature']
#     output_feature = "Metadata_PCR"
#
#     return Features(categorical_features, numeric_features, output_feature)
