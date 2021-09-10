import inspect
import os
import traceback
from pathlib import Path

import click
import joblib
import shap
import tikzplotlib
import yaml
import pandas as pd
import seaborn as sns
from marshmallow import ValidationError
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

from config.meta import load_meta
from .base import get_cat_codes
from .metrics import get_metric_callables_from_dict
from .models import TestTimeImputingClassifier
from ..config import ConfigSchema, HyperOptConfigSchema
from .evaluation import evaluate
from .hyperopt import hyperopt


def gen_latex_table(results, index):
    df_pivot = pd.pivot(results, values=['value', 'lb', 'ub'], index=index, columns='metric')
    df_output = pd.pivot(results, values='value', index=index, columns='metric')

    for i, metric in enumerate(df_output.columns):
        df_output.insert(2 * i + 1, metric + '_bar', 0)
        for model in df_output[metric].index:
            df_output.loc[
                model, metric + '_bar'] = f'bar({df_pivot.value.loc[model, metric]:.3f}, {df_pivot.lb.loc[model, metric]:.3f}, {df_pivot.ub.loc[model, metric]:.3f})'

    for col in df_output.columns:
        if '_bar' in col:
            continue
        is_max = df_output[col] == df_output[col].max()
        df_output.loc[is_max, col] = f'\\textbf{{{df_output[col].max():.3f}}}'
        df_output.loc[~is_max, col] = [f'{x:.3f}' for x in df_output.loc[~is_max, col]]

    df_output.index.names = [n[0].upper() + n[1:].replace('_', ' ') for n in df_output.index.names]
    df_output.columns.name = df_output.columns.name[0].upper() + df_output.columns.name[1:].replace('_', ' ')

    latex = (df_output.rename(columns={col: '' for col in df_output.columns if '_bar' in col})
             .to_latex(escape=False))

    import re
    for match in re.finditer(r'bar\(((?:\d+\.\d+|nan)), ((?:\d+\.\d+|nan)), ((?:\d+\.\d+|nan))\)', latex):
        value = float(match.group(1))
        if np.isnan(value):
            latex = latex.replace(match.group(0), '')
            continue
        lb = float(match.group(2))
        if np.isnan(lb):
            lb = value
        ub = float(match.group(3))
        if np.isnan(ub):
            ub = value

        scale_factor = 1.15
        y1 = 0.06
        y2 = 0.352
        yh = (y1 + y2) / 2
        yd = 0.04
        yb1 = yh - yd
        yb2 = yh + yd

        latex = latex.replace(match.group(0), r'''\begin{minipage}[c]{1.2cm}
       \begin{tikzpicture}
        \draw (0cm,0cm) (5,0);
        \draw[barchart] (0,''' + str(y1) + ''') rectangle (''' + str(value * scale_factor) + ''',''' + str(y2) + ''');
        \draw[errorbar] (''' + str(lb * scale_factor) + ''',''' + str(yh) + ''') -- (''' + str(
            ub * scale_factor) + ''',''' + str(yh) + ''');
        \draw[errorbar] (''' + str(lb * scale_factor) + ''',''' + str(yb2) + ''') -- (''' + str(
            lb * scale_factor) + ''',''' + str(yb1) + ''');
        \draw[errorbar] (''' + str(ub * scale_factor) + ''',''' + str(yb2) + ''') -- (''' + str(
            ub * scale_factor) + ''',''' + str(yb1) + ''');
       \end{tikzpicture}
      \end{minipage}
        ''')

    #     \draw[scale] (0,0) node[] {};
    
    return latex


def run_experiments(experiments_path: str, rerun_hyperopt: bool = False, rerun_evaluation: bool = False,
                    rerun_results: bool = False):
    # FIXME: Ensure test split is consistent!!!
    if len([f.name for f in list(os.scandir(experiments_path)) if f.name.lower() == 'meta.yaml']) == 0:
        for f in os.scandir(experiments_path):
            if os.path.isdir(f.path):
                run_experiments(f.path, rerun_hyperopt=rerun_hyperopt, rerun_evaluation=rerun_evaluation,
                                rerun_results=rerun_results)
    else:
        # if '/between_dataset\\' in str(experiments_path) or \
        #         '/between_dataset_hyperopt\\' in str(experiments_path) or \
        #         '/impute_predict_missing\\' in str(experiments_path) or \
        #         '/predict_with_impute\\' in str(experiments_path) or \
        #         '/shap_zeroing\\' in str(experiments_path) or \
        #         '/shap_zeroing_hyperopt\\' in str(experiments_path):
        #     print('skipping', str(experiments_path))
        #     return

        print('doing', str(experiments_path))
        # return

        meta = load_meta(os.path.join(experiments_path, 'meta.yaml'))

        for f in os.scandir(experiments_path):
            if f.name != 'meta.yaml' and f.name.endswith('.yaml'):
                with open(f.path, 'r') as fh:
                    config = yaml.load(fh, Loader=yaml.SafeLoader)
                try:
                    ConfigSchema.load(config)
                except:
                    path_to_file, filename = os.path.split(f.path)
                    path_to_pkl = os.path.join(path_to_file, f'{filename.split(".")[0]}.pkl')
                    if not os.path.exists(path_to_pkl) or rerun_hyperopt:
                        config = HyperOptConfigSchema.load(config)
                        hyperopt(config, f.path)

        for f in os.scandir(experiments_path):
            if f.name != 'meta.yaml' and not f.name.startswith('result_') and f.name.endswith('.pkl'):
                study = joblib.load(f.path)
                path_to_file, filename = os.path.split(f.path)
                if os.path.exists(path_to_file + '/' + 'result_' + filename) and not rerun_evaluation:
                    continue
                config_path = os.path.join(path_to_file, f'{filename.split(".")[0]}.yaml')
                with open(config_path, 'r') as fh:
                    config_dict = yaml.load(fh, Loader=yaml.SafeLoader)

                for k, v in study.best_params.items():
                    keys = k.split('__')
                    current = config_dict['parameters']
                    for key in keys[:-1]:
                        current = current[key]
                    current[keys[-1]] = v
                del config_dict['n_trials']
                del config_dict['dataset']['validation']

                if 'n_bagging' in config_dict['pipeline'] and config_dict['pipeline']['n_bagging'] is not None:
                    config_dict['pipeline']['n_bagging'] = 500 if 'test_time_impute' not in experiments_path else 40

                if 'estimator' in config_dict['parameters'] and 'boosting' in config_dict['parameters']['estimator'] and \
                        config_dict['parameters']['estimator']['boosting'] == 'rf':
                    config_dict['parameters']['estimator']['n_estimators'] = 500 if 'test_time_impute' not in experiments_path else 40

                if config_dict['pipeline']['estimator_type'] == 'RF':
                    if 'n_estimators' in config_dict['parameters']['estimator']:
                        config_dict['parameters']['estimator']['n_estimators'] = 500 if 'test_time_impute' not in experiments_path else 40
                    elif 'base_estimator' in config_dict['parameters']['estimator'] and 'n_estimators' in config_dict['parameters']['estimator']['base_estimator']:
                        config_dict['parameters']['estimator']['base_estimator']['n_estimators'] = 500 if 'test_time_impute' not in experiments_path else 40
                    else:
                        raise ValueError("Can't increase n_estimators")

                config = ConfigSchema.load(config_dict)
                evaluate(config, config_dict, save_path=path_to_file + '/' + 'result_' + filename,
                         name=filename.split('.')[0], save_pipeline=meta.plot_shap, adjust_thresholds=meta.adjust_thresholds)
            elif f.name != 'meta.yaml' and f.name.endswith('.yaml'):
                try:
                    path_to_file, filename = os.path.split(f.path)
                    if os.path.exists(
                            path_to_file + '/' + 'result_' + filename.split('.')[0] + '.pkl') and not rerun_evaluation:
                        continue
                    with open(f.path, 'r') as fh:
                        config_dict = yaml.load(fh, Loader=yaml.SafeLoader)

                    if 'n_bagging' in config_dict['pipeline'] and config_dict['pipeline']['n_bagging'] is not None:
                        config_dict['pipeline']['n_bagging'] = 500 if 'test_time_impute' not in experiments_path else 40

                    if 'estimator' in config_dict['parameters'] and 'boosting' in config_dict['parameters'][
                        'estimator'] and \
                            config_dict['parameters']['estimator']['boosting'] == 'rf':
                        config_dict['parameters']['estimator'][
                            'n_estimators'] = 500 if 'test_time_impute' not in experiments_path else 40

                    if config_dict['pipeline']['estimator_type'] == 'RF':
                        if 'n_estimators' in config_dict['parameters']['estimator']:
                            config_dict['parameters']['estimator'][
                                'n_estimators'] = 500 if 'test_time_impute' not in experiments_path else 40
                        elif 'base_estimator' in config_dict['parameters']['estimator'] and 'n_estimators' in \
                                config_dict['parameters']['estimator']['base_estimator']:
                            config_dict['parameters']['estimator']['base_estimator'][
                                'n_estimators'] = 500 if 'test_time_impute' not in experiments_path else 40
                        else:
                            raise ValueError("Can't increase n_estimators")

                    config = ConfigSchema.load(config_dict)
                    print(config.parameters)

                    evaluate(config, config_dict,
                             save_path=path_to_file + '/' + 'result_' + filename.split('.')[0] + '.pkl',
                             name=filename.split('.')[0], save_pipeline=meta.plot_shap, adjust_thresholds=meta.adjust_thresholds)
                except ValidationError as e:
                    if e.messages != {'dataset': {'validation': ['Unknown field.']}, 'n_trials': ['Unknown field.']}:
                        traceback.print_exc()
                        raise

        if Path(os.path.join(experiments_path, 'results')).exists() and not rerun_results:
            return

        dfs = []
        # dfs_val = []
        shap_df = []
        is_multi = False
        for f in os.scandir(experiments_path):
            if f.name != 'meta.yaml' and f.name.startswith('result_'):
                print(f.name)
                path_to_file, filename = os.path.split(f.path)
                config_path = os.path.join(path_to_file, f'{filename.split(".")[0].replace("result_", "")}.yaml')
                with open(config_path, 'r') as fh:
                    config_dict = yaml.load(fh, Loader=yaml.SafeLoader)
                data = joblib.load(f.path)
                print(len(data['preds_proba']))
                print(data.keys())
                if not isinstance(data['preds_proba'], list):
                    if meta.plot_shap:
                        try:
                            cols_to_use = None
                            if isinstance(data['pipeline'], BaggingClassifier):
                                if isinstance(data['pipeline'].estimators_[0].steps[-1][1], TestTimeImputingClassifier):
                                    shap_values_list = []
                                    cols_to_use_list = []
                                    for est in data['pipeline'].estimators_:
                                        shap_values, _, cols_to_use = est.steps[-1][1].shap_values(
                                            data['X_test'])
                                        shap_values_list.append(shap_values[:, :, 1])
                                        cols_to_use_list.append(cols_to_use)
                                    shap_values = np.mean(shap_values_list, axis=0)
                                else:
                                    shap_values = np.mean([shap.Explainer(est.steps[-1][1])(data['X_test']).values[:, :, 1] for est in data['pipeline'].estimators_], axis=0)
                            elif isinstance(data['pipeline'].steps[-1][1], TestTimeImputingClassifier):
                                shap_values, _, cols_to_use = data['pipeline'].steps[-1][1].shap_values(data['X_test'])
                                shap_values = shap_values[:, :, 1]
                            elif isinstance(data['pipeline'].steps[-1][1], BaggingClassifier):
                                if isinstance(data['pipeline'].steps[-1][1].estimators_[0], TestTimeImputingClassifier):
                                    cols_to_use = data['pipeline'].steps[-1][1].estimators_[0].cols_to_use
                                    shap_values = np.mean([est.shap_values(data['X_test'])[0][:, :, 1] for est in
                                         data['pipeline'].steps[-1][1].estimators_], axis=0)
                                else:
                                    shap_values = np.mean([shap.Explainer(est)(data['X_test'])[:, :, 1].values for est in data['pipeline'].steps[-1][1].estimators_], axis=0)
                            else:
                                try:
                                    explainer = shap.Explainer(data['pipeline'].steps[-1][1])
                                    shap_values = explainer(data['X_test'])[:, :, 1].values
                                except:
                                    # masker = shap.maskers.Independent(data=data['X_test'])
                                    # explainer = shap.LinearExplainer(data['pipeline'].steps[-1][1], data['X_train'])
                                    # shap_values = explainer(data['X_test']).values
                                    # shap_values[np.isnan(shap_values)] = 0
                                    raise

                            if cols_to_use is None:
                                vis_idx = data['X_test'].columns.str.startswith('Visual')
                                mean_abs_missing = np.mean(np.abs(
                                    shap_values[:, ~vis_idx] * data['X_test'].isna().values[:,
                                                               ~vis_idx]))
                                mean_abs_present = np.mean(np.abs(
                                    shap_values[:, ~vis_idx] * (~data['X_test'].isna()).values[:,
                                                               ~vis_idx]))
                            else:
                                vis_idx = data['X_test'].iloc[:, cols_to_use].columns.str.startswith('Visual')
                                mean_abs_missing = np.mean(np.abs(
                                    shap_values[:, ~vis_idx] * data['X_test'].iloc[:, cols_to_use].isna().values[:,
                                                               ~vis_idx]))
                                mean_abs_present = np.mean(np.abs(
                                    shap_values[:, ~vis_idx] * (~data['X_test'].iloc[:, cols_to_use].isna()).values[:,
                                                               ~vis_idx]))

                            if np.any(vis_idx):
                                mean_abs_vis = np.mean(np.abs(shap_values[:, vis_idx]))

                            meta_info = {}
                            for k, v in config_dict['meta'].items():
                                meta_info[f'{k}'] = v

                            shap_df.append(
                                {'type': 'missing', 'value': mean_abs_missing, **meta_info})
                            shap_df.append(
                                {'type': 'present', 'value': mean_abs_present, **meta_info})
                            if np.any(vis_idx):
                                shap_df.append(
                                    {'type': 'visual', 'value': mean_abs_vis, **meta_info})
                        except:
                            traceback.print_exc()
                            #                 explainer = shap.KernelExplainer(data['pipeline'].predict_proba, data=data['X_test'])
                            # raise

                    # data_val = pd.DataFrame({'true_0': get_cat_codes(data['y_val']), 'pred_0': data['preds_val_proba'],
                    #                          'pred_label_0': data['preds_val']})
                    data = pd.DataFrame({'true_0': get_cat_codes(data['y_test']), 'pred_0': data['preds_proba'],
                                         'pred_label_0': data['preds']})
                else:
                    is_multi = True
                    preds = data['preds_proba']
                    preds_label = data['preds']
                    print([i for i in range(len(preds)) if preds[i].shape[1] == 1])
                    data_dict = {f'pred_{i}': preds[i][:, 1] for i in range(len(preds)) if preds[i].shape[1] == 2}
                    data_dict.update({f'pred_label_{i}': preds_label[:, i] for i in range(preds_label.shape[1])})
                    data_dict.update({f'true_{i}': np.array(data['y_test'])[:, i] for i in range(len(preds))})
                    data = pd.DataFrame(data_dict)
                    # data_val = None
                # assert all(not (df.model == config_dict['human_name']).any() for df in
                #            dfs), f'{[df.model.unique() for df in dfs]}, {config_dict["human_name"]}'
                for k, v in config_dict['meta'].items():
                    data[f'meta_{k}'] = v

                dfs.append(data)
                # if data_val is not None:
                #     for k, v in config_dict['meta'].items():
                #         data_val[f'meta_{k}'] = v
                #     dfs_val.append(data_val)

        df = pd.concat(dfs, ignore_index=True)
        # df_val = pd.concat(dfs_val, ignore_index=True)

        def bootstrap(a, n_repeat, n_resample, metric):
            values = []
            for i in range(n_repeat):
                indices = np.random.choice(len(a), size=n_resample)
                a_resampled = a[indices]
                values.append(metric(a_resampled))
            return values

        # df['complex'] = get_cat_codes(df.y_test) + 1j * df.pred

        def compute_metrics(df, metrics):
            trues = df.columns[df.columns.str.startswith('true')]
            preds = trues.str.replace('true', 'pred')
            preds_label = trues.str.replace('true', 'pred_label')

            metric_values = []

            for model, model_df in df.groupby(list(df.columns[df.columns.str.startswith('meta_')].values)):
                for col_true, col_pred, col_pred_label in zip(trues, preds, preds_label):
                    if col_pred not in df.columns:
                        continue
                    true = model_df[col_true]
                    pred = model_df[col_pred]
                    pred_label = model_df[col_pred_label]
                    for metric_name, m in metrics.items():
                        def safe_m(a):
                            try:
                                a = a.T
                                if 'zero_division' in inspect.signature(m).parameters.keys():
                                    result = m(a[0], a[1], zero_division=0)
                                    return result if result != 0 else np.nan
                                else:
                                    return m(a[0], a[1])
                            except:
                                try:
                                    if 'zero_division' in inspect.signature(m).parameters.keys():
                                        result = m(a[0], a[2], zero_division=0)
                                        return result if result != 0 else np.nan
                                    else:
                                        return m(a[0], a[2])
                                    # return m(a[0], a[2])
                                except:
                                    return np.nan

                        a = np.array(list(zip(true.values, pred.values, pred_label.values)))
                        metric_value = safe_m(a)
                        if np.isnan(metric_value):
                            continue

                        if is_multi:
                            metric_bootstrapped = bootstrap(a, 5, len(true), safe_m)
                        else:
                            metric_bootstrapped = bootstrap(a, 1000, len(true), safe_m)
                        row = {'metric': metric_name, 'feat': col_true.split('_')[1],
                               'value': metric_value,
                               'lb': np.percentile(metric_bootstrapped, 2.5),
                               'ub': np.percentile(metric_bootstrapped, 97.5)}
                        if isinstance(model, tuple):
                            for k, v in zip(df.columns[df.columns.str.startswith('meta_')], model):
                                row[k.replace('meta_', '')] = v
                        else:
                            k = df.columns[df.columns.str.startswith('meta_')][0]
                            v = model
                            row[k.replace('meta_', '')] = v
                        metric_values.append(row)

            return pd.DataFrame(metric_values)

        r = compute_metrics(df, get_metric_callables_from_dict(meta.overview.metrics))
        # r_val = compute_metrics(df_val, get_metric_callables_from_dict(meta.overview.metrics))

        def grouped_bar_plot(df):
            rows = df[meta.overview.row].unique() if meta.overview.row is not None else [None]
            for i, row in enumerate(rows):
                plt.subplot(len(rows), 1, i + 1)
                if row is not None:
                    plt.title(f'{meta.overview.row} = {row}')
                    row_df = df[df[meta.overview.row] == row]
                else:
                    row_df = df
                width = 0.8
                n_metrics = len(row_df.metric.unique())
                n_x = len(row_df[meta.overview.x].unique())
                x = np.arange(n_x) * (n_metrics * width + 0.2)
                for i, (metric, metric_df) in enumerate(row_df.groupby('metric')):
                    err = np.stack(
                        [metric_df.value.values - metric_df.lb.values, metric_df.ub.values - metric_df.value.values])
                    x_metric = x[np.where([model in list(metric_df.model) for model in row_df.model.unique()])[0]]
                    plt.bar(x_metric + i * width, metric_df.value.values, width=width, label=metric, yerr=err)

                plt.xticks(x + (n_metrics * width) / 2 - width / 2, row_df[meta.overview.x].unique())
                plt.yticks(np.linspace(0.0, 1.0, 21))
                plt.xlabel('Model')
                plt.ylabel('Metric value')
                plt.legend()

        Path(os.path.join(experiments_path, 'results')).mkdir(parents=True, exist_ok=True)

        if is_multi:
            if meta.overview.row is None:
                kind = 'swarm' if len(r[(r.metric == r.metric.unique()[0]) &
                                        (r[meta.overview.x] == r[meta.overview.x].unique()[0])]) < 100 else 'box'
                kwargs = dict(dodge=True) if kind == 'swarm' else {}
                # r.rename(columns={meta.overview.x: meta.overview.x.capitalize(), 'value': 'Metric value', 'metric': 'Metric'}, inplace=True)
                r.rename(columns={'impute': 'Imputation Method', 'value': 'Metric value', 'metric': 'Metric'}, inplace=True)
                sns.catplot(data=r, x='Imputation Method', y='Metric value', hue='Metric', kind=kind, **kwargs)
            else:
                kind = 'swarm' if len(r[(r.metric == r.metric.unique()[0]) &
                                        (r[meta.overview.x] == r[meta.overview.x].unique()[0]) &
                                        (r[meta.overview.row] == r[meta.overview.row].unique()[0])]) < 100 else 'box'
                kwargs = dict(dodge=True) if kind == 'swarm' else {}
                # meta.overview.x: meta.overview.x.capitalize()
                r.rename(columns={'impute': 'Imputation Method', 'value': 'Metric value', 'metric': 'Metric'}, inplace=True)
                sns.catplot(data=r, x='Imputation Method', y='value', hue='metric', row=meta.overview.row, kind=kind, **kwargs)
            # plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)
            plt.savefig(os.path.join(experiments_path, 'results', 'overview.pdf'))
        else:
            if meta.plot_shap:
                shap_df = pd.DataFrame(shap_df)
                plt.figure(figsize=(10, 5))
                shap_groupby = shap_df.groupby(meta.overview.x if meta.overview.row is None else (meta.overview.x, meta.overview.row))
                shap_df['value'] = shap_groupby.value.transform(lambda x: x/x.sum())
                shap_df.loc[shap_df.type == 'missing', 'type'] = 'Missing'
                shap_df.loc[shap_df.type == 'present', 'type'] = 'Present'
                shap_df.loc[shap_df.type == 'visual', 'type'] = 'Visual'

                if meta.overview.row is None:
                    shap_df.rename(columns={meta.overview.x: meta.overview.x.capitalize(), 'value': 'Mean absolute SHAP value', 'type': 'Feature type'}, inplace=True)
                    g = sns.barplot(data=shap_df, x=meta.overview.x.capitalize(), y='Mean absolute SHAP value', hue='Feature type')
                else:
                    shap_df.rename(columns={'value': 'Mean absolute SHAP value', 'type': 'Feature type', meta.overview.row: meta.overview.row.capitalize()}, inplace=True)
                    g = sns.barplot(data=shap_df, x=meta.overview.x, y='Mean absolute SHAP value', hue='Feature type', row=meta.overview.row.capitalize())
                # plt.gcf().get_axes()[0].set_yscale('log')
                tikzplotlib.save(os.path.join(experiments_path, 'results', 'shaps.tex'), axis_width=r'1.3\textwidth')
                plt.savefig(os.path.join(experiments_path, 'results', 'shaps.pdf'))

            plt.figure(figsize=(10, 5 * (1 if meta.overview.row is None else len(r[meta.overview.row].unique()))))
            grouped_bar_plot(r)
            tikzplotlib.save(os.path.join(experiments_path, 'results', 'overview.tex'), axis_width=r'1.3\textwidth')
            plt.savefig(os.path.join(experiments_path, 'results', 'overview.pdf'))

            r.to_csv(os.path.join(experiments_path, 'results', 'data.csv'))
            with open(os.path.join(experiments_path, 'results', 'data.tex'), 'w') as f:
                f.write(gen_latex_table(r, [meta.overview.x] if meta.overview.row is None else [meta.overview.x, meta.overview.row]))

            # plt.figure(figsize=(10, 5 * (1 if meta.overview.row is None else len(r_val[meta.overview.row].unique()))))
            # grouped_bar_plot(r_val)
            # tikzplotlib.save(os.path.join(experiments_path, 'results', 'overview_val.tex'), axis_width=r'1.3\textwidth')
            # plt.savefig(os.path.join(experiments_path, 'results', 'overview_val.pdf'))
            #
            # r_val.to_csv(os.path.join(experiments_path, 'results', 'data_val.csv'))
            # with open(os.path.join(experiments_path, 'results', 'data_val.tex'), 'w') as f:
            #     f.write(gen_latex_table(r_val, [meta.overview.x] if meta.overview.row is None else [meta.overview.x,
            #                                                                                     meta.overview.row]))
            # from covidcf.reporting.statistics import compare_models, delong_roc_test, bootstrap_test, mcnemar_test
            # # names = df[f'meta_{meta.overview.x}'].unique()
            # groups = list(df.groupby(list(df.columns[df.columns.str.startswith('meta_')].values)))
            # names = [ind if isinstance(ind, str) else ', '.join(ind) for ind, df_model in groups]
            # print('Generating stats')
            # stats = compare_models(get_cat_codes(groups[0][1].true_0).astype(np.float), names,
            #                        [df_model.pred_0 for ind, df_model in groups], lambda y_true, y1, y2: 10**delong_roc_test(y_true, y1, y2), test_all=False)
            # print('Saving stats')
            # stats.to_csv(os.path.join(experiments_path, 'results', 'delong.csv'), sep=';')
            # with open(os.path.join(experiments_path, 'results', 'delong.tex'), 'w') as f:
            #     stats.to_latex()


@click.command(name='run-experiments')
@click.argument('experiments-path', type=click.Path(exists=True, dir_okay=True))
@click.option('--rerun-hyperopt/--no-rerun-hyperopt', default=False)
@click.option('--rerun-evaluation/--no-rerun-evaluation', default=False)
@click.option('--rerun-results/--no-rerun-results', default=False)
def run_experiments_command(experiments_path: str, rerun_hyperopt: bool = False, rerun_evaluation: bool = False,
                            rerun_results: bool = False):
    run_experiments(experiments_path, rerun_hyperopt, rerun_evaluation, rerun_results)
