from lightgbm.basic import _ConfigAliases
from operator import gt, lt
from typing import Callable, List
from lightgbm.callback import EarlyStopException, CallbackEnv


def early_stopping(stopping_rounds: int, target: str, first_metric_only: bool = False,
                   verbose: bool = True) -> Callable:
    """Create a callback that activates early stopping.

    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.

    Parameters
    ----------
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
       Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information.

    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list: list = []
    cmp_op = []
    enabled = [True]
    first_metric = ['']

    def _init(env: CallbackEnv) -> None:
        enabled[0] = not any(env.params.get(boost_alias, "") == 'dart' for boost_alias
                             in _ConfigAliases.get("boosting"))
        if not enabled[0]:
            print('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')

        # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
        first_metric[0] = env.evaluation_result_list[0][1].split(" ")[-1]
        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)

    def _final_iteration_check(env: CallbackEnv, eval_name_splitted: List[str], i: int) -> None:
        if env.iteration == env.end_iteration - 1:
            raise EarlyStopException(best_iter[i], best_score_list[i])

    def _callback(env: CallbackEnv) -> None:
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        for i in range(len(env.evaluation_result_list)):
            score = env.evaluation_result_list[i][2]
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
            # split is needed for "<dataset type> <metric>" case (e.g. "train l1")
            eval_name_splitted = env.evaluation_result_list[i][1].split(" ")
            # print(eval_name_splitted[0])
            # print(env.evaluation_result_list[i][0])
            if first_metric_only and first_metric[0] != eval_name_splitted[-1]:
                continue  # use only the first metric for early stopping
            if ((env.evaluation_result_list[i][0] == "cv_agg" and eval_name_splitted[0] == "train"
                 or env.evaluation_result_list[i][0] == env.model._train_data_name)) or env.evaluation_result_list[i][
                0] != target:
                _final_iteration_check(env, eval_name_splitted, i)
                continue  # train data for lgb.cv or sklearn wrapper (underlying lgb.train)
            elif env.iteration - best_iter[i] >= stopping_rounds:
                print('early stopping activated...')
                raise EarlyStopException(best_iter[i], best_score_list[i])
            _final_iteration_check(env, eval_name_splitted, i)

    _callback.order = 30  # type: ignore
    return _callback