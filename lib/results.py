import functools
import itertools
import string
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

import lib.experiment
import lib.util
from lib.types import JSONDict

from . import paper_data

# Extended reports are flattened and converted to dataframe rows.
type ExtendedReport = JSONDict


@functools.lru_cache(None)
def _load_dataset_info_cached(*args, **kwargs) -> dict[str, Any]:
    return paper_data.load_dataset_info(*args, **kwargs)


@functools.lru_cache(None)
def _load_report_cached(*args, **kwargs) -> dict[str, Any]:
    return lib.experiment.load_report(*args, **kwargs)


def make_extended_report(
    *,
    report: lib.experiment.Report,
    config: JSONDict,
    name: str,
    seed: str,
    dataset_dir: str | Path,
    cache_dataset_info: bool = True,
) -> ExtendedReport:
    assert 'metrics' in report, 'The report is missing the required "metrics" field'

    load_dataset_info_fn = (
        _load_dataset_info_cached
        if cache_dataset_info
        else paper_data.load_dataset_info
    )

    extended_report = deepcopy(report)
    extended_report['config'] = config
    extended_report['Name'] = name
    extended_report['Seed'] = seed
    extended_report['Dataset'] = load_dataset_info_fn(dataset_dir)

    for part_metrics in extended_report['metrics'].values():
        part_metrics['unified_score'] = part_metrics[
            'r2' if extended_report['Dataset']['task_type'] == 'regression' else 'score'
        ]

    return extended_report


def load_evaluation_extended_reports(
    exp: str | Path,
    report: None | lib.experiment.Report,
    *,
    n_seeds: None | int = None,
    **kwargs,
) -> list[ExtendedReport]:
    if report is None:
        report = lib.experiment.load_report(exp)
    experiments = report['experiments']
    if n_seeds is not None:
        experiments = experiments[:n_seeds]
    return [
        make_extended_report(
            **x,
            seed=x['config']['seed'],
            dataset_dir=x['config']['data']['path'],
            **kwargs,
        )
        for x in experiments
    ]


def load_tuning_extended_reports(
    exp: str | Path, report: None | lib.experiment.Report, **kwargs
) -> list[ExtendedReport]:
    if report is None:
        report = lib.experiment.load_report(exp)
    best_experiment = report['best']
    return [
        make_extended_report(
            **best_experiment,
            seed=best_experiment['config']['seed'],
            dataset_dir=best_experiment['config']['data']['path'],
            **kwargs,
        )
    ]


type LoadFn = Callable[..., list[ExtendedReport]]


def load_records(
    named_experiments_: dict[str, str | Path | list[str] | list[Path]],
    /,
    load_fn: None | LoadFn | dict[str, LoadFn] = None,
    *,
    require_done: bool = True,
    allow_skipping: bool = False,
    progress_bar: bool = False,
    verbose: bool = False,
    cache: bool = False,
    **kwargs,
) -> list[JSONDict]:
    named_experiments = {
        k: v if isinstance(v, list) else [v] for k, v in named_experiments_.items()
    }

    if load_fn is None:
        load_fn = {
            'bin.evaluate.main': load_evaluation_extended_reports,
            'bin.tune.main': load_tuning_extended_reports,
        }

    records = []
    skipped = {}
    with tqdm(
        total=sum(map(len, named_experiments.values())), disable=not progress_bar
    ) as pbar:
        for name, exps in named_experiments.items():
            for exp in exps:
                if lib.experiment.is_experiment(exp) and (
                    lib.experiment.is_done(exp) or not require_done
                ):
                    report = (
                        _load_report_cached(Path(exp).resolve())
                        if cache
                        else lib.experiment.load_report(exp)
                    )
                    exp_load_fn = (
                        load_fn[report['function']]
                        if isinstance(load_fn, dict)
                        else load_fn
                    )
                    records.extend(
                        lib.util.flatten_dict(x)
                        for x in exp_load_fn(
                            exp, report, name=name, cache_dataset_info=cache, **kwargs
                        )
                    )
                elif allow_skipping:
                    if verbose:
                        print(f'Skipping {name=} {exp=}')
                    skipped.setdefault(name, []).append(exp)
                else:
                    raise RuntimeError(
                        f'The experiment is either missing or not done: {name=} {exp=}.'
                        ' Consider passing require_done=False or allow_skipping=True.'
                    )
                pbar.update()

    if skipped:
        logger.warning(
            f'#Skipped: {{ {", ".join(f"{k}: {len(v)}" for k, v in skipped.items())} }}'
        )

    return records


def clear_cache():
    _load_dataset_info_cached.cache_clear()
    _load_report_cached.cache_clear()


def load_dataframe(
    *experiments: (
        dict[str, str | Path | list[str] | list[Path]]
        | str
        | Path
        | list[str]
        | list[Path]
    ),
    name_fn: None | Callable[[str | Path], str] = None,
    **kwargs,
) -> pd.DataFrame:
    if len(experiments) == 1 and isinstance(experiments[0], dict):
        if name_fn is not None:
            raise ValueError(
                'When experiments are passed as a dictionary, name_fn must not be provided'
            )
        named_experiments = experiments[0]
    else:
        if name_fn is None:
            letters = string.ascii_uppercase
            named_experiments = {
                letter * (i // len(letters) + 1): x
                for i, (x, letter) in enumerate(
                    zip(experiments, itertools.cycle(letters))
                )
            }
        else:
            named_experiments = {
                name_fn(x[0] if isinstance(x, list) else x): x  # type: ignore
                for x in experiments  # type: ignore
            }
            if len(named_experiments) < len(experiments):
                raise RuntimeError(
                    'The provided name_fn generated the same name for different experiments'
                )

    records = load_records(named_experiments, **kwargs)  # type: ignore
    return pd.DataFrame.from_records(records)


def aggregate_all_columns(
    df: pd.DataFrame,
    num_statistics: str | list[str],
    *,
    by: None | str | list[str],
) -> pd.DataFrame:
    if isinstance(num_statistics, str):
        num_statistics = [num_statistics]

    aggregations = {}
    for column in df.columns:
        if by is not None and column in by:
            continue

        aggregations.setdefault('Count', (column, 'count'))
        if column in ('Dataset.split', 'Seed', 'config.seed'):
            aggregations[column] = (column, list)
        elif column.startswith(
            ('config.', 'Dataset.')
        ) or not pd.api.types.is_numeric_dtype(df[column].dtype):
            aggregations[column] = (column, 'first')
        else:
            aggregations.update((f'{column}.{x}', (column, x)) for x in num_statistics)

    return (df if by is None else df.groupby(by)).agg(**aggregations)


def drop_incomplete_datasets(df: pd.DataFrame) -> pd.DataFrame:
    name_column = 'Name'
    n_models = (
        df[name_column].nunique()
        if name_column in df.columns
        else len(df.index.unique(level=name_column))
    )
    return df.groupby('Dataset.name').filter(lambda x: len(x) == n_models)


def _compute_ranks_impl_(
    df: pd.DataFrame, mean_column: str, std_column: None | str
) -> pd.DataFrame:
    df = (
        df.sort_values(mean_column, ascending=False)
        if std_column is None
        else df.sort_values([mean_column, std_column], ascending=[False, True])
    )
    ranks = []
    current_mean = None
    current_std = None
    for _, columns in df.iterrows():
        mean = columns[mean_column]
        std = 0.0 if std_column is None else columns[std_column]
        if current_mean is None:
            ranks.append(1)
            current_mean = mean
            current_std = std
        elif current_mean - mean <= current_std:
            ranks.append(ranks[-1])
        else:
            ranks.append(ranks[-1] + 1)
            current_mean = mean
            current_std = std
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df[f'{mean_column}.rank'] = ranks
    return df


def compute_ranks(
    df: pd.DataFrame,
    mean_column: str,
    std_column: None | str,
    *,
    by: None | str | list[str],
    inplace: bool = False,
    **kwargs,
) -> pd.DataFrame:
    if by is None:
        if not inplace:
            df = df.copy()
        _compute_ranks_impl_(df, mean_column, std_column, **kwargs)
        return df
    if inplace:
        raise ValueError('inplace=True is not allowed when `by` is not None')
    return df.groupby(by, group_keys=False).apply(
        _compute_ranks_impl_, mean_column=mean_column, std_column=std_column, **kwargs
    )


def compute_relative_metrics_(
    df: pd.DataFrame,
    reference_name: str,
    metric_columns: list[str],
    *,
    by: None | str | list[str],
) -> None:
    name_column = 'Name'
    if by is None:
        by = []
    elif isinstance(by, str):
        by = [by]

    index = df.index.names
    if index == [None]:
        index = None
    if index is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
            df.reset_index(inplace=True)

    df.set_index([*by, name_column], inplace=True)
    df_reference_metrics = df.loc[
        (*[slice(None) for _ in by], reference_name), metric_columns
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df[[f'{x}.relative.{reference_name}' for x in metric_columns]] = 100.0 * (
            df[metric_columns] / df_reference_metrics.droplevel(name_column) - 1.0
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        df.reset_index(inplace=True)
    if index is not None:
        df.set_index(index, inplace=True)


def interquantile_mean(
    x: np.ndarray | pd.Series, /, lower: None | float, upper: None | float
) -> float:
    if lower is None and upper is None:
        raise ValueError('At least one of lower and upper must be provided')
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(
            f'lower must be less than upper, however: {lower=} and {upper=}'
        )

    lower_mask = None if lower is None else x >= np.quantile(x, lower)
    upper_mask = None if upper is None else x < np.quantile(x, upper)
    mask = (
        lower_mask
        if upper_mask is None
        else upper_mask
        if lower_mask is None
        else (lower_mask & upper_mask)
    )
    assert mask is not None
    x_masked = x[mask] if isinstance(x, np.ndarray) else x.loc[mask]
    return float(x_masked.mean())
