import time
import tempfile
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import delu
import optuna
import optuna.samplers
import optuna.trial

import lib.experiment
import lib.util
from lib.types import KWArgs


type ConfigSpace = dict[str, Any]


class Config(TypedDict):
    seed: int
    function: str
    space: ConfigSpace
    n_trials: NotRequired[int]
    timeout: NotRequired[int]
    sampler: NotRequired[KWArgs]


def _sample_value(
    trial: optuna.trial.Trial,
    distribution: str,
    label: str,
    *args,
):
    trial_suggest, kwargs = {
        'int': (trial.suggest_int, {}),
        'uniform': (trial.suggest_float, {}),
        'loguniform': (trial.suggest_float, {'log': True}),
        'categorical': (trial.suggest_categorical, {}),
    }[distribution]
    if distribution in ('int', 'uniform', 'loguniform') and len(args) == 3:
        args, kwargs['step'] = args[:2], args[2]
    return trial_suggest(label, *args, **kwargs)


def _sample_config(
    trial: optuna.trial.Trial,
    space: bool | int | float | str | bytes | list | dict,
    label_parts: list,
) -> Any:
    if isinstance(space, bool | int | float | str | bytes):
        return space
    if isinstance(space, list):
        if space and space[0] == '_tune_':
            _, distribution, *args = space
            label = '.'.join(map(str, label_parts))
            if distribution.startswith('?'):
                default, args_ = args[0], args[1:]
                if trial.suggest_categorical(f'?{label}', [False, True]):
                    return _sample_value(trial, distribution.lstrip('?'), label, *args_)
                return default
            if distribution == '$list':
                size, item_distribution, *item_args = args
                return [
                    _sample_value(trial, item_distribution, f'{label}.{i}', *item_args)
                    for i in range(size)
                ]
            return _sample_value(trial, distribution, label, *args)
        else:
            return [
                _sample_config(trial, subspace, [*label_parts, i])
                for i, subspace in enumerate(space)
            ]
    if isinstance(space, dict):
        if '_tune_' in space:
            raise ValueError(f'Unknown custom distribution: {space["_tune_"]}')
        return {
            key: _sample_config(trial, subspace, [*label_parts, key])
            for key, subspace in space.items()
        }


def _objective(
    trial: optuna.trial.Trial,
    *,
    function: lib.experiment.MainFunction,
    space: ConfigSpace,
    timer: delu.tools.Timer,
) -> float:
    trial_config = _sample_config(trial, space, [])

    with tempfile.TemporaryDirectory(suffix=f'_trial_{trial.number}') as tmp_exp:
        print()
        tmp_exp = lib.experiment.create(tmp_exp, config=trial_config, force=True)
        trial_report = lib.experiment.run(function, None, tmp_exp)

    assert trial_report is not None
    trial_report['tuning'] = {'trial_id': trial.number, 'time': timer.elapsed()}
    trial.set_user_attr('experiment', {'config': trial_config, 'report': trial_report})
    delu.cuda.free_memory()
    return trial_report['metrics']['val']['score']


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    report = lib.experiment.create_report(main, add_gpu_info=False)

    assert exp.name == 'tuning'

    delu.random.seed(config['seed'])
    function = lib.util.import_(config['function'])

    n_trials = config.get('n_trials')
    timeout = config.get('timeout')

    if lib.experiment.get_checkpoint_path(exp).exists():
        checkpoint = lib.experiment.load_checkpoint(exp)
        report = checkpoint['report']
        study = checkpoint['study']
        timer = checkpoint['timer']
        delu.random.set_state(checkpoint['random_state'])

        n_completed_trials = len(
            study.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)
        )
        if n_trials is not None:
            n_trials -= n_completed_trials
        if timeout is not None:
            timeout -= timer.elapsed()

        report.setdefault('resumed_after_n_trials', []).append(n_completed_trials)
        print(
            'Resuming from a checkpoint.'
            f' Completed {n_completed_trials}/{config.get("n_trials", "inf")} trials'
        )
        time.sleep(1.0)
    else:
        sampler_config = config.get('sampler', {}).copy()
        sampler_cls = getattr(optuna.samplers, sampler_config.pop('type', 'TPESampler'))
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler_cls(seed=config['seed'], **sampler_config),
        )
        report = lib.experiment.create_report(main, add_gpu_info=False)
        timer = delu.tools.Timer()

    lib.experiment.dump_report(exp, report)

    def callback(study: optuna.study.Study, _: optuna.trial.FrozenTrial):
        best = study.best_trial.user_attrs['experiment']
        report = lib.experiment.load_report(exp)
        report['best'] = best
        report['time'] = timer.elapsed()
        report['n_completed_trials'] = len(
            study.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)
        )
        lib.experiment.dump_checkpoint(
            exp,
            {
                'report': report,
                'study': study,
                'timer': timer,
                'random_state': delu.random.get_state(),
            },
        )
        lib.experiment.dump_report(exp, report)
        lib.experiment.dump_summary(exp, lib.experiment.summarize(report))
        if report['n_completed_trials'] != config.get('n_trials'):
            print(lib.util.add_frame(lib.experiment.load_summary(exp)))

    timer = delu.tools.Timer()
    timer.run()

    study.optimize(
        lambda trial: _objective(
            trial,
            function=function,
            space=config['space'],
            timer=timer,
        ),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback],
    )

    report = lib.experiment.load_report(exp)
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main, resumable=True)
