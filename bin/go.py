import argparse
import shutil
import sys
from pathlib import Path
from typing import cast

from loguru import logger

import bin.evaluate
import bin.tune
import lib.experiment
import lib.util


def main(
    exp: str | Path,  # A tuning or evaluation experiment.
    *,
    n_seeds: int,
    force: bool = False,
    resume: bool = False,
):
    if not resume:
        # Supporting resume=False for a sequence of experiments is tricky.
        raise ValueError('Currently, only resume=True is supported')

    exp = Path(exp).resolve()
    assert lib.experiment.is_experiment(exp)
    assert exp.name in ('tuning', 'evaluation')

    is_tuning = exp.name == 'tuning'
    tuning_exp = exp.with_name('tuning')
    evaluation_exp = exp.with_name('evaluation')
    del exp

    if force:
        logger.warning('Resetting and removing existing experiments')
        # The tuning, evaluation and ensemble experiments are processed
        # in the reversed order.
        if is_tuning:
            if evaluation_exp.exists():
                shutil.rmtree(evaluation_exp)
            lib.experiment.reset(tuning_exp)
        else:
            lib.experiment.reset(evaluation_exp)

    if is_tuning:
        if not lib.experiment.is_done(tuning_exp):
            assert not evaluation_exp.exists()

        lib.experiment.run(bin.tune.main, None, tuning_exp, resume=resume)

        tuning_config = cast(bin.tune.Config, lib.experiment.load_config(tuning_exp))
        tuning_report = lib.experiment.load_report(tuning_exp)
        evaluation_config: bin.evaluate.Config = {
            'function': tuning_config['function'],
            'n_seeds': n_seeds,
            'base_config': tuning_report['best']['config'],
        }
        for key in ['n_workers', 'n_gpus_per_worker']:
            if key in tuning_config:
                evaluation_config[key] = tuning_config[key]
            del key
        evaluation_config['base_config'].pop('seed', None)
        if lib.experiment.get_config_path(evaluation_exp).exists():
            # NOTE
            # If tuning_exp was done, this branch is active. In theory, the existing
            # evaluation config must be identical to `evaluation_config`. In practice,
            # it is unclear if TOML and JSON serializations can lead to a difference.
            assert lib.experiment.load_config(evaluation_exp) == evaluation_config
        else:
            lib.experiment.create(evaluation_exp, config=evaluation_config, force=True)  # type: ignore

    lib.experiment.run(bin.evaluate.main, None, evaluation_exp, resume=resume)


if __name__ == '__main__':
    lib.util.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    parser.add_argument('--n-seeds', type=int, required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--resume', action='store_true')

    main(**vars(parser.parse_args(sys.argv[1:])))
