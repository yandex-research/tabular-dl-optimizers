import shutil
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import delu
from loguru import logger

import lib.experiment
import lib.util


class Config(TypedDict):
    function: str
    n_seeds: int
    base_config: dict[str, Any]


def _evaluate_seed(config: Config, exp: Path, seed: int, timer: delu.tools.Timer) -> None:
    function = lib.util.import_(config['function'])

    seed_exp = exp / str(seed)
    if seed_exp.exists():
        logger.warning(f'Removing the incomplete experiment {seed_exp}')
        shutil.rmtree(seed_exp)

    seed_config: dict[str, Any] = {'seed': seed, **config['base_config']}
    with tempfile.TemporaryDirectory(suffix=f'_evaluation_{seed}') as tmp_exp_dir:
        tmp_exp = lib.experiment.create(tmp_exp_dir, config=seed_config, force=True)
        seed_report = lib.experiment.run(function, None, tmp_exp)
        assert seed_report is not None
        lib.experiment.remove_tracked_files(tmp_exp)
        lib.experiment.move(tmp_exp, seed_exp)

    report = lib.experiment.load_report(exp)
    report.setdefault('experiments', []).append({'config': seed_config, 'report': seed_report})
    report['experiments'].sort(key=lambda x: x['config']['seed'])
    report['time'] = timer.elapsed()
    lib.experiment.dump_report(exp, report)
    lib.experiment.dump_summary(exp, lib.experiment.summarize(report))

    if len(report['experiments']) < config['n_seeds']:
        print(lib.util.add_frame(lib.experiment.load_summary(exp)))


def main(config: Config, exp: str | Path) -> lib.experiment.Report:
    exp = Path(exp)
    assert 'seed' not in config['base_config']
    assert exp.name == 'evaluation'

    report = lib.experiment.create_report(main, add_gpu_info=False)
    lib.experiment.dump_report(exp, report)

    timer = delu.tools.Timer()
    timer.run()
    for seed in range(config['n_seeds']):
        _evaluate_seed(config, exp, seed, timer)

    report = lib.experiment.load_report(exp)
    lib.experiment.finish(exp, report)
    return report


if __name__ == '__main__':
    lib.util.init()
    lib.experiment.run_cli(main)
