import json
from pathlib import Path
from typing import Any

import numpy as np

import lib.util

CHURN = 'churn'
CALIFORNIA = 'california'
HOUSE = 'house'
ADULT = 'adult'
DIAMOND = 'diamond'
OTTO = 'otto'
HIGGS_SMALL = 'higgs-small'
BLACK_FRIDAY = 'black-friday'
MICROSOFT = 'microsoft'

SBERBANK_HOUSING = 'sberbank-housing'
ECOM_OFFERS = 'ecom-offers'
MAPS_ROUTING = 'maps-routing'
HOMESITE_INSURANCE = 'homesite-insurance'
COOKING_TIME = 'cooking-time'
HOMECREDIT_DEFAULT = 'homecredit-default'
DELIVERY_ETA = 'delivery-eta'
WEATHER = 'weather'

DATASETS_DEFAULT = [
    CHURN,
    CALIFORNIA,
    HOUSE,
    ADULT,
    DIAMOND,
    OTTO,
    HIGGS_SMALL,
    BLACK_FRIDAY,
    MICROSOFT,
]

DATASETS_TABRED = [
    SBERBANK_HOUSING,
    ECOM_OFFERS,
    MAPS_ROUTING,
    HOMESITE_INSURANCE,
    COOKING_TIME,
    HOMECREDIT_DEFAULT,
    DELIVERY_ETA,
    WEATHER,
]


def datasets_all() -> list[str]:
    return [*DATASETS_DEFAULT, *DATASETS_TABRED]


def wrap_dataset_name(name: str) -> str:
    """Return the public experiment directory name for a dataset.

    The public repo flattens dataset paths, so this is the identity.
    """
    return f'tabred/{name}' if name in DATASETS_TABRED else name


normalize_dataset_name = wrap_dataset_name


def infer_dataset_split(name: str) -> int:
    del name
    return 0


def load_dataset_info(path: str | Path) -> dict[str, Any]:
    if isinstance(path, str):
        path = path.removeprefix(':')
    path = Path(path)
    assert path.exists(), f'Dataset {path} does not exist.'

    info = {
        'path': str(lib.util.try_get_relative_path(path)),
        'name': normalize_dataset_name(path.name),
        'split': infer_dataset_split(path.name),
        'task_type': json.loads(path.joinpath('info.json').read_text())['task_type'],
    }
    for part in ['train', 'val', 'test']:
        info[f'{part}_size'] = len(np.load(path / f'Y_{part}.npy'))

    info['n_features'] = 0
    for ftype in ['num', 'bin', 'cat']:
        x_path = path / f'X_{ftype}_val.npy'
        n = np.load(x_path).shape[1] if x_path.exists() else 0
        info[f'n_{ftype}_features'] = n
        info['n_features'] += n

    return info
