from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    return resolve_paths(config)


def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    path_keys = {
        ('data', 'labeled_dir'),
        ('data', 'processed_dataset_path'),
        ('data', 'splits_dir'),
        ('results_dir',),
    }
    for keys in path_keys:
        cursor = config
        for key in keys[:-1]:
            cursor = cursor.get(key, {})
        leaf = keys[-1]
        value = cursor.get(leaf)
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                cursor[leaf] = str((PROJECT_ROOT / path).resolve())
    return config
