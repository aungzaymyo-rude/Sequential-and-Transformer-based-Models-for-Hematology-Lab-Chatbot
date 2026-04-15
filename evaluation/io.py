from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def create_run_dir(results_dir: Path, experiment_name: str) -> Path:
    run_dir = results_dir / f'{timestamp()}_{experiment_name}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_confusion_matrix(path: Path, matrix: List[List[int]], label_names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write(',' + ','.join(label_names) + '\n')
        for label, row in zip(label_names, matrix):
            handle.write(label + ',' + ','.join(str(value) for value in row) + '\n')
