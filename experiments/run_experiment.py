from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.train import run_sequential_experiment
from experiments.common import load_config
from transformer_models.train import run_transformer_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a comparable medical intent experiment')
    parser.add_argument('--config', required=True, type=str, help='Path to experiment config')
    args = parser.parse_args()

    config = load_config(args.config)
    start = time.perf_counter()
    family = config['model']['family']
    if family == 'transformer':
        run_dir = run_transformer_experiment(config)
    elif family == 'sequential':
        run_dir = run_sequential_experiment(config)
    else:
        raise ValueError(f'Unsupported model family: {family}')
    elapsed = time.perf_counter() - start

    summary = {
        'experiment_name': config['experiment_name'],
        'family': family,
        'run_dir': str(run_dir),
        'elapsed_seconds': elapsed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
