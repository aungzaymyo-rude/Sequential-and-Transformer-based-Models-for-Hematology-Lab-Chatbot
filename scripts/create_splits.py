from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.splits import create_stratified_splits, load_dataset, write_splits

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'intent_dataset.jsonl'
DEFAULT_SPLIT_DIR = Path(__file__).resolve().parents[1] / 'data' / 'splits'


def main() -> None:
    parser = argparse.ArgumentParser(description='Create fixed train/validation/test split files')
    parser.add_argument('--dataset', default=str(DEFAULT_DATASET))
    parser.add_argument('--output-dir', default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    rows = load_dataset(dataset_path)
    splits = create_stratified_splits(rows, test_size=args.test_size, val_size=args.val_size, seed=args.seed)
    write_splits(splits, Path(args.output_dir).resolve(), overwrite=args.overwrite)
    print({name: len(records) for name, records in splits.items()})


if __name__ == '__main__':
    main()
