from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.dataset import dedupe_records, discover_labeled_files, load_records, write_jsonl

DEFAULT_BASE = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'bootstrap.jsonl'
DEFAULT_LABELED_DIR = Path(__file__).resolve().parents[1] / 'data' / 'labeled'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'intent_dataset.jsonl'


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge labeled data into one processed dataset')
    parser.add_argument('--base', default=str(DEFAULT_BASE))
    parser.add_argument('--external-dir', default=str(DEFAULT_LABELED_DIR))
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    external_dir = Path(args.external_dir).resolve()
    output_path = Path(args.output).resolve()

    base_rows = load_records(base_path) if base_path.exists() else []
    external_rows = []
    for file_path in discover_labeled_files(external_dir):
        external_rows.extend(load_records(file_path))
    merged = dedupe_records(base_rows + external_rows)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f'{output_path} already exists. Use --overwrite to replace it.')
    write_jsonl(merged, output_path)
    print(f'Base rows: {len(base_rows)}')
    print(f'External rows: {len(external_rows)}')
    print(f'Written: {len(merged)} -> {output_path}')


if __name__ == '__main__':
    main()
