from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

from preprocessing.dataset import Record, iter_jsonl, write_jsonl

SPLIT_NAMES = ('train', 'val', 'test')


def load_dataset(path: Path) -> List[Record]:
    return list(iter_jsonl(path))


def create_stratified_splits(records: List[Record], test_size: float, val_size: float, seed: int) -> Dict[str, List[Record]]:
    if not records:
        raise ValueError('Dataset is empty.')
    labels = [row['intent'] for row in records]
    train_val, test = train_test_split(records, test_size=test_size, random_state=seed, stratify=labels)
    if val_size <= 0:
        return {'train': train_val, 'val': [], 'test': test}
    train_val_labels = [row['intent'] for row in train_val]
    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=adjusted_val_size, random_state=seed, stratify=train_val_labels)
    return {'train': train, 'val': val, 'test': test}


def write_splits(splits: Dict[str, List[Record]], output_dir: Path, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        split_path = output_dir / f'{split_name}.jsonl'
        if split_path.exists() and not overwrite:
            raise FileExistsError(f'{split_path} already exists. Use overwrite=True to replace it.')
        write_jsonl(splits.get(split_name, []), split_path)


def ensure_split_files(dataset_path: Path, splits_dir: Path, test_size: float, val_size: float, seed: int) -> Dict[str, Path]:
    expected = {name: splits_dir / f'{name}.jsonl' for name in SPLIT_NAMES}
    if all(path.exists() for path in expected.values()):
        return expected
    splits = create_stratified_splits(load_dataset(dataset_path), test_size=test_size, val_size=val_size, seed=seed)
    write_splits(splits, splits_dir, overwrite=True)
    return expected
