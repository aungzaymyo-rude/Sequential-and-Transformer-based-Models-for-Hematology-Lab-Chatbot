from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from preprocessing.text import normalize_text

Record = Dict[str, str]
ALLOWED_EXTENSIONS = {'.jsonl', '.csv'}


def iter_jsonl(path: Path) -> Iterable[Record]:
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_csv(path: Path) -> Iterable[Record]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def load_records(path: Path) -> List[Record]:
    loader = iter_jsonl if path.suffix.lower() == '.jsonl' else iter_csv
    records: List[Record] = []
    for row in loader(path):
        normalized = normalize_record(row, source=str(path))
        if normalized:
            records.append(normalized)
    return records


def normalize_record(row: Record, source: str | None = None, default_lang: str = 'en') -> Record | None:
    text = (row.get('text') or row.get('utterance') or '').strip()
    intent = (row.get('intent') or row.get('label') or '').strip()
    lang = (row.get('lang') or row.get('language') or default_lang).strip().lower() or default_lang
    if not text or not intent:
        return None
    return {
        'text': text,
        'intent': intent,
        'lang': lang,
        'source': row.get('source') or source or 'unknown',
        'text_normalized': normalize_text(text),
    }


def discover_labeled_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([path for path in folder.rglob('*') if path.suffix.lower() in ALLOWED_EXTENSIONS])


def dedupe_records(records: Sequence[Record]) -> List[Record]:
    seen: set[tuple[str, str, str]] = set()
    output: List[Record] = []
    for row in records:
        key = (row['text_normalized'], row['intent'], row['lang'])
        if key in seen:
            continue
        seen.add(key)
        output.append({
            'text': row['text'],
            'intent': row['intent'],
            'lang': row['lang'],
            'source': row.get('source', 'unknown'),
        })
    return output


def write_jsonl(records: Sequence[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def build_label_maps(records: Sequence[Record]) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted({row['intent'] for row in records})
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label
