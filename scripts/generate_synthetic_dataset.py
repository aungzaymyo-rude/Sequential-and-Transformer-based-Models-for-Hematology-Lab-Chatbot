from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.dataset import write_jsonl

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'bootstrap.jsonl'

TEMPLATES = {
    'greeting': ['hello','hi there','good morning','hello, i have a hematology question','good evening, can you help'],
    'help': ['i need help with hematology lab procedures','can you assist me with a cbc question','please help me with specimen handling','i need guidance on cbc workflow','help me understand the hematology process'],
    'cbc_info': ['what is a cbc','explain complete blood count','what is rbc count','what is wbc count','what does hemoglobin mean','what is hematocrit','what is mcv','what is rdw','why is a cbc ordered','what does a cbc include'],
    'sample_collection': ['which tube is used for cbc','how many inversions for edta tube','how should i collect a cbc specimen','what is the tube color for cbc','how do i avoid hemolysis for cbc','how long is cbc sample stable','what labeling step matters for cbc collection','what is the correct fill volume for edta'],
    'fallback': ['tell me about quantum bananas','blue sandwich','asdf qwerty','nonsense input','what is 7 plus purple'],
}


def generate_samples(intent: str, target: int) -> list[dict[str, str]]:
    templates = TEMPLATES[intent]
    samples = []
    index = 0
    while len(samples) < target:
        text = templates[index % len(templates)]
        samples.append({'text': text, 'intent': intent, 'lang': 'en', 'source': 'synthetic'})
        index += 1
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate bootstrap synthetic data')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument('--per-intent', type=int, default=60)
    parser.add_argument('--fallback-count', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    rows = []
    for intent in TEMPLATES:
        count = args.fallback_count if intent == 'fallback' else args.per_intent
        rows.extend(generate_samples(intent, count))
    random.shuffle(rows)
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f'{output_path} already exists. Use --overwrite to replace it.')
    write_jsonl(rows, output_path)
    print(f'Wrote {len(rows)} samples to {output_path}')


if __name__ == '__main__':
    main()
