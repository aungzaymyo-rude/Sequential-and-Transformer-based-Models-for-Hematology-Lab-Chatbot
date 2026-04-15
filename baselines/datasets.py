from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from preprocessing.text import tokenize_basic

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokenize_basic(text)] or [self.unk_id]


class IntentDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, str]], vocab: Vocabulary, label2id: Dict[str, int]):
        self.records = list(records)
        self.vocab = vocab
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        input_ids = torch.tensor(self.vocab.encode(row['text']), dtype=torch.long)
        label = torch.tensor(self.label2id[row['intent']], dtype=torch.long)
        return input_ids, label


def build_vocab(texts: Iterable[str], min_freq: int = 1, max_vocab_size: int | None = None) -> Vocabulary:
    counter = Counter()
    for text in texts:
        counter.update(tokenize_basic(text))
    tokens = [token for token, freq in counter.items() if freq >= min_freq]
    tokens.sort(key=lambda token: (-counter[token], token))
    if max_vocab_size is not None:
        tokens = tokens[: max(0, max_vocab_size - 2)]
    itos = [PAD_TOKEN, UNK_TOKEN, *tokens]
    stoi = {token: index for index, token in enumerate(itos)}
    return Vocabulary(stoi=stoi, itos=itos)


def collate_batch(batch, pad_id: int):
    inputs, labels = zip(*batch)
    lengths = torch.tensor([len(item) for item in inputs], dtype=torch.long)
    padded = pad_sequence(inputs, batch_first=True, padding_value=pad_id)
    return padded, lengths, torch.stack(labels)
