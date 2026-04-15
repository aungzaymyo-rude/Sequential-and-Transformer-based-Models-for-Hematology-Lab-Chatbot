from __future__ import annotations

import torch
from torch import nn


class SequentialClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        architecture: str = 'lstm',
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.GRU if architecture == 'gru' else nn.LSTM
        self.encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)
        self.architecture = architecture
        self.bidirectional = bidirectional

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        if self.architecture == 'lstm':
            hidden = hidden[0]
        if self.bidirectional:
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            features = hidden[-1]
        return self.classifier(self.dropout(features))
