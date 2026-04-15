from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from baselines.datasets import IntentDataset, build_vocab, collate_batch
from baselines.models import SequentialClassifier
from evaluation.io import create_run_dir, save_confusion_matrix, save_json
from evaluation.metrics import compute_classification_metrics
from preprocessing.dataset import build_label_maps, iter_jsonl
from preprocessing.splits import ensure_split_files


def _load_split_records(split_dir: Path) -> Dict[str, List[dict[str, str]]]:
    return {name: list(iter_jsonl(split_dir / f'{name}.jsonl')) for name in ('train', 'val', 'test')}


def _evaluate(model, loader, device):
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits = model(input_ids, lengths)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(1, len(all_labels))
    return avg_loss, all_labels, all_preds


def run_sequential_experiment(config: Dict[str, object]) -> Path:
    seed = int(config.get('seed', 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_cfg = config['data']
    split_dir = Path(data_cfg['splits_dir'])
    dataset_path = Path(data_cfg['processed_dataset_path'])
    ensure_split_files(
        dataset_path=dataset_path,
        splits_dir=split_dir,
        test_size=float(data_cfg['test_size']),
        val_size=float(data_cfg['val_size']),
        seed=seed,
    )
    splits = _load_split_records(split_dir)
    label2id, id2label = build_label_maps(splits['train'] + splits['val'] + splits['test'])
    label_names = [id2label[index] for index in range(len(id2label))]

    model_cfg = config['model']
    train_cfg = config['training']
    vocab = build_vocab(
        (row['text'] for row in splits['train']),
        min_freq=int(model_cfg.get('min_freq', 1)),
        max_vocab_size=int(model_cfg.get('max_vocab_size', 10000)),
    )

    datasets = {name: IntentDataset(records, vocab=vocab, label2id=label2id) for name, records in splits.items()}
    collate = partial(collate_batch, pad_id=vocab.pad_id)
    train_loader = DataLoader(datasets['train'], batch_size=int(train_cfg['train_batch_size']), shuffle=True, collate_fn=collate)
    val_loader = DataLoader(datasets['val'], batch_size=int(train_cfg['eval_batch_size']), shuffle=False, collate_fn=collate)
    test_loader = DataLoader(datasets['test'], batch_size=int(train_cfg['eval_batch_size']), shuffle=False, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SequentialClassifier(
        vocab_size=len(vocab.itos),
        embedding_dim=int(model_cfg['embedding_dim']),
        hidden_dim=int(model_cfg['hidden_dim']),
        num_classes=len(label2id),
        architecture=str(model_cfg['architecture']),
        num_layers=int(model_cfg.get('num_layers', 1)),
        dropout=float(model_cfg.get('dropout', 0.2)),
        bidirectional=bool(model_cfg.get('bidirectional', False)),
        padding_idx=vocab.pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg['learning_rate']), weight_decay=float(train_cfg.get('weight_decay', 0.0)))
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_f1 = -1.0
    history = []
    start = time.perf_counter()

    for epoch in range(1, int(train_cfg['epochs']) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / max(1, seen)
        val_loss, val_labels, val_preds = _evaluate(model, val_loader, device)
        val_metrics = compute_classification_metrics(val_labels, val_preds, label_names)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, **val_metrics['metrics']})
        if val_metrics['metrics']['f1_macro'] > best_f1:
            best_f1 = val_metrics['metrics']['f1_macro']
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_labels, test_preds = _evaluate(model, test_loader, device)
    artifact = compute_classification_metrics(test_labels, test_preds, label_names)
    run_dir = create_run_dir(Path(config['results_dir']), str(config['experiment_name']))
    torch.save({'state_dict': model.state_dict(), 'label2id': label2id, 'vocab': vocab.itos, 'config': config}, run_dir / 'model.pt')

    runtime = {
        'train_and_eval_seconds': time.perf_counter() - start,
        'device': str(device),
        'num_parameters': sum(parameter.numel() for parameter in model.parameters()),
        'test_loss': test_loss,
    }
    metadata = {
        'experiment_name': config['experiment_name'],
        'model_family': config['model']['family'],
        'model_architecture': config['model']['architecture'],
        'dataset_path': str(dataset_path),
        'splits_dir': str(split_dir),
        'label_names': label_names,
    }

    save_json({'metrics': artifact['metrics'], 'metadata': metadata}, run_dir / 'metrics.json')
    save_json(artifact['classification_report'], run_dir / 'classification_report.json')
    save_confusion_matrix(run_dir / 'confusion_matrix.csv', artifact['confusion_matrix'], label_names)
    save_json({'epochs': history}, run_dir / 'training_history.json')
    save_json(runtime, run_dir / 'runtime.json')
    save_json({'itos': vocab.itos}, run_dir / 'vocab.json')
    save_json(label2id, run_dir / 'label_map.json')
    return run_dir
