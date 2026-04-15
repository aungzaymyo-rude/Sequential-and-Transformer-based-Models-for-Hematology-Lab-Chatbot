from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

from evaluation.io import create_run_dir, save_confusion_matrix, save_json
from evaluation.metrics import compute_classification_metrics
from preprocessing.dataset import build_label_maps, iter_jsonl
from preprocessing.splits import ensure_split_files
from preprocessing.text import normalize_text


def _records_to_dataset(records: List[dict[str, str]]) -> Dataset:
    normalized = [{**row, 'text': normalize_text(row['text'])} for row in records]
    return Dataset.from_list(normalized)


def _load_split_records(split_dir: Path) -> Dict[str, List[dict[str, str]]]:
    return {name: list(iter_jsonl(split_dir / f'{name}.jsonl')) for name in ('train', 'val', 'test')}


def run_transformer_experiment(config: Dict[str, object]) -> Path:
    seed = int(config.get('seed', 42))
    set_seed(seed)

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
    split_records = _load_split_records(split_dir)
    label2id, id2label = build_label_maps(split_records['train'] + split_records['val'] + split_records['test'])
    label_names = [id2label[index] for index in range(len(id2label))]

    model_cfg = config['model']
    train_cfg = config['training']
    tokenizer = AutoTokenizer.from_pretrained(model_cfg['architecture'])
    max_length = int(model_cfg.get('max_length', 128))

    def tokenize_batch(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)

    datasets = {}
    for split_name, records in split_records.items():
        dataset = _records_to_dataset(records)
        dataset = dataset.map(tokenize_batch, batched=True)
        dataset = dataset.map(lambda row: {'label': label2id[row['intent']]})
        dataset = dataset.remove_columns([column for column in dataset.column_names if column in {'text', 'intent', 'lang', 'source'}])
        datasets[split_name] = dataset

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg['architecture'],
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    run_dir = create_run_dir(Path(config['results_dir']), str(config['experiment_name']))
    model_dir = run_dir / 'model'
    start = time.perf_counter()

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_classification_metrics(labels.tolist(), preds.tolist(), label_names)['metrics']

    training_kwargs = {
        'output_dir': str(model_dir),
        'evaluation_strategy': 'epoch',
        'save_strategy': 'epoch',
        'learning_rate': float(train_cfg['learning_rate']),
        'per_device_train_batch_size': int(train_cfg['train_batch_size']),
        'per_device_eval_batch_size': int(train_cfg['eval_batch_size']),
        'num_train_epochs': int(train_cfg['epochs']),
        'weight_decay': float(train_cfg.get('weight_decay', 0.0)),
        'warmup_steps': int(train_cfg.get('warmup_steps', 0)),
        'warmup_ratio': float(train_cfg.get('warmup_ratio', 0.0)),
        'load_best_model_at_end': True,
        'metric_for_best_model': 'f1_macro',
        'logging_steps': 10,
        'save_total_limit': 1,
        'report_to': [],
    }
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    if 'evaluation_strategy' not in params and 'eval_strategy' in params:
        training_kwargs['eval_strategy'] = training_kwargs.pop('evaluation_strategy')
    filtered_kwargs = {key: value for key, value in training_kwargs.items() if key in params}
    training_args = TrainingArguments(**filtered_kwargs)

    trainer_kwargs = {
        'model': model,
        'args': training_args,
        'train_dataset': datasets['train'],
        'eval_dataset': datasets['val'],
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }
    trainer_sig = inspect.signature(Trainer.__init__)
    trainer = Trainer(**{key: value for key, value in trainer_kwargs.items() if key in set(trainer_sig.parameters.keys())})
    trainer.train()
    predictions = trainer.predict(datasets['test'])
    preds = np.argmax(predictions.predictions, axis=-1).tolist()
    labels = predictions.label_ids.tolist()
    artifact = compute_classification_metrics(labels, preds, label_names)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    save_json(label2id, run_dir / 'label_map.json')

    runtime = {
        'train_and_eval_seconds': time.perf_counter() - start,
        'device': str(training_args.device),
        'num_parameters': sum(parameter.numel() for parameter in model.parameters()),
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
    save_json(runtime, run_dir / 'runtime.json')
    return run_dir
