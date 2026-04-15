# Medical Intent Classification Study

Paper-focused project for comparing sequential neural networks and Transformer models on medical intent classification.

## Scope
This repository is for reproducible ML experiments. It is not the chatbot application repo.

## Layout
- `data/raw`: raw source material
- `data/labeled`: manually labeled training files
- `data/processed`: merged training dataset
- `data/splits`: fixed train/validation/test splits
- `preprocessing`: cleaning, dataset IO, split generation
- `baselines`: LSTM, BiLSTM, and GRU baselines
- `transformer_models`: Transformer baseline training
- `experiments`: config loading and experiment runner
- `evaluation`: metrics and run artifact writers
- `results`: saved experiment outputs
- `paper`: paper notes, figures, and tables
- `scripts`: dataset preparation entrypoints

## Workflow
1. Put labeled CSV/JSONL files into `data/labeled`.
2. Optionally generate synthetic bootstrap data.
3. Merge data into one processed dataset.
4. Create fixed train/validation/test splits.
5. Run comparable experiments from the same split files.
6. Use `results/` artifacts for paper tables and figures.

## Quick Start
```bash
python scripts/generate_synthetic_dataset.py --per-intent 80 --fallback-count 20 --output data/processed/bootstrap.jsonl --overwrite
python scripts/merge_labeled_data.py --base data/processed/bootstrap.jsonl --external-dir data/labeled --output data/processed/intent_dataset.jsonl --overwrite
python scripts/create_splits.py --dataset data/processed/intent_dataset.jsonl --output-dir data/splits --overwrite
python experiments/run_experiment.py --config configs/bert.yaml
python experiments/run_experiment.py --config configs/lstm.yaml
```

## Expected Dataset Format
CSV or JSONL rows should use:
- `text`
- `intent`
- optional `lang` (defaults to `en`)
- optional `source`

## Current Seed Data
A bootstrap labeled file is copied from the prototype into `data/labeled/bootstrap_cbc_real_phrases_400.csv`.
