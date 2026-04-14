# Medical Intent Classification Study

Paper-focused project scaffold for comparing sequential and Transformer models on medical intent classification.

## Purpose
This project is organized for research experiments, reproducibility, and paper writing.
It is separate from the chatbot application prototype.

## Structure
- `data/raw` : raw source material
- `data/labeled` : manually labeled intent data
- `data/processed` : cleaned datasets
- `data/splits` : fixed train/val/test splits
- `preprocessing` : text cleaning, tokenization, label encoding
- `baselines` : LSTM, BiLSTM, GRU implementations
- `transformers` : BERT/RoBERTa-based classifiers
- `experiments` : unified runners and configs
- `evaluation` : metrics, reports, comparison tools
- `results/metrics` : experiment metric outputs
- `results/figures` : exported figures
- `results/tables` : generated comparison tables
- `paper/drafts` : paper draft files
- `paper/figures` : figures used in the paper
- `paper/tables` : tables used in the paper
- `scripts` : helper scripts for data and experiments
- `configs` : experiment configs

## Next Implementation Steps
1. Freeze dataset format and create split generation script.
2. Port current preprocessing and Transformer pipeline from the prototype.
3. Add sequential baselines under `baselines`.
4. Create a unified experiment runner.
5. Save all outputs in `results` with reproducible metadata.
6. Write paper notes and result summaries under `paper`.
