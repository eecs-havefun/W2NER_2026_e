# AGENTS.md - Guidelines for Agentic Coding in W2NER × ProcNet

This document provides essential information for AI agents working on the W2NER codebase, which is being coupled with ProcNet for unified event/NER extraction.

## Project Overview

W2NER (Word-Word Relation Classification for Unified NER) is a PyTorch model for flat, overlapped, and discontinuous NER. Architecture: BERT + BiLSTM + 2D conv + biaffine classifier. This fork extends W2NER to export ProcNet-compatible typed entity structures via `utils.decode_for_procnet()` and `utils.build_prediction_record()`.

**Source repos**: Official W2NER at `official_W2NER/W2NER/`, official ProcNet at `official_procnet/procnet/`. Always compare changes against these.

## Environment Setup

- Python 3.8+, CUDA 11.4 (GPU training)
- `pip install -r requirements.txt`
- Key deps: torch==1.13.1, transformers==4.37.2, numpy==1.23.5, scikit-learn==1.3.2
- BERT model expected at `../models/bert_base_chinese`

## Build / Lint / Test Commands

### Training
```bash
python main.py --config config/<dataset>.json --device 0
```

### Single Dataset Smoke Test (1 epoch)
```bash
python main.py --config config/id_cards_with_queries.json --device 0
```
Edit the `epochs` field in the JSON config to control training length. Available configs: `flight_orders.json`, `hotel_orders.json`, `id_cards.json`, `mixed_data.json`, `train_orders.json`, and their `_with_queries` variants.

### Multi-Dataset Smoke Tests
```bash
python scripts_maybeuseful/smoke_test.py          # Prepare small subset from all 5 datasets
python main.py --config config/smoke_test.json    # Train on combined smoke data
python scripts_maybeuseful/run_5_smoke_tests.py   # Run per-dataset 1-epoch tests (creates symlinks + configs)
python scripts_maybeuseful/run_all_smoke_tests.py # Sequential smoke tests via shared config
```

### Data Pipeline Scripts
```bash
# Convert between data formats
python scripts_maybeuseful/regenerate_all_data_step1_v1b_to_procnet.py
python scripts_maybeuseful/regenerate_all_data_step2_procnet_to_w2ner.py
python scripts_maybeuseful/regenerate_all_data_step3_fold_types.py

# Validation and alignment checks
python scripts_maybeuseful/deep_check.py
python scripts_maybeuseful/verify_alignment.py
python scripts_maybeuseful/check_procnet_doc_ids.py
```

### Linting & Formatting (optional)
```bash
pip install black isort flake8
black . --exclude=./data
isort .
flake8 . --max-line-length=120
```

### Type Checking
```bash
mypy --ignore-missing-imports .
```

## Code Style Guidelines

### Imports
Order: standard library → third-party → local, separated by blank lines. No wildcard imports. Use absolute imports.
```python
import json
import os

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

import config
import data_loader
from model import Model
```

### Naming
- **Classes**: `CamelCase` (e.g., `LayerNorm`, `Vocabulary`, `RelationDataset`)
- **Functions/Methods**: `snake_case` (e.g., `get_logger`, `process_bert`, `decode_for_procnet`)
- **Variables/Attributes**: `snake_case` (e.g., `bert_inputs`, `grid_mask2d`, `sent_length`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `PAD`, `RANDOM_SEED`, `dis2idx`)
- **Private members**: prefix with `_`

### Formatting
- 4 spaces per indentation (no tabs)
- Max line length: 120 characters
- Break long lines after operators, before parentheses

### Types
- Type hints encouraged but not required (existing code is inconsistent)
- Use `typing` module: `List`, `Dict`, `Optional`, `Any`, `Tuple`, `Union`
- Use `np.int64` and `np.bool_` (NOT deprecated `np.int` / `np.bool`)

### Error Handling & Logging
- Use explicit `try-except`; never silently catch exceptions
- Log via `utils.get_logger(dataset)` — returns logger with file + stream handlers
- Use `assert` for internal invariants (e.g., batch alignment checks)
- Add `###<name>_checker` comments before critical assertions for grep-ability

### Documentation
- Docstrings (`"""`) for public functions/classes
- Inline comments: concise, English preferred (some Chinese comments exist)
- File headers: optional `#!/usr/bin/env python3` and `# -*- coding: utf-8 -*-`

## Architecture & Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point; `Trainer` class with `self.model`/`self.config` (not globals) |
| `model.py` | BERT + BiLSTM + 2D conv + biaffine classifier |
| `config.py` | `Config` class with type validation, `data_root`/`cache_dir` support |
| `data_loader.py` | `Vocabulary`, `RelationDataset`, `process_bert()`, `collate_fn()` |
| `utils.py` | Logging, serialization, `decode()`, `decode_for_procnet()`, `build_prediction_record()` |
| `config/*.json` | Per-dataset training configs |
| `scripts_maybeuseful/` | Data conversion, validation, smoke test orchestration |

## Data Conventions

- **Source data**: `data_v1b/` (in parent project dir)
- **W2NER data**: `data/data_w2ner_folded_with_dev/<dataset>/` (train.json, dev.json, test.json)
- **ProcNet format**: `procnet_format/<dataset>/` (in parent project dir)
- **Config `data_root`**: defaults to `./data`; dataset path = `{data_root}/{dataset}/`
- Each JSON record must have: `sentence` (list of tokens), `ner` (list of `{index, type}`)
- ProcNet coupling requires: `doc_id`, `sent_id` fields in records

## ProcNet Coupling Conventions

- `utils.decode_for_procnet()` returns scored entity dicts with `token_indices`, `type_id`, `score`, `head`
- `utils.build_prediction_record()` produces output with both `entity` (W2NER format) and `procnet_entities` (ProcNet format)
- `--continuous_only` flag (default=1): filters discontinuous entities from ProcNet export
- Output JSON fields: `doc_id`, `sent_id`, `sentence`, `entity`, `procnet_entities`
- ProcNet sidecar files: `sidecar_entities/{split}_typed_entities.jsonl`

## Git Conventions

- Commit messages: descriptive, start with a verb (e.g., "Fix decode graph traversal in utils.py")
- Do NOT commit: `*.pt`, `*.bin`, `*.pth`, `log/`, `cache/`, `output.json`, `data/`
- Exception: `data/data_w2ner_folded_with_dev/` is tracked
- Always verify changes against `official_W2NER/` and `official_procnet/` before committing

## Conversation History

Design decisions, data pipeline analysis, and cross-repo coupling notes are stored in:
`conversation_history/`

Key files:
- `2026-04-01_w2ner_to_procnet_data_conversion.md` — W2NER → ProcNet data format conversion design

## Known Issues & Workarounds

- `transformers==4.37.2` may conflict with other deps; pinning recommended
- Smoke tests assume GPU (`--device 0`); use `--device -1` for CPU
- `run_all_smoke_tests.py` has a bug: `Path` used before import (lines 3-4 precede shebang)
- Random seeds are commented out in `main.py`; uncomment for reproducible runs
- `Vocabulary.__len__` in official repo returns `len(self.token2id)` (nonexistent); our fork fixes this to `len(self.label2id)`
