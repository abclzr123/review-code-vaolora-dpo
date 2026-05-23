# Value-Aligned Orthogonal Low-Rank Adaptation for Preference Optimization

This repository contains the reviewer-facing code release for a parameter-efficient post-training study built on `LoRA + DPO`. The release includes:

- the main training implementation for baseline, value-alignment, and orthogonal variants
- value-anchor construction code
- evaluation scripts for preference pairs and safety-style prompt generations
- sanitized example configs that use placeholder paths

This reviewer release intentionally excludes paper-writing assets, plotting utilities, generated figures, local experiment logs, checkpoints, and machine-specific deployment helpers.

## Repository Layout

```text
.
├── configs/
├── scripts/
└── src/
```

- `src/train_dpo.py`: main DPO training entry with value-alignment and orthogonal regularization switches
- `src/eval_preference_pairs.py`: offline pairwise reward-margin evaluation
- `src/eval_safety_generations.py`: generation-based safety-style evaluation helper
- `scripts/build_value_anchors.py`: precompute value-direction anchors from a preference dataset
- `configs/*.example.yaml`: sanitized configs with placeholder paths

## Environment

Install the minimal dependencies with:

```bash
pip install -r requirements.txt
```

The code was developed around the versions listed in `requirements.txt`. Flash attention is optional; the evaluation scripts already fall back to the default attention implementation when it is unavailable.

## Data and Models

The configs are written to accept either:

- a local dataset path via `data.dataset_path`, or
- a Hugging Face dataset via `data.dataset_repo`

The default example points to `trl-lib/ultrafeedback_binarized` because it is publicly available and matches the training format expected by the code.

You should replace the following placeholders before running:

- `/path/to/base-model`
- `/path/to/local-dataset`
- `/path/to/value-anchor.pt`
- `/path/to/logs`, `/path/to/checkpoints`, `/path/to/eval`

## Example Workflow

1. Prepare or download a causal language model checkpoint.
2. Edit one of the example configs under `configs/`.
3. Optionally build a value anchor:

```bash
python scripts/build_value_anchors.py \
  --config configs/config.value_alignment.example.yaml \
  --output-path artifacts/value_anchor.pt
```

4. Train:

```bash
python src/train_dpo.py --config configs/config.value_alignment.example.yaml
```

5. Evaluate preference pairs:

```bash
python src/eval_preference_pairs.py \
  --config configs/config.value_alignment.example.yaml \
  --adapter-path /path/to/adapter \
  --output-dir artifacts/pair_eval
```

6. Evaluate safety-style generations:

```bash
python src/eval_safety_generations.py \
  --config configs/config.value_alignment.example.yaml \
  --adapter-path /path/to/adapter \
  --output-dir artifacts/safety_eval \
  --dataset-path /path/to/safety_prompts.jsonl
```

## Notes For Reviewers

- The repository is intentionally minimal and focused on reproduction of the main training/evaluation pipeline.
- Plotting and paper-specific packaging code are not included in this release.
- Paths in the example configs are placeholders and should be adapted to the local environment.
