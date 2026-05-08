# Contributing

## Development Setup

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

If you share a `uv`-based workflow with other repos under `bas/`:

```bash
uv pip install -e . --python ../dude.k/.venv/Scripts/python.exe
```

## Local Checks

```bash
python -m compileall custom_spotting
```

When dev dependencies are installed:

```bash
ruff check .
ruff format --check .
```

## Repository Hygiene

- Do not commit datasets, extracted frames, checkpoints, raw prediction dumps, or videos.
- Tracked inference examples live under `predictions/*.example.json`; other `predictions/*.json` are ignored.
- Keep CLI commands thin; reusable logic stays in package modules.
- Prefer JSON configs for reproducible training and inference runs.
- Keep labels and training weights in `custom_spotting/actions.py`; pair updates with **`even_choice_proba`** / **`ce_foreground_scale`** in JSON when label density changes.

## Checkpoint Compatibility

When using `load_backbone()` from T-DEED / other RegNet checkpoints, align:

- `features_model_name`
- `temporal_shift_mode`
- `clip_frames_count`
- `n_layers`
- `sgp_ks`
- `sgp_k`

The classifier head must match **`NUM_ACTION_CLASSES`** for this repo (currently four actions → **nine** logits including background).
