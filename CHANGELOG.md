# Changelog

Notable changes are listed here using a lightweight [Keep a Changelog](https://keepachangelog.com/)-style format.

## Unreleased

### Changed

- `custom-spotting` is now action-only: model head, training targets, inference, and mAP use `N+1` classes (`background + actions`) and ignore dataset team fields.
- Checkpoint metadata now records `head_type: "action_only"`; older team-aware `2*N+1` full checkpoints are not load-compatible with the new head.

## [0.1.0] - 2026-05-07

### Added

- `custom-spotting` package: T-DEED–style temporal model with `(2*N+1)`-class **team** head for **broadcast / scene-level** custom labels (`N = 4` foreground actions).
- Same layout as sibling [`custom-ballspotting`](../custom-ballspotting): dataset discovery (`dataset_root`, `ground_truth.json` / optional `Labels-ball.json`), frame extraction, clip building (`overlap`, **`accepted_gap`** aligned with extraction **`stride`**), training / pretrain / posttrain / inference CLI.
- Defaults tuned for denser tiling than SoccerNet T-DEED action stride 12: extract **`stride = 6`**, **`overlap = 50`** frames for `clip_frames_count = 100` (half overlap).
- Example configs under `configs/`, optional SoccerNet challenge mAP path (`pip install -e ".[challenge]"`).
- `scripts/run-posttrain-pm2.sh` helper for long posttrain jobs.

### Changed

- **Sparse-label training defaults:** `TrainConfig.even_choice_proba` **0.35**, `ce_foreground_scale` **6.0**; higher `TRAINING_CE_RELATIVE_WEIGHTS` spread in `actions.py`.
- Extract example configs **`radius_seconds` 12**; training JSON presets set `even_choice_proba`, `ce_foreground_scale`; **`posttrain_from_custom.example.json`** aligned to overlap 50 / batch 1; added **`train_quick_sparse_iteration.example.json`**.
- CLI: **`--ce-foreground-scale`**.
