# Changelog

Notable changes are listed here using a lightweight [Keep a Changelog](https://keepachangelog.com/)-style format.

## [0.1.0] - 2026-05-07

### Added

- `custom-spotting` package: T-DEED–style temporal model with `(2*N+1)`-class **team** head for **broadcast / scene-level** custom labels (`N = 4` foreground actions).
- Same layout as sibling [`custom-ballspotting`](../custom-ballspotting): dataset discovery (`dataset_root`, `ground_truth.json` / optional `Labels-ball.json`), frame extraction, clip building (`overlap`, **`accepted_gap`** aligned with extraction **`stride`**), training / pretrain / posttrain / inference CLI.
- Defaults tuned for denser tiling than SoccerNet T-DEED action stride 12: extract **`stride = 6`**, **`overlap = 50`** frames for `clip_frames_count = 100` (half overlap).
- Example configs under `configs/`, optional SoccerNet challenge mAP path (`pip install -e ".[challenge]"`).
- `scripts/run-posttrain-pm2.sh` helper for long posttrain jobs.
