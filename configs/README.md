# Example Configs (`custom-spotting`)

Workflow mirrors [`custom-ballspotting`](../custom-ballspotting) but custom-spotting is **action-only**: model output is `background + N actions`. Geometry defaults target **stride 6** extraction, **`overlap`: 50** (half overlap on 100-frame clips), and **`accepted_gap`: 6** during training clip grouping (`TrainConfig`). **Architecture defaults** match T-DEED SoccerNet **`rny002_gsf`** / ``SoccerNet_small.json`` (**``regnety_002``**, **`n_layers` 3**, **`sgp_ks` 9**, **`sgp_k` 4**, **`displacement_radius` 3**); use **`sgp_ks` 11** with **`regnety_008`** (``SoccerNet_big.json``). See **`custom_spotting/actions.py`** for the four label strings.

Primary 720p-style workflow:

- `extract_frames.example.json`
- `final_posttrain_from_tdeed.example.json`
- `inference.example.json`

Explicit aliases:

- `extract_frames_720p.example.json`
- `final_posttrain_from_tdeed_720p.example.json`
- `inference_720p.example.json`

Low-resolution smoke path:

- `extract_frames_224.example.json`
- `inference_224.example.json`

Typical framing is **1280×720** (or your chosen width/height) with `train_batch_size` / `val_batch_size` **1** and **`acc_grad_iter`: 8** in longer runs — matching the memory pattern from sibling configs.

Experimental / staging:

- `pretrain.example.json` — train from scratch; includes **`enforce_train_epoch_size`: 6144** for sparse quick loops.
- `posttrain_from_tdeed.example.json` — alternate posttrain template paths.
- `posttrain_from_custom.example.json` — full-checkpoint continuation if shapes match your run; legacy incompatible custom-spotting heads are not compatible with the action-only `N+1` head.
- `posttrain_soccernet_challenge.example.json` — SoccerNet-style validation wiring when applicable.
- `train_quick_sparse_iteration.example.json` — stronger sparse-label defaults (`even_choice_proba` 0.45, enforced steps).

Adjust **`pretrained_checkpoint_path`** and **`features_model_name`** so RegNet width matches the weight file (`regnety_008` for common SoccerNetBall T‑DEED releases).

For day-to-day work, start from **`final_posttrain_from_tdeed.example.json`** after editing **`dataset_root`** and checkpoint URLs.

## Rare / sparse labels (most windows are background-only)

These repos target **infrequent** broadcast events. Without extra care, random clip sampling mostly sees class **0** (background).

1. **Foreground-biased clip sampling** — `TrainConfig.even_choice_proba` (**default `0.35` in code**, also set in JSON): with that probability, each training step draws a clip that contains **at least one** foreground annotation; otherwise a uniform random clip (`CustomTDeedDataset`).
2. **Stronger foreground CE** — `ce_foreground_scale` (**default `6.0`** in code) multiplied by **`TRAINING_CE_RELATIVE_WEIGHTS`** in `actions.py` (tuned upward for fouls / ball-out classes).
3. **Extraction** — Example extract configs use **`radius_seconds`: 12** (with `save_all: false`): keeps **stride** frames **plus** a temporal neighborhood around annotated times so positives land inside 100-frame windows more often. Increase further if extracts still miss labels.
4. **Fixed steps per epoch** — `enforce_train_epoch_size` / `enforce_val_epoch_size` (see `pretrain.example.json` **6144** and **`train_quick_sparse_iteration.example.json`**) stabilize iteration count when clip lists are tiny; omit those keys for a plain full pass over all clips each epoch.

Aggressive iteration preset:

- `train_quick_sparse_iteration.example.json` — higher `even_choice_proba` (**0.45**), higher `ce_foreground_scale` (**7.0**), enforced train/val steps.

Tune **down** `even_choice_proba` if the model pushes too many false positives (not enough exposure to pure background).
