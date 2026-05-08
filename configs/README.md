# Example Configs (`custom-spotting`)

Workflow mirrors [`custom-ballspotting`](../custom-ballspotting) but geometry defaults target **stride 6** extraction, **`overlap`: 50** (half overlap on 100‑frame clips), and **`accepted_gap`: 6** during training clip grouping (`TrainConfig`). See **`custom_spotting/actions.py`** for the four label strings.

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

- `pretrain.example.json` — train from labels only (`load_backbone` optional depending on JSON).
- `posttrain_from_tdeed.example.json` — alternate posttrain template paths.
- `posttrain_from_custom.example.json` — full-checkpoint continuation if shapes match your run.
- `posttrain_soccernet_challenge.example.json` — SoccerNet-style validation wiring when applicable.

Adjust **`pretrained_checkpoint_path`** and **`features_model_name`** so RegNet width matches the weight file (`regnety_008` for common SoccerNetBall T‑DEED releases).

For day-to-day work, start from **`final_posttrain_from_tdeed.example.json`** after editing **`dataset_root`** and checkpoint URLs.
