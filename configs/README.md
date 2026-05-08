# Example Configs (custom-spotting)

Training defaults match **overlap 50** (half overlap on `clip_frames_count=100` clips),
**accepted_gap / extract stride 6**, and **`custom-spotting` four-class taxonomy** (see
`custom_spotting/actions.py`). Adjust paths and checkpoint names locally.

Primary 720p workflow:

- `extract_frames.example.json`
- `final_posttrain_from_tdeed.example.json`
- `inference.example.json`

Equivalent explicit 720p aliases:

- `extract_frames_720p.example.json`
- `final_posttrain_from_tdeed_720p.example.json`
- `inference_720p.example.json`

This package defaults to **1280x720 frames** (`clip_frames_count=100`), **overlap=50**, and dense-enough extracts (`stride` **6**) unless overridden in JSON.
They use batch size 1 plus gradient accumulation because 720p frames are much
heavier than low-resolution crops.

Advanced/experimental configs:

- `pretrain.example.json`: only use if your source data already uses the final custom labels.
- `posttrain_from_tdeed.example.json`: older generic example kept for reference.
- `posttrain_from_custom.example.json`: only useful after adding explicit full-checkpoint resume or staged fine-tuning behavior.

For the current product plan, use `final_posttrain_from_tdeed.example.json`.
