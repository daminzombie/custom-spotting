# custom-spotting

`custom-spotting` is a reusable Python package for broadcast / scene-level **action spotting** with a fixed custom label set. It mirrors sibling [`custom-ballspotting`](../custom-ballspotting): RegNet + temporal shift backbone, SGP-Mixer temporal stack, displacement auxiliary loss, class weighting, clip sampling, and the same dataset + CLI shape. It predicts only action classes:

```text
background + N actions   →   N + 1 classes
```

Here **`N = 4`**, so the head has **5** classes. Labels are defined only in `custom_spotting/actions.py` (do not expect SoccerNet’s full 17-class action vocabulary in this repo).

## Package layout

```text
custom_spotting/
  actions.py       # four custom actions, CE weights, inference scales
  data.py          # clip folders, extraction, VideoClip / dataset
  training.py      # TrainConfig, train_from_dataset, train_model
  inference.py     # infer_video, score_video, decoding + NMS
  cli.py           # command-line entry
  eval.py          # validation mAP (optional SoccerNet tooling)
  map_scoring.py   # dudek-style fusion + soft-NMS helper
  soccernet_challenge_eval.py  # optional zip mAP (needs [challenge])
  model/
    tdeed.py
    layers.py
    shift.py
```

Library-first, CLI-second:

```python
from custom_spotting.training import TrainConfig, train_from_dataset
from custom_spotting.inference import infer_video
```

```bash
custom-spotting --help
```

## Defaults vs `custom-ballspotting` / T-DEED SoccerNet

| Idea | `custom-ballspotting` (typical) | `custom-spotting` (this repo) |
|------|----------------------------------|-------------------------------|
| Label focus | Ball-centric actions (many classes) | Four broadcast-style events |
| Train / infer frame `stride` fallback | often `2` | **`6`** (denser than T-DEED SN `12`, sparser than 2) |
| `TrainConfig.overlap` (100-frame clips) | `88` (ball / T‑DEED ball preset) | **`50`** (**50%** overlap) |
| `TrainConfig.accepted_gap` | (not exposed; clips used gap `2`) | **`6`** — must match extraction stride |

## Install

From this repo root:

```bash
pip install -e .
```

Optional SoccerNet evaluation helpers:

```bash
pip install -e ".[challenge]"
```

```bash
custom-spotting --help
```

## Action vocabulary (`actions.py`)

```python
class Action(str, Enum):
    FOUL = "foul"
    FREE_KICK = "free_kick"
    BALL_OUT_OF_PLAY_CLEAR = "ball_out_of_play_clear"
    BALL_OUT_OF_PLAY_DISTANCE = "ball_out_of_play_distance"
```

Input annotations are action-only: each event has a label and timestamp.

## Dataset layout

Point **`dataset_root`** at the root of your clip-folder tree (paths in JSON resolve relative to the config file).

The loader discovers folders containing **`ground_truth.json`** or optional **`Labels-ball.json`** (filename kept for parity with BAS tooling). Each clip folder uses the lexicographically first **`*.mp4`**.

### `ground_truth.json`

| Field | Meaning |
|------|---------|
| **`label`** | Must match **`Action.value`** (`foul`, `free_kick`, …). |
| **`position`** | Event time in **milliseconds** from the start of that video file. |

```json
{
  "annotations": [
    { "label": "foul", "position": 120000 },
    { "label": "free_kick", "position": 245500 }
  ]
}
```

Unknown labels are skipped with a one-time warning.

## Frame extraction

```bash
custom-spotting extract-frames --config configs/extract_frames.example.json
```

Override from CLI as needed (`--stride`, `--frame_target_width`, …). Default CLI stride when JSON omits it is **6**.

## Config files

`extract-frames`, `train`, `pretrain`, and `posttrain` expect **`--config <json>`**. **`infer-video`** may use **`--config`** or explicit `--video_path` / `--video_dir`.

See **`configs/README.md`** and the JSON files beside it. Typical flow:

```text
extract_frames.example.json
final_posttrain_from_tdeed.example.json
inference.example.json
```

(and `*_720p` / `_224` aliases).

Training defaults (**`TrainConfig`**): **`run_validation`**, **`eval_metric`** (`map` or `loss`), internal **`map_mine`**, optional SoccerNet **`mAPevaluateTest`** when **`val_run_soccernet_challenge_map`** + **`soccernet_path`** are set (needs **`pip install -e ".[challenge]"`**).

Logging matches ballspotting (TensorBoard **`runs/`**, **`epoch_summary.log`**, throttled plain logs without TTY).

## Training presets

Default clip geometry shipped in code:

```json
{
  "clip_frames_count": 100,
  "overlap": 50,
  "accepted_gap": 6
}
```

Typical backbone init is still a T-DEED / RegNet checkpoint via **`posttrain`** and **`model.load_backbone()`** — use matching **`features_model_name`** (`regnety_008` for common SoccerNetBall releases, **`regnety_002`** only if the checkpoint width matches).

```bash
custom-spotting posttrain --config configs/final_posttrain_from_tdeed.example.json
```

Only **`_features.*`** and **`_temp_fine.*`** weights transfer; the action-only **`N+1`** head is trained fresh for **`N = 4`**. Legacy custom-spotting checkpoints with incompatible heads are not `load_all()` compatible with this action-only model.

### Sparse / rare events (most sampled windows are background)

Events are **much rarer** than “ball in play” labels, so defaults bias training toward positives and strengthen foreground CE:

| Knob | Role |
|------|------|
| **`even_choice_proba`** (default **`0.35`**) | Often sample a clip that already contains a labeled event; still sometimes sample any clip for background diversity. |
| **`ce_foreground_scale`** (default **`6.0`**) | Scales all non-background CE weights vs class 0. |
| **`TRAINING_CE_RELATIVE_WEIGHTS`** in `actions.py` | Per-class relative multipliers (raised for fouls / ball-out). |
| Extract **`radius_seconds`** (examples use **12**) | When `save_all` is false, widens the temporal band around each annotation so stride-6 stores still land near events. |
| **`enforce_train_epoch_size`** | Optional fixed number of train steps per epoch (`pretrain.example.json`, **`train_quick_sparse_iteration.example.json`**). |

CLI overrides: `--even_choice_proba`, `--ce_foreground_scale`.

Details and tuning notes: **`configs/README.md`** (section *Rare / sparse labels*).

## Checkpoints

`checkpoints/` and `runs/` are **gitignored**. Best weights are saved beside **`*.metadata.json`** (training config snapshot, **`head_type: "action_only"`**, **`num_action_classes`**, etc.). Inference refuses mismatched enums or incompatible metadata.

## Inference

Prefer a config file that points at your **`model_checkpoint_path`**. Metadata fills omitted architecture / clip settings.

```bash
custom-spotting infer-video --config configs/inference.example.json
```

Direct invocation:

```bash
custom-spotting infer-video \
  --video_path="../data/videos/sample.mp4" \
  --model_checkpoint_path="../checkpoints/your_run_best.pt" \
  --output_path="../predictions/sample.json" \
  --stride=6 \
  --overlap=50 \
  --clip_frames_count=100
```

Output schema:

```json
{
  "video_path": "videos/sample.mp4",
  "fps": 25.0,
  "predictions": [
    {
      "label": "foul",
      "position": 120000,
      "gameTime": "1 - 02:00",
      "confidence": 0.62
    }
  ]
}
```

A tiny illustration lives at **`predictions/sample_predictions.example.json`**.

## Python API

```python
from custom_spotting.training import TrainConfig, train_from_dataset

config = TrainConfig(
    overlap=50,
    accepted_gap=6,
    clip_frames_count=100,
    even_choice_proba=0.35,
    ce_foreground_scale=6.0,
)

train_from_dataset(
    save_as="checkpoints/{experiment_name}_{timestamp}_best.pt",
    dataset_root="data/custom/dataset",
    pretrained_checkpoint_path="checkpoints/sn_ball_tdeed_best.pt",
    experiment_name="my_spotting_run",
    config=config,
)
```

```python
from custom_spotting.inference import infer_video

result = infer_video(
    video_path="videos/sample.mp4",
    model_checkpoint_path="checkpoints/my_spotting_run_YYYYMMDD_HHMMSS_best.pt",
    output_path="predictions/out.json",
)
```

## Scripts

- **`scripts/run-posttrain-pm2.sh`** — run **`final_posttrain_from_tdeed.example.json`** via PM2 (see header comments).

## Recommended workflow

1. Confirm `custom_spotting/actions.py` matches your JSON labels (four strings above).
2. Lay out clip folders with `ground_truth.json` + one `*.mp4` each.
3. `custom-spotting extract-frames --config configs/extract_frames.example.json`
4. `custom-spotting posttrain --config configs/final_posttrain_from_tdeed.example.json`
5. `custom-spotting infer-video --config configs/inference.example.json`
6. Tune `ActionConfig`, decode thresholds (`inference.py` defaults), and augmentation JSON fields.

For formatting / checks, see **`CONTRIBUTING.md`**. For release notes, **`CHANGELOG.md`**.
