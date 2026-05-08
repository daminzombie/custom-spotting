# custom-spotting

T‑DEED–style temporal action spotting for **broadcast / scene-level labels** using the same pipeline shape as [`custom-ballspotting`](../custom-ballspotting): clips, displacement target, `(2 × N + 1)`-class team head (`left` / `right` × your action set), and the same [`CustomTDeedModule`](custom_spotting/model/tdeed.py).

## Label taxonomy

Foreground classes come from [`custom_spotting/actions.py`](custom_spotting/actions.py):

| Label (`ground_truth.json` `label`) | Enum |
| --- | --- |
| `foul` | `foul` |
| `free_kick` | `free_kick` |
| `ball_out_of_play_clear` | `ball_out_of_play_clear` |
| `ball_out_of_play_distance` | `ball_out_of_play_distance` |

Each annotation may carry a **`team`** field (`left` / `right` / `not applicable`, same semantics as dudek/custom-ballspotting).

## Geometry defaults vs SoccerNet‑T‑DEED

| Setting | Typical T‑DEED SN action spotting | This package defaults |
| --- | --- | --- |
| Frame extract `stride` (inference fallback) | 12 (`STRIDE_SN` in upstream T‑DEED) | **6** |
| Temporal clip overlap | 50% sliding window typical | **`overlap`: 50** frames for `clip_frames_count=100` (50% overlap) |

Training uses **`accepted_gap`** (default **6**) when grouping contiguous extracted frames via [`VideoRecord.get_clips`](custom_spotting/data.py); align it with extraction stride.

## Install

From this directory:

```bash
pip install -e .
custom-spotting --help
```

Optional SoccerNet evaluator:

```bash
pip install -e '.[challenge]'
```

## CLI

Mirrors [`custom-ballspotting`](../custom-ballspotting/README.md): **`extract-frames`**, **`train`**, **`pretrain`**, **`posttrain`**, **`infer-video`**.

Use JSON examples under [`configs/`](configs/); [`configs/README.md`](configs/README.md) summarises overlaps and strides.
