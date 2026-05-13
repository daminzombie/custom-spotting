from enum import Enum
from typing import NamedTuple


class Action(str, Enum):
    FOUL = "foul"
    FREE_KICK = "free_kick"
    BALL_OUT_OF_PLAY_CLEAR = "ball_out_of_play_clear"
    BALL_OUT_OF_PLAY_DISTANT = "ball_out_of_play_distant"


# Legacy label spelling still seen in some exports / older clips.
_LABEL_ALIASES: dict[str, str] = {
    "ball_out_of_play_distance": Action.BALL_OUT_OF_PLAY_DISTANT.value,
}


def parse_action_label(label: str) -> Action:
    """Map a dataset label string to :class:`Action`, including legacy aliases."""
    normalized = _LABEL_ALIASES.get(label, label)
    return Action(normalized)


class ActionConfig(NamedTuple):
    #: Inference / post-processing scale (not used for training CE; see
    #: :data:`TRAINING_CE_RELATIVE_WEIGHTS`).
    weight: float
    min_score: float
    tolerance_seconds: float


ACTION_CONFIGS: dict[Action, ActionConfig] = {
    Action.FOUL: ActionConfig(7.7, 0.5, 2.5),
    Action.FREE_KICK: ActionConfig(1.41, 0.0, 1.5),
    Action.BALL_OUT_OF_PLAY_CLEAR: ActionConfig(3.1, 0.5, 2.0),
    Action.BALL_OUT_OF_PLAY_DISTANT: ActionConfig(2.9, 0.5, 2.0),
}

# Cross-entropy only: relative importance among actions (most frequent ≈ 1.0).
# Final CE weight for each foreground class is ``ce_foreground_scale * value``;
# background stays 1.0. Independent of :attr:`ActionConfig.weight`.
# Edges raised vs a flat prior so rare spot-like events are not washed out by background frames.
TRAINING_CE_RELATIVE_WEIGHTS: dict[Action, float] = {
    Action.FOUL: 5.0,
    Action.FREE_KICK: 1.5,
    Action.BALL_OUT_OF_PLAY_CLEAR: 3.5,
    Action.BALL_OUT_OF_PLAY_DISTANT: 3.5,
}

if len(TRAINING_CE_RELATIVE_WEIGHTS) != len(Action):
    raise RuntimeError(
        "TRAINING_CE_RELATIVE_WEIGHTS must define exactly one entry per Action enum member"
    )

ACTION_CLASS_INDEX: dict[str, int] = {
    action.value: idx for idx, action in enumerate(Action)
}
NUM_ACTION_CLASSES: int = len(ACTION_CLASS_INDEX)


def label_to_index(action: Action | str) -> int:
    """Return the model class index for an action (background = 0)."""
    if isinstance(action, str):
        action = parse_action_label(action)
    return ACTION_CLASS_INDEX[action.value] + 1  # 1-based


def index_to_label(index: int) -> Action | None:
    """Decode an action class index, or None for background/out of range."""
    if index <= 0 or index > NUM_ACTION_CLASSES:
        return None
    return list(Action)[index - 1]
