from enum import Enum
from typing import NamedTuple


class Team(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    NOT_APPLICABLE = "not applicable"

    def flip(self) -> "Team":
        if self == Team.LEFT:
            return Team.RIGHT
        if self == Team.RIGHT:
            return Team.LEFT
        return Team.NOT_APPLICABLE


def parse_team_string(raw: str | None) -> Team:
    """Parse a dataset ``team`` field into :class:`Team`.

    Accepts enum values (``left`` / ``right`` / ``not applicable``), common
    variants such as ``not_applicable`` or ``n/a``, and falls back to
    ``Team.LEFT`` for missing or unrecognised values (same behaviour as the
    previous try/except default).
    """
    if raw is None:
        return Team.LEFT
    s = str(raw).strip()
    if not s:
        return Team.LEFT
    lower = s.lower()
    if lower == "n/a":
        return Team.NOT_APPLICABLE
    normalized = lower.replace("_", " ")
    try:
        return Team(normalized)
    except ValueError:
        return Team.LEFT


class Action(str, Enum):
    FOUL = "foul"
    FREE_KICK = "free_kick"
    BALL_OUT_OF_PLAY_CLEAR = "ball_out_of_play_clear"
    BALL_OUT_OF_PLAY_DISTANCE = "ball_out_of_play_distance"


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
    Action.BALL_OUT_OF_PLAY_DISTANCE: ActionConfig(2.9, 0.5, 2.0),
}

# Cross-entropy only: relative importance among actions (most frequent ≈ 1.0).
# Final CE weight for each foreground class is ``ce_foreground_scale * value``;
# background stays 1.0. Independent of :attr:`ActionConfig.weight`.
TRAINING_CE_RELATIVE_WEIGHTS: dict[Action, float] = {
    Action.FOUL: 4.0,
    Action.FREE_KICK: 1.2,
    Action.BALL_OUT_OF_PLAY_CLEAR: 2.5,
    Action.BALL_OUT_OF_PLAY_DISTANCE: 2.5,
}

if len(TRAINING_CE_RELATIVE_WEIGHTS) != len(Action):
    raise RuntimeError(
        "TRAINING_CE_RELATIVE_WEIGHTS must define exactly one entry per Action enum member"
    )

ACTION_CLASS_INDEX: dict[str, int] = {
    action.value: idx for idx, action in enumerate(Action)
}
NUM_ACTION_CLASSES: int = len(ACTION_CLASS_INDEX)
# Total foreground classes = N actions × 2 teams; head output = 2*N + 1 (incl. background)
NUM_TEAM_ACTION_CLASSES: int = 2 * NUM_ACTION_CLASSES


def label_to_index(action: Action | str, team: Team = Team.LEFT) -> int:
    """Return the model class index for a (action, team) pair.

    Layout (background = 0):
      indices 1 .. N          → LEFT  team, actions[0..N-1]
      indices N+1 .. 2*N      → RIGHT team, actions[0..N-1]
    """
    action = Action(action)
    base = ACTION_CLASS_INDEX[action.value] + 1  # 1-based
    if team == Team.RIGHT:
        base += NUM_ACTION_CLASSES
    return base


def index_to_label(index: int) -> tuple[Action, Team] | None:
    """Decode a class index back to (Action, Team), or None for background (0)."""
    if index == 0:
        return None
    actions = list(Action)
    if index <= NUM_ACTION_CLASSES:
        return actions[index - 1], Team.LEFT
    right_index = index - NUM_ACTION_CLASSES
    if right_index <= NUM_ACTION_CLASSES:
        return actions[right_index - 1], Team.RIGHT
    return None
