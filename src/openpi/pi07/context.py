"""Validated textual conditioning and the dropout recipe in pi0.7, sections V/VII.

No learned annotator is supplied by the paper: callers provide ground-truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import math
import numbers
from typing import Any

import numpy as np


def scalar(value: Any) -> Any:
    """Unwrap scalar numpy values, without silently reducing arrays."""
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise TypeError("Expected a scalar, not an array.")
        return value.item()
    return value.item() if isinstance(value, np.generic) else value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    value = scalar(value)
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _text(value: Any, name: str, *, allow_empty: bool = False) -> str:
    value = scalar(value)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError(f"{name} cannot be empty.")
    return value


def bin_episode_steps(steps: float) -> int:
    """Nearest 500-step bin; exact halfway values round upward, not to even.

    The paper leaves tie handling unspecified. Half-up is an explicit convention
    of this implementation. Real values permit a linearly interpolated percentile.
    """
    steps = scalar(steps)
    if isinstance(steps, bool) or not isinstance(steps, numbers.Real):
        raise TypeError("Episode steps must be a real number.")
    if not math.isfinite(steps) or steps < 0:
        raise ValueError("Episode steps must be finite and nonnegative.")
    return math.floor(float(steps) / 500 + 0.5) * 500


@dataclass(frozen=True)
class EpisodeMetadata:
    """Optional labels; speed is an episode length measured in timesteps."""

    speed: int | None = None
    quality: int | None = None
    mistake: bool | None = None

    def __post_init__(self) -> None:
        if self.speed is not None:
            object.__setattr__(self, "speed", _integer(self.speed, "Speed"))
        if self.quality is not None:
            quality = _integer(self.quality, "Quality", 1)
            if quality > 5:
                raise ValueError("Quality must be in [1, 5].")
            object.__setattr__(self, "quality", quality)
        if self.mistake is not None:
            mistake = scalar(self.mistake)
            if not isinstance(mistake, bool):
                raise TypeError("Mistake must be a boolean, not a string or number.")
            object.__setattr__(self, "mistake", mistake)


def runtime_metadata(episode_lengths: list[int] | np.ndarray) -> EpisodeMetadata:
    """Use the task's 15th-percentile duration, quality 5, and no mistake."""
    if isinstance(episode_lengths, np.ndarray) and episode_lengths.ndim != 1:
        raise ValueError("Provide a nonempty one-dimensional set of episode lengths.")
    lengths = [_integer(item, "Episode length", 1) for item in episode_lengths]
    if not lengths:
        raise ValueError("Provide a nonempty one-dimensional set of episode lengths.")
    speed = bin_episode_steps(float(np.percentile(lengths, 15, method="linear")))
    return EpisodeMetadata(speed=speed, quality=5, mistake=False)


@dataclass(frozen=True)
class PromptContext:
    task: str
    subtask: str | None = None
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)
    control_mode: str = "joint"
    language_memory: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _text(self.task, "Task"))
        if self.subtask is not None:
            object.__setattr__(self, "subtask", _text(self.subtask, "Subtask"))
        if not isinstance(self.metadata, EpisodeMetadata):
            raise TypeError("metadata must be EpisodeMetadata.")
        mode = _text(self.control_mode, "Control mode")
        if mode not in {"joint", "ee"}:
            raise ValueError("Control mode must be 'joint' or 'ee'.")
        object.__setattr__(self, "control_mode", mode)
        if isinstance(self.language_memory, str):
            raise TypeError("Language memory must be a sequence of instructions.")
        object.__setattr__(self, "language_memory", tuple(_text(x, "Language memory") for x in self.language_memory))

    def to_text(self) -> str:
        """Render the paper's prompt; memory serialization is a local convention."""
        pieces = [f"Task: {self.task.rstrip('.')}."]
        if self.language_memory:
            pieces.append("Memory: " + "; ".join(x.rstrip(".") for x in self.language_memory) + ".")
        if self.subtask is not None:
            pieces.append(f"Subtask: {self.subtask.rstrip('.')}.")
        if self.metadata.speed is not None:
            pieces.append(f"Speed: {bin_episode_steps(self.metadata.speed)}.")
        if self.metadata.quality is not None:
            pieces.append(f"Quality: {self.metadata.quality}.")
        if self.metadata.mistake is not None:
            pieces.append(f"Mistake: {str(self.metadata.mistake).lower()}.")
        pieces.append(f"Control Mode: {self.control_mode}.")
        return " ".join(pieces)


@dataclass(frozen=True)
class DropoutConfig:
    goal_keep_probability: float = 0.25
    subtask_drop_with_goal: float = 0.30
    metadata_drop_probability: float = 0.15
    metadata_field_drop_probability: float = 0.05
    history_drop_probability: float = 0.30
    rear_drop_probability: float = 0.30

    def __post_init__(self) -> None:
        for name, raw_value in vars(self).items():
            value = scalar(raw_value)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{name} must be a probability.")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclass(frozen=True)
class ContextSelection:
    context: PromptContext
    keep_goals: bool
    keep_history: bool
    keep_rear: bool


DEFAULT_DROPOUT = DropoutConfig()


def sample_context_dropout(
    context: PromptContext,
    rng: np.random.Generator,
    *,
    goals_available: bool = True,
    config: DropoutConfig = DEFAULT_DROPOUT,
) -> ContextSelection:
    """Seedable training-only dropout. The control mode is always retained.

    Draws are made for every field, including unavailable fields, so adding one
    optional label does not change the random decisions for the other modalities.
    """
    draws = rng.random(8)
    keep_goals = bool(goals_available and draws[0] < config.goal_keep_probability)
    subtask = context.subtask
    if keep_goals and draws[1] < config.subtask_drop_with_goal:
        subtask = None
    drop_all = draws[2] < config.metadata_drop_probability
    metadata = EpisodeMetadata(
        **{
            name: None
            if drop_all or draws[index + 3] < config.metadata_field_drop_probability
            else getattr(context.metadata, name)
            for index, name in enumerate(("speed", "quality", "mistake"))
        }
    )
    return ContextSelection(
        context=replace(context, subtask=subtask, metadata=metadata),
        keep_goals=keep_goals,
        keep_history=bool(draws[6] >= config.history_drop_probability),
        keep_rear=bool(draws[7] >= config.rear_drop_probability),
    )
