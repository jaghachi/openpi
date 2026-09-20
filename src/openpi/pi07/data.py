"""Episode-safe pi0.7 sampling and a small, explicit NPZ training-data adapter.

NPZ files contain one episode, never concatenated trajectories. Required arrays:
``images[N,V,H,W,3]``, ``states[N,S]``, ``actions[N,A]``, and scalar Unicode
``task``, ``control_mode`` and numeric ``fps``. Images are RGB uint8 or float
in [0, 1]. Optional fields are documented by :class:`Episode`.

This is an in-memory reference adapter, not the paper's private data pipeline.
Action normalization and FAST encoding must be provided by the data owner.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np

from openpi.pi07.context import ContextSelection
from openpi.pi07.context import DropoutConfig
from openpi.pi07.context import EpisodeMetadata
from openpi.pi07.context import PromptContext
from openpi.pi07.context import sample_context_dropout
from openpi.pi07.context import scalar


def _positive_integer(value: Any, name: str) -> int:
    value = scalar(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _images(value: Any, name: str, ndim: int) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != ndim or value.shape[-1] != 3 or any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must have {ndim} nonempty dimensions and RGB channels last.")
    if value.dtype == np.uint8:
        return value
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"{name} must be uint8 or floating point RGB.")
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError(f"Floating-point {name} must be finite and in [0, 1].")
    return value


@dataclass
class Episode:
    """A single contiguous trajectory with explicitly aligned annotations.

    ``segment_end[N]`` holds inclusive final frame indices and cannot point before
    the current frame. ``subtask`` and ``mistake`` can be scalar or length N.
    ``action_mask[N,A]`` marks available action coordinates. ``generated_goals``
    optionally supplies [N,G,H,W,3] already generated goals, with G <= 3.
    ``fast_token_ids[N,F]``/``fast_mask[N,F]`` optionally supply cached FAST labels
    for the exact action chunks returned by this adapter.
    """

    images: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    task: str
    control_mode: str
    fps: float
    subtask: str | np.ndarray | None = None
    quality: int | None = None
    mistake: bool | np.ndarray | None = None
    segment_end: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    camera_names: Sequence[str] | None = None
    action_mask: np.ndarray | None = None
    generated_goals: np.ndarray | None = None
    fast_token_ids: np.ndarray | None = None
    fast_mask: np.ndarray | None = None
    _subtask_run_starts: np.ndarray = field(init=False, repr=False)
    _subtask_run_labels: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.images = _images(self.images, "images", 5)
        length, views = self.images.shape[:2]
        if views > 4:
            raise ValueError("pi0.7 accepts at most four camera views.")
        for name in ("states", "actions"):
            array = np.asarray(getattr(self, name))
            if array.ndim != 2 or array.shape[0] != length or array.shape[1] == 0:
                raise ValueError(f"{name} must have shape [N,D] matching the images.")
            if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array) or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain finite real numbers.")
            setattr(self, name, array.astype(np.float32))
        self.fps = scalar(self.fps)
        if (
            isinstance(self.fps, bool)
            or not isinstance(self.fps, (float, int))
            or not np.isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError("fps must be finite and positive.")
        validated = PromptContext(
            self.task, metadata=EpisodeMetadata(quality=self.quality), control_mode=self.control_mode
        )
        self.task, self.control_mode, self.quality = validated.task, validated.control_mode, validated.metadata.quality
        if self.timestamps is None:
            self.timestamps = np.arange(length, dtype=np.float64) / self.fps
        else:
            self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
            if (
                self.timestamps.shape != (length,)
                or not np.isfinite(self.timestamps).all()
                or np.any(np.diff(self.timestamps) <= 0)
            ):
                raise ValueError("timestamps must be finite, strictly increasing, and shape [N].")
        if self.segment_end is None:
            self.segment_end = np.full(length, length - 1, dtype=np.int64)
        else:
            ends = np.asarray(self.segment_end)
            if ends.shape != (length,) or not np.issubdtype(ends.dtype, np.integer):
                raise ValueError("segment_end must contain integer inclusive indices with shape [N].")
            if np.any(ends < np.arange(length)) or np.any(ends >= length):
                raise ValueError("segment_end must stay inside its episode and cannot precede the current frame.")
            # A new boundary is allowed only immediately after the previous end.
            transitions = np.flatnonzero(ends[1:] != ends[:-1])
            if np.any(ends[transitions] != transitions):
                raise ValueError("segment_end annotations must describe contiguous consistent segments.")
            self.segment_end = ends.astype(np.int64)
        for name in ("subtask", "mistake"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim == 0:
                array = np.repeat(array[None], length)
            if array.shape != (length,):
                raise ValueError(f"{name} must be a scalar or shape [N].")
            for item in array:
                if name == "mistake":
                    EpisodeMetadata(mistake=item)
                else:
                    PromptContext(self.task, subtask=item)
            setattr(self, name, array)
        if self.subtask is None:
            self._subtask_run_starts = np.empty(0, dtype=np.int64)
            self._subtask_run_labels = ()
        else:
            self._subtask_run_starts = np.concatenate(([0], np.flatnonzero(self.subtask[1:] != self.subtask[:-1]) + 1))
            self._subtask_run_labels = tuple(str(self.subtask[start]) for start in self._subtask_run_starts)
        if self.camera_names is None:
            self.camera_names = ("front", "left_wrist", "right_wrist", "rear")[:views]
        else:
            self.camera_names = tuple(str(x) for x in self.camera_names)
        if len(self.camera_names) != views or len(set(self.camera_names)) != views:
            raise ValueError("camera_names must contain one unique name per view.")
        if self.action_mask is None:
            self.action_mask = np.ones(self.actions.shape, dtype=bool)
        else:
            self.action_mask = np.asarray(self.action_mask)
            if self.action_mask.dtype != bool or self.action_mask.shape != self.actions.shape:
                raise ValueError("action_mask must be boolean with the same shape as actions.")
        if self.generated_goals is not None:
            self.generated_goals = _images(self.generated_goals, "generated_goals", 5)
            if self.generated_goals.shape[0] != length or self.generated_goals.shape[1] > 3:
                raise ValueError("generated_goals must have N frames and at most three views.")
        if self.fast_token_ids is not None:
            self.fast_token_ids = np.asarray(self.fast_token_ids)
            if (
                self.fast_token_ids.ndim != 2
                or self.fast_token_ids.shape[0] != length
                or not np.issubdtype(self.fast_token_ids.dtype, np.integer)
            ):
                raise ValueError("fast_token_ids must be integers with shape [N,F].")
            if np.any(self.fast_token_ids < 0):
                raise ValueError("FAST token IDs cannot be negative.")
            self.fast_token_ids = self.fast_token_ids.astype(np.int64)
            if self.fast_mask is None:
                self.fast_mask = np.ones(self.fast_token_ids.shape, dtype=bool)
            self.fast_mask = np.asarray(self.fast_mask)
            if self.fast_mask.dtype != bool or self.fast_mask.shape != self.fast_token_ids.shape:
                raise ValueError("fast_mask must be boolean with the same shape as fast_token_ids.")
        elif self.fast_mask is not None:
            raise ValueError("fast_mask requires fast_token_ids.")

    def __len__(self) -> int:
        return self.images.shape[0]

    def language_memory(self, index: int, *, limit: int = 64) -> tuple[str, ...]:
        """Return bounded prior instruction runs without current/future labels."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Language memory limit must be a nonnegative integer.")
        if not limit or not self._subtask_run_labels:
            return ()
        current_run = int(np.searchsorted(self._subtask_run_starts, index, side="right") - 1)
        return self._subtask_run_labels[max(0, current_run - limit) : current_run]

    @classmethod
    def from_npz(cls, path: str | Path) -> Episode:
        """Load only primitive arrays; object arrays/pickle are never enabled."""
        with np.load(path, allow_pickle=False) as archive:
            required = {"images", "states", "actions", "task", "control_mode", "fps"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Missing required NPZ fields: {sorted(missing)}")
            accepted = {name for name, definition in cls.__dataclass_fields__.items() if definition.init}
            unknown = set(archive.files) - accepted
            if unknown:
                raise ValueError(f"Unknown NPZ fields: {sorted(unknown)}")
            values = {name: archive[name] for name in archive.files}
        return cls(**values)


@dataclass(frozen=True)
class SamplingConfig:
    image_size: int = 448
    history_frames: int = 6
    history_stride_seconds: float = 1.0
    action_horizon: int = 50
    segment_end_probability: float = 0.25
    future_goal_seconds: float = 4.0
    generated_goal_probability: float = 0.0
    language_memory_limit: int = 64
    dropout: DropoutConfig = field(default_factory=DropoutConfig)

    def __post_init__(self) -> None:
        for name in ("image_size", "history_frames", "action_horizon"):
            _positive_integer(getattr(self, name), name)
        if self.history_frames > 6:
            raise ValueError("At most six observation frames are supported.")
        if (
            isinstance(self.language_memory_limit, bool)
            or not isinstance(self.language_memory_limit, int)
            or self.language_memory_limit < 0
        ):
            raise ValueError("language_memory_limit must be a nonnegative integer.")
        for name in ("history_stride_seconds", "future_goal_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("segment_end_probability", "generated_goal_probability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")


DEFAULT_SAMPLING = SamplingConfig()


def sample_history_indices(episode: Episode, index: int, config: SamplingConfig) -> tuple[np.ndarray, np.ndarray]:
    """Oldest-to-current indices, using only frames at or before each target time.

    Six frames includes the current observation; missing pre-episode history is
    masked rather than borrowing from another episode or repeating valid tokens.
    """
    if not 0 <= index < len(episode):
        raise IndexError(index)
    targets = episode.timestamps[index] - np.arange(config.history_frames - 1, -1, -1) * config.history_stride_seconds
    indices = np.searchsorted(episode.timestamps, targets + 1e-9, side="right") - 1
    valid = indices >= 0
    indices = np.clip(indices, 0, index)
    return indices, valid


def sample_goal_index(episode: Episode, index: int, rng: np.random.Generator, config: SamplingConfig) -> int:
    """Choose a segment-end image or a uniform feasible future frame.

    Uniform sampling is truncated only at the episode boundary, and can cross
    semantic segment boundaries as permitted by the paper's 0-4 second range,
    avoiding the endpoint mass that clipping an out-of-range sample would create.
    """
    if not 0 <= index < len(episode):
        raise IndexError(index)
    end = int(episode.segment_end[index])
    if rng.random() < config.segment_end_probability:
        return end
    latest = int(
        np.searchsorted(episode.timestamps, episode.timestamps[index] + config.future_goal_seconds, side="right") - 1
    )
    return int(rng.integers(index, min(latest, len(episode) - 1) + 1))


def subtask_memory(episode: Episode, index: int, *, limit: int = 64) -> tuple[str, ...]:
    """Prior instruction runs, excluding the current run and all future labels.

    Consecutive identical annotations represent one instruction; repetitions
    after another instruction remain meaningful. Precomputed run boundaries make
    lookup O(log(number of runs) + limit), without rescanning the episode.
    """
    return episode.language_memory(index, limit=limit)


def _resize_rgb(images: np.ndarray, size: int) -> np.ndarray:
    """Bilinear RGB resize with half-pixel centers, output CHW float in [0,1]."""
    values = images.astype(np.float32)
    if images.dtype == np.uint8:
        values /= 255.0
    height, width = values.shape[-3:-1]
    if (height, width) != (size, size):
        ys = np.clip((np.arange(size) + 0.5) * height / size - 0.5, 0, height - 1)
        xs = np.clip((np.arange(size) + 0.5) * width / size - 0.5, 0, width - 1)
        y0, x0 = ys.astype(int), xs.astype(int)
        y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
        wy, wx = (ys - y0).astype(np.float32)[:, None, None], (xs - x0).astype(np.float32)[None, :, None]
        top = values[..., y0[:, None], x0[None, :], :] * (1 - wx) + values[..., y0[:, None], x1[None, :], :] * wx
        bottom = values[..., y1[:, None], x0[None, :], :] * (1 - wx) + values[..., y1[:, None], x1[None, :], :] * wx
        values = top * (1 - wy) + bottom * wy
    return np.moveaxis(values, -1, -3).astype(np.float32)


def sample_example(
    episode: Episode,
    index: int,
    rng: np.random.Generator,
    *,
    config: SamplingConfig = DEFAULT_SAMPLING,
    training: bool = True,
) -> dict[str, Any]:
    indices, state_mask = sample_history_indices(episode, index, config)
    context = PromptContext(
        episode.task,
        subtask=None if episode.subtask is None else episode.subtask[index],
        metadata=EpisodeMetadata(
            len(episode), episode.quality, None if episode.mistake is None else episode.mistake[index]
        ),
        control_mode=episode.control_mode,
        language_memory=subtask_memory(episode, index, limit=config.language_memory_limit),
    )
    selection = (
        sample_context_dropout(context, rng, config=config.dropout)
        if training
        else ContextSelection(context, keep_goals=True, keep_history=True, keep_rear=True)
    )
    if not selection.keep_history:
        state_mask[:-1] = False
    image_mask = np.broadcast_to(state_mask, (episode.images.shape[1], config.history_frames)).copy()
    if not selection.keep_rear and "rear" in episode.camera_names:
        image_mask[episode.camera_names.index("rear")] = False
    images = _resize_rgb(np.swapaxes(episode.images[indices], 0, 1), config.image_size)
    images[~image_mask] = 0
    states = episode.states[indices].copy()
    states[~state_mask] = 0
    goal_index = sample_goal_index(episode, index, rng, config)
    goal_views = [i for i, name in enumerate(episode.camera_names) if name != "rear"][:3]
    use_generated = episode.generated_goals is not None and rng.random() < config.generated_goal_probability
    raw_goals = episode.generated_goals[index] if use_generated else episode.images[goal_index, goal_views]
    goal_images = _resize_rgb(raw_goals, config.image_size)
    goal_mask = np.full(len(raw_goals), selection.keep_goals, dtype=bool)
    goal_images[~goal_mask] = 0
    actions = np.zeros((config.action_horizon, episode.actions.shape[1]), dtype=np.float32)
    action_mask = np.zeros(actions.shape, dtype=bool)
    count = min(config.action_horizon, len(episode) - index)
    actions[:count] = episode.actions[index : index + count]
    action_mask[:count] = episode.action_mask[index : index + count]
    actions[~action_mask] = 0
    result = {
        "images": images,
        "image_mask": image_mask,
        "states": states,
        "state_mask": state_mask,
        "goal_images": goal_images,
        "goal_mask": goal_mask,
        "prompt": selection.context.to_text(),
        "actions": actions,
        "action_mask": action_mask,
        "context": selection.context,
        "history_indices": indices,
        "goal_index": goal_index,
        "generated_goal": use_generated,
    }
    if episode.fast_token_ids is not None:
        result["fast_token_ids"] = episode.fast_token_ids[index].copy()
        result["fast_mask"] = episode.fast_mask[index].copy()
    return result


class EpisodeDataset:
    """Deterministic random access, with fresh sampling after ``set_epoch``.

    The seed is keyed by epoch and global sample index, making augmentation
    independent of data-loader worker ordering. Episodes are retained in memory.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        *,
        config: SamplingConfig = DEFAULT_SAMPLING,
        seed: int = 0,
        training: bool = True,
    ):
        if not episodes:
            raise ValueError("At least one episode is required.")
        self.episodes = tuple(episodes)
        self.config, self.seed, self.training, self.epoch = config, seed, training, 0
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        self._ends = np.cumsum([len(episode) for episode in self.episodes])

    @classmethod
    def from_npz(cls, paths: Sequence[str | Path], **kwargs: Any) -> EpisodeDataset:
        return cls([Episode.from_npz(path) for path in paths], **kwargs)

    def __len__(self) -> int:
        return int(self._ends[-1])

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer.")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = int(np.searchsorted(self._ends, index, side="right"))
        local_index = index - (0 if episode_index == 0 else int(self._ends[episode_index - 1]))
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        return sample_example(
            self.episodes[episode_index], local_index, rng, config=self.config, training=self.training
        )


def collate_examples(
    examples: Sequence[dict[str, Any]],
    tokenizer: Callable[[str], Sequence[int]],
    *,
    action_tokenizer: Callable[[np.ndarray, np.ndarray], Sequence[int]] | None = None,
    max_text_length: int | None = None,
    pad_token_id: int = 0,
    target_state_dim: int | None = None,
    target_action_dim: int | None = None,
    as_torch: bool = False,
) -> dict[str, Any]:
    """Pad a batch and preserve every validity mask, without silent text truncation.

    ``action_tokenizer`` receives normalized actions and their boolean coordinate
    mask. It must implement the actual FAST codec for a paper-scale model; no
    scalar-quantization surrogate is silently substituted. FAST encoding occurs
    on each original action tensor before padding to ``target_action_dim``.
    History is right-aligned so current observations/states are always last.
    Torch is optional.
    """
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    batch = len(examples)
    for name, value in (("target_state_dim", target_state_dim), ("target_action_dim", target_action_dim)):
        if value is not None:
            _positive_integer(value, name)
    output: dict[str, np.ndarray] = {}
    for name in ("images", "image_mask", "states", "state_mask", "goal_images", "goal_mask", "actions", "action_mask"):
        arrays = [np.asarray(example[name]) for example in examples]
        if len({array.ndim for array in arrays}) != 1:
            raise ValueError(f"Inconsistent ranks for {name}.")
        if name in {"images", "goal_images"} and len({array.shape[-3:] for array in arrays}) != 1:
            raise ValueError("Images must share channel count and spatial resolution within a batch.")
        shape = tuple(max(array.shape[axis] for array in arrays) for axis in range(arrays[0].ndim))
        target_dim = (
            target_state_dim if name == "states" else target_action_dim if name in {"actions", "action_mask"} else None
        )
        if target_dim is not None:
            if shape[-1] > target_dim:
                raise ValueError(f"{name} exceeds the requested target dimension {target_dim}.")
            shape = (*shape[:-1], target_dim)
        output[name] = np.zeros((batch, *shape), dtype=arrays[0].dtype)
        for row, array in enumerate(arrays):
            slices = [slice(size) for size in array.shape]
            history_axis = 1 if name in {"images", "image_mask"} else 0 if name in {"states", "state_mask"} else None
            if history_axis is not None:
                slices[history_axis] = slice(shape[history_axis] - array.shape[history_axis], shape[history_axis])
            output[name][(row, *slices)] = array
    text_ids = []
    fast_ids, fast_masks = [], []
    for example in examples:
        ids = np.asarray(tokenizer(example["prompt"]))
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
            raise ValueError("Text tokenizer must return a one-dimensional sequence of nonnegative integers.")
        if not ids.size:
            raise ValueError("Text tokenizer returned no tokens for a nonempty prompt.")
        if max_text_length is not None and len(ids) > max_text_length:
            raise ValueError("Prompt exceeds max_text_length; refusing to truncate conditioning silently.")
        text_ids.append(ids.astype(np.int64))
        tokens = example.get("fast_token_ids")
        if tokens is None and action_tokenizer is not None:
            tokens = action_tokenizer(example["actions"], example["action_mask"])
        tokens = np.empty(0, dtype=np.int64) if tokens is None else np.asarray(tokens)
        if tokens.size == 0:
            tokens = tokens.astype(np.int64)
        mask = np.asarray(example.get("fast_mask", np.ones(tokens.shape, dtype=bool)))
        if (
            tokens.ndim != 1
            or not np.issubdtype(tokens.dtype, np.integer)
            or np.any(tokens < 0)
            or mask.shape != tokens.shape
            or mask.dtype != bool
        ):
            raise ValueError("FAST labels must be nonnegative token IDs with matching boolean masks.")
        fast_ids.append(tokens)
        fast_masks.append(mask)
    text_length = max(len(ids) for ids in text_ids)
    output["input_ids"] = np.full((batch, text_length), pad_token_id, dtype=np.int64)
    output["text_mask"] = np.zeros((batch, text_length), dtype=bool)
    fast_length = max(len(ids) for ids in fast_ids)
    output["fast_token_ids"] = np.full((batch, fast_length), pad_token_id, dtype=np.int64)
    output["fast_mask"] = np.zeros((batch, fast_length), dtype=bool)
    for row in range(batch):
        output["input_ids"][row, : len(text_ids[row])] = text_ids[row]
        output["text_mask"][row, : len(text_ids[row])] = True
        output["fast_token_ids"][row, : len(fast_ids[row])] = fast_ids[row]
        output["fast_mask"][row, : len(fast_ids[row])] = fast_masks[row]
    if as_torch:
        import torch  # noqa: PLC0415 -- NumPy-only data preparation does not require torch.

        return {name: torch.from_numpy(value) for name, value in output.items()}
    return output
