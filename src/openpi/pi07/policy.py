"""Bridge pi0.7 runtime requests to the Torch VLA, without hardware dispatch.

Observations are mappings with RGB ``images[V,H,W,3]`` and ``state[S]``.
Images use uint8 or floating [0,1]. Goals, when present, use ``[G,H,W,3]``.
States and actions must use the normalization of the training dataset. Optional
paired action transforms convert model outputs to another coordinate scale and
convert the runtime's remaining RTC actions back to model coordinates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
from typing import Any

import numpy as np
import torch

from openpi.pi07.context import EpisodeMetadata
from openpi.pi07.context import PromptContext
from openpi.pi07.data import _images
from openpi.pi07.data import _resize_rgb
from openpi.pi07.runtime import GenerationRequest
from openpi.pi07.runtime import PolicyRequest


class ObservationProcessor:
    """Shared preprocessing for action and semantic language policy requests.

    The six chronological 1-second history slots match ``sample_history_indices``
    in the training adapter: select the most recent available observation at or
    before each target time, and mask missing pre-episode history. Image resize
    uses the same implementation as the training adapter.
    """

    def __init__(
        self,
        tokenizer: Callable[[str], Sequence[int]],
        *,
        state_dim: int,
        image_size: int,
        vocab_size: int,
        device: str | torch.device = "cpu",
        max_text_length: int | None = None,
    ):
        for name, value in (
            ("state_dim", state_dim),
            ("image_size", image_size),
            ("vocab_size", vocab_size),
            ("max_text_length", max_text_length),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.state_dim = state_dim
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.max_text_length = max_text_length

    def _observation(self, observation: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(observation, Mapping) or "images" not in observation or "state" not in observation:
            raise ValueError("Observations require 'images' [V,H,W,3] and 'state' [S]")
        images = _images(observation["images"], "observation images", 4)
        if images.shape[0] > 4:
            raise ValueError("pi0.7 supports at most four observation cameras")
        state = np.asarray(observation["state"])
        if (
            state.shape != (self.state_dim,)
            or not np.issubdtype(state.dtype, np.number)
            or np.iscomplexobj(state)
            or not np.isfinite(state).all()
        ):
            raise ValueError(f"state must contain {self.state_dim} finite real coordinates")
        return images, state.astype(np.float32)

    def _context(self, request: GenerationRequest) -> PromptContext:
        metadata = request.context.metadata
        if metadata is None:
            metadata = EpisodeMetadata(quality=5, mistake=False)
        elif isinstance(metadata, Mapping):
            metadata = EpisodeMetadata(**metadata)
        if not isinstance(metadata, EpisodeMetadata):
            raise TypeError("Runtime metadata must be EpisodeMetadata, its field mapping, or None")
        language_memory = request.subtask_history
        if language_memory and language_memory[-1] == request.context.subtask:
            language_memory = language_memory[:-1]
        return PromptContext(
            task=request.context.task,
            subtask=request.context.subtask,
            metadata=metadata,
            control_mode=request.context.control_mode,
            language_memory=language_memory,
        )

    def tokenize(self, context: PromptContext) -> tuple[torch.Tensor, torch.Tensor]:
        ids = np.asarray(self.tokenizer(context.to_text()))
        if (
            ids.ndim != 1
            or ids.size == 0
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.any(ids >= self.vocab_size)
        ):
            raise ValueError("Tokenizer must return nonempty integer IDs inside the model vocabulary")
        if self.max_text_length is not None and len(ids) > self.max_text_length:
            raise ValueError("Prompt exceeds max_text_length; refusing to truncate conditioning")
        tokens = torch.as_tensor(ids.astype(np.int64), device=self.device).unsqueeze(0)
        return tokens, torch.ones_like(tokens, dtype=torch.bool)

    def prepare_batch(self, request: GenerationRequest) -> tuple[dict[str, torch.Tensor], PromptContext]:
        """Validate and prepare one observation; exposed for offline inspection."""
        if not request.history:
            raise ValueError("Policy requests require at least the current history frame")
        timestamps = np.asarray([frame.timestamp for frame in request.history], dtype=np.float64)
        if (
            not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) < 0)
            or abs(timestamps[-1] - request.timestamp) > 1e-8
        ):
            raise ValueError("History must be chronological and end at the request timestamp")
        targets = request.timestamp - np.arange(5, -1, -1, dtype=np.float64)
        indices = np.searchsorted(timestamps, targets + 1e-9, side="right") - 1
        valid = indices >= 0
        indices = np.maximum(indices, 0)
        observations = [self._observation(request.history[index].observation) for index in indices]
        views = observations[-1][0].shape[0]
        if any(images.shape[0] != views for images, _ in observations):
            raise ValueError("Camera count and ordering must stay consistent within a history")
        image_size = self.image_size
        images = np.stack([_resize_rgb(images, image_size) for images, _ in observations], axis=1)
        states = np.stack([state for _, state in observations])
        image_mask = np.broadcast_to(valid, (views, 6)).copy()
        images[~image_mask] = 0
        states[~valid] = 0
        if request.context.goal_images is None:
            goals = np.zeros((0, 3, image_size, image_size), dtype=np.float32)
        else:
            goals = _images(request.context.goal_images, "goal images", 4)
            if goals.shape[0] > 3:
                raise ValueError("pi0.7 supports at most three goal cameras")
            goals = _resize_rgb(goals, image_size)
        arrays = {
            "images": images,
            "image_mask": image_mask,
            "states": states,
            "state_mask": valid,
            "goal_images": goals,
            "goal_mask": np.ones(goals.shape[0], dtype=bool),
        }
        batch = {name: torch.as_tensor(value, device=self.device).unsqueeze(0) for name, value in arrays.items()}
        context = self._context(request)
        batch["input_ids"], batch["text_mask"] = self.tokenize(context)
        return batch, context


class TorchPolicy:
    """Callable action policy for :class:`openpi.pi07.runtime.Pi07Runtime`.

    Supply a model with appropriate trained weights and a callable text tokenizer.
    Metadata-only classifier-free guidance uses ``guidance_beta``. Paired action
    transforms preserve RTC consistency if outputs are denormalized. This adapter
    does not fetch weights or dispatch robot commands.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Callable[[str], Sequence[int]],
        *,
        device: str | torch.device | None = None,
        guidance_beta: float = 0.0,
        num_steps: int | None = None,
        max_text_length: int | None = None,
        action_denormalizer: Callable[[np.ndarray], np.ndarray] | None = None,
        action_normalizer: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        if model.config.action_horizon != 50:
            raise ValueError("Runtime policies require a 50-step action horizon")
        if not math.isfinite(guidance_beta) or guidance_beta < 0:
            raise ValueError("guidance_beta must be finite and nonnegative")
        if num_steps is not None and (isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1):
            raise ValueError("num_steps must be a positive integer")
        if (action_denormalizer is None) != (action_normalizer is None):
            raise ValueError("Supply paired action_normalizer/action_denormalizer so RTC uses model coordinates")
        if device is None:
            first_parameter = next(model.parameters(), None)
            device = first_parameter.device if first_parameter is not None else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.processor = ObservationProcessor(
            tokenizer,
            state_dim=model.config.state_dim,
            image_size=model.backbone.image_size,
            vocab_size=model.backbone.vocab_size,
            device=self.device,
            max_text_length=max_text_length,
        )
        self.guidance_beta = guidance_beta
        self.num_steps = num_steps
        self.action_denormalizer = action_denormalizer
        self.action_normalizer = action_normalizer

    def prepare_batch(self, request: PolicyRequest) -> tuple[dict[str, torch.Tensor], PromptContext]:
        return self.processor.prepare_batch(request)

    def _action_array(self, value: Any, *, steps: int, name: str) -> np.ndarray:
        array = np.asarray(value)
        if (
            array.shape != (steps, self.model.config.action_dim)
            or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} must have shape [{steps},{self.model.config.action_dim}] and finite values")
        return array.astype(np.float32)

    @torch.inference_mode()
    def __call__(self, request: PolicyRequest) -> np.ndarray:
        batch, context = self.prepare_batch(request)
        unconditional = None
        if self.guidance_beta:
            unconditional = dict(batch)
            unconditional["input_ids"], unconditional["text_mask"] = self.processor.tokenize(
                dataclasses.replace(context, metadata=EpisodeMetadata())
            )
        delay = request.inference_delay_steps
        if (
            isinstance(delay, bool)
            or not isinstance(delay, int)
            or not 0 <= delay <= min(self.model.config.max_delay, len(request.previous_actions))
        ):
            raise ValueError("RTC delay must fit the previous actions and model's trained delay range")
        previous = None
        if delay:
            previous_array = self._action_array(request.previous_actions[:delay], steps=delay, name="RTC actions")
            if self.action_normalizer is not None:
                previous_array = self._action_array(
                    self.action_normalizer(previous_array), steps=delay, name="Normalized RTC actions"
                )
            previous = torch.as_tensor(previous_array, device=self.device).unsqueeze(0)
        result = self.model.sample_actions(
            batch,
            num_steps=self.num_steps,
            previous_actions=previous,
            delay=delay if previous is not None else None,
            unconditional_batch=unconditional,
            guidance_beta=self.guidance_beta,
        )
        if not isinstance(result, torch.Tensor) or tuple(result.shape) != (1, 50, self.model.config.action_dim):
            raise ValueError("Model must return one [1,50,action_dim] action chunk")
        actions = self._action_array(result[0].float().cpu().numpy(), steps=50, name="Model actions")
        if self.action_denormalizer is not None:
            actions = self._action_array(self.action_denormalizer(actions), steps=50, name="Denormalized actions")
        return actions


class TorchHighLevelPolicy:
    """Decode a trained :class:`HighLevelModel` into runtime subtask strings.

    ``tokenizer`` encodes a string, and ``decode`` converts token IDs into text
    (for Hugging Face, pass a function using ``skip_special_tokens=True``).
    The high-level prompt contains task and issued-instruction history. Its
    explicit current-subtask field and all goals are omitted. History is the
    record of issued instructions; completion is not inferred by this adapter.

    Truncated generations raise by default rather than silently presenting an
    incomplete instruction. Set ``require_eos=False`` to explicitly accept text
    truncated at ``max_new_tokens``. Empty decoded instructions always raise.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Callable[[str], Sequence[int]],
        decode: Callable[[Sequence[int]], str],
        *,
        device: str | torch.device | None = None,
        max_new_tokens: int = 64,
        max_text_length: int | None = None,
        require_eos: bool = True,
    ):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if device is None:
            first_parameter = next(model.parameters(), None)
            device = first_parameter.device if first_parameter is not None else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.processor = ObservationProcessor(
            tokenizer,
            state_dim=model.config.state_dim,
            image_size=model.backbone.image_size,
            vocab_size=model.backbone.vocab_size,
            device=self.device,
            max_text_length=max_text_length,
        )
        self.decode = decode
        self.max_new_tokens = max_new_tokens
        self.require_eos = require_eos

    def prepare_batch(self, request: GenerationRequest) -> dict[str, torch.Tensor]:
        request = dataclasses.replace(
            request, context=dataclasses.replace(request.context, subtask=None, goal_images=None)
        )
        batch, _ = self.processor.prepare_batch(request)
        return {key: value for key, value in batch.items() if key not in ("goal_images", "goal_mask")}

    @torch.inference_mode()
    def __call__(self, request: GenerationRequest) -> str:
        generation = self.model.generate(self.prepare_batch(request), max_new_tokens=self.max_new_tokens)
        if (
            generation.token_ids.ndim != 2
            or generation.token_ids.shape[0] != 1
            or generation.token_mask.shape != generation.token_ids.shape
            or generation.token_mask.dtype != torch.bool
            or generation.finished.shape != (1,)
        ):
            raise ValueError("High-level model must return a single valid TextGeneration")
        if self.require_eos and not bool(generation.finished[0]):
            raise ValueError("High-level generation reached max_new_tokens without EOS")
        ids = generation.token_ids[0][generation.token_mask[0]].cpu().tolist()
        text = self.decode(ids)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("High-level generation decoded to an empty instruction")
        return text.strip()
