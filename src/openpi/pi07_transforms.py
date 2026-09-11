import dataclasses
from typing import Any

import numpy as np

from openpi import transforms as _transforms


def _scalar(value: Any) -> Any:
    """Convert numpy scalar-like values to ordinary Python values."""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()

    if isinstance(value, np.generic):
        return value.item()

    return value


@dataclasses.dataclass(frozen=True)
class AddPi07Context:
    """Adds π0.7-style textual context to the task prompt.

    Expected optional fields:
        subtask
        speed
        quality
        mistake
        control_mode

    The original `prompt` is treated as the overall task instruction.

    This transform intentionally runs before TokenizePrompt, so the existing
    PaliGemma tokenizer handles the resulting rich-context prompt normally.
    """

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "prompt" not in data:
            raise ValueError("π0.7 context requires a prompt.")

        task = _scalar(data["prompt"])

        if not isinstance(task, str):
            raise TypeError(f"Expected prompt to be a string, got {type(task)}.")

        task = task.strip().rstrip(".")

        context_parts: list[str] = [task]

        if "subtask" in data:
            subtask = _scalar(data["subtask"])
            context_parts.append(f"Subtask: {str(subtask).strip().rstrip('.')}")

        if "speed" in data:
            speed = int(_scalar(data["speed"]))
            context_parts.append(f"Speed: {speed}")

        if "quality" in data:
            quality = int(_scalar(data["quality"]))

            if not 1 <= quality <= 5:
                raise ValueError(f"Quality must be in [1, 5], got {quality}.")

            context_parts.append(f"Quality: {quality}")

        if "mistake" in data:
            mistake = bool(_scalar(data["mistake"]))
            context_parts.append(f"Mistake: {'true' if mistake else 'false'}")

        if "control_mode" in data:
            control_mode = str(_scalar(data["control_mode"])).lower()

            if control_mode not in {"joint", "ee"}:
                raise ValueError(
                    f"Control mode must be 'joint' or 'ee', got {control_mode!r}."
                )

            context_parts.append(f"Control Mode: {control_mode}")

        prompt = ". ".join(context_parts) + "."

        return {
            **data,
            "prompt": prompt,
        }