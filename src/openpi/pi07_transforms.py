"""Text-only pi0.7 conditioning adapters for the legacy pi0.5 data pipeline.

These adapters do not change the legacy model architecture. The paper-based
model implementation lives separately in ``openpi.pi07``.
"""

from collections.abc import Callable
import dataclasses
from typing import Any

from openpi.pi07.context import EpisodeMetadata
from openpi.pi07.context import PromptContext

PI07_CONTEXT_FIELDS = ("prompt", "subtask", "speed", "quality", "mistake", "control_mode", "language_memory")


@dataclasses.dataclass(frozen=True)
class PreservePi07Context:
    """Carry optional context across a transform that reconstructs its dictionary.

    RepackTransform and AlohaInputs discard fields outside their base schemas.
    Wrapping them only for the pi0.7 context configuration prevents metadata loss
    without changing those upstream schemas. Deliberately transformed output
    fields take precedence over the original values; absent metadata stays absent.
    """

    transform: Callable[[dict[str, Any]], dict[str, Any]]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        preserved = {name: data[name] for name in PI07_CONTEXT_FIELDS if name in data}
        return {**preserved, **self.transform(data)}


@dataclasses.dataclass(frozen=True)
class AddPi07Context:
    """Validate and append context before legacy prompt tokenization.

    The historical leading task text is retained for pi0.5 checkpoint prompt
    compatibility. The new paper-based PromptContext API includes ``Task:``.
    Numeric/bool strings and fractional integer labels are rejected rather than
    silently coerced; speed is binned into 500-step intervals.
    """

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "prompt" not in data:
            raise ValueError("π0.7 context requires a prompt.")

        context = PromptContext(
            task=data["prompt"],
            subtask=data.get("subtask"),
            metadata=EpisodeMetadata(speed=data.get("speed"), quality=data.get("quality"), mistake=data.get("mistake")),
            control_mode=data.get("control_mode", "joint"),
            language_memory=data.get("language_memory", ()),
        )
        prompt = context.to_text().removeprefix("Task: ")
        if "control_mode" not in data:
            prompt = prompt.removesuffix(" Control Mode: joint.")
        return {**data, "prompt": prompt}
