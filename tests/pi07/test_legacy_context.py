"""Legacy adapter tests that do not require the legacy JAX/model dependencies."""

import numpy as np
import pytest

from openpi.pi07_transforms import AddPi07Context
from openpi.pi07_transforms import PreservePi07Context


def test_preserve_optional_context_across_lossy_dictionary_transforms():
    original = {
        "prompt": "fold the shirt",
        "subtask": "fold the left sleeve",
        "speed": np.asarray(2000),
        "quality": np.asarray(5),
        "mistake": np.asarray(0, dtype=bool),
        "control_mode": np.asarray("joint"),
        "state": np.ones(14),
        "unrelated": "discard this",
    }
    # A transform is allowed to reconstruct a restricted output schema.
    transform = PreservePi07Context(lambda data: {"state": data["state"] + 1})
    result = AddPi07Context()(transform(transform(original)))
    assert result["prompt"] == (
        "fold the shirt. Subtask: fold the left sleeve. Speed: 2000. Quality: 5. Mistake: false. Control Mode: joint."
    )
    assert "unrelated" not in result
    np.testing.assert_array_equal(result["state"], 3)
    np.testing.assert_array_equal(original["state"], 1)


def test_preserve_does_not_overwrite_deliberate_remapping_or_add_missing_context():
    transform = PreservePi07Context(lambda data: {"prompt": "new task", "state": data["state"]})
    assert transform({"prompt": "old task", "state": 1}) == {"prompt": "new task", "state": 1}
    absent = PreservePi07Context(lambda data: {"state": data["state"]})({"state": 1})
    assert absent == {"state": 1}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("speed", 2000.9),
        ("quality", 4.9),
        ("quality", True),
        ("mistake", "false"),
        ("mistake", 0),
        ("subtask", 1),
        ("control_mode", 1),
    ],
)
def test_legacy_context_rejects_coercions(field, value):
    with pytest.raises(TypeError):
        AddPi07Context()({"prompt": "fold", field: value})


def test_legacy_context_keeps_old_prompt_style_and_optional_mode():
    assert AddPi07Context()({"prompt": "fold", "speed": 2250})["prompt"] == "fold. Speed: 2500."
    assert (
        AddPi07Context()({"prompt": "pick up the cup", "control_mode": "ee"})["prompt"]
        == "pick up the cup. Control Mode: ee."
    )
