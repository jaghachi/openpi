import numpy as np
import pytest

from openpi import pi07_transforms


def test_add_pi07_context():
    transform = pi07_transforms.AddPi07Context()

    data = transform(
        {
            "prompt": "fold the shirt",
            "subtask": "fold the left sleeve",
            "speed": np.asarray(2000),
            "quality": np.asarray(5),
            "mistake": np.asarray(False),
            "control_mode": np.asarray("joint"),
        }
    )

    assert data["prompt"] == (
        "fold the shirt. "
        "Subtask: fold the left sleeve. "
        "Speed: 2000. "
        "Quality: 5. "
        "Mistake: false. "
        "Control Mode: joint."
    )


def test_add_pi07_context_allows_missing_optional_metadata():
    transform = pi07_transforms.AddPi07Context()

    data = transform(
        {
            "prompt": "pick up the cup",
            "control_mode": "ee",
        }
    )

    assert data["prompt"] == (
        "pick up the cup. "
        "Control Mode: ee."
    )


def test_add_pi07_context_rejects_bad_quality():
    transform = pi07_transforms.AddPi07Context()

    with pytest.raises(ValueError, match="Quality must be"):
        transform(
            {
                "prompt": "pick up the cup",
                "quality": 7,
            }
        )


def test_add_pi07_context_rejects_bad_control_mode():
    transform = pi07_transforms.AddPi07Context()

    with pytest.raises(ValueError, match="Control mode must"):
        transform(
            {
                "prompt": "pick up the cup",
                "control_mode": "telepathy",
            }
        )