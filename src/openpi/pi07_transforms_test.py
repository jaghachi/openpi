import numpy as np
import pytest

from openpi import pi07_transforms
from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config


def test_add_pi07_context():
    transform = pi07_transforms.AddPi07Context()

    data = transform(
        {
            "prompt": "fold the shirt",
            "subtask": "fold the left sleeve",
            "speed": np.asarray(2000),
            "quality": np.asarray(5),
            "mistake": np.False_,
            "control_mode": np.asarray("joint"),
        }
    )

    assert data["prompt"] == (
        "fold the shirt. Subtask: fold the left sleeve. Speed: 2000. Quality: 5. Mistake: false. Control Mode: joint."
    )


def test_add_pi07_context_allows_missing_optional_metadata():
    transform = pi07_transforms.AddPi07Context()

    data = transform(
        {
            "prompt": "pick up the cup",
            "control_mode": "ee",
        }
    )

    assert data["prompt"] == ("pick up the cup. Control Mode: ee.")


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


def test_pi07_context_tokenizes_with_pi05_state():
    transform = _transforms.compose(
        [
            pi07_transforms.AddPi07Context(),
            _transforms.TokenizePrompt(
                _tokenizer.PaligemmaTokenizer(max_len=200),
                discrete_state_input=True,
            ),
        ]
    )

    data = transform(
        {
            "prompt": "fold the shirt",
            "subtask": "fold the left sleeve",
            "speed": 2000,
            "quality": 5,
            "mistake": False,
            "control_mode": "joint",
            "state": np.zeros(32, dtype=np.float32),
        }
    )

    assert data["tokenized_prompt"].shape == (200,)
    assert data["tokenized_prompt_mask"].shape == (200,)
    assert data["tokenized_prompt_mask"].sum() > 0


def test_pi07_model_transform_factory():
    model_config = pi0_config.Pi0Config(pi05=True)

    group = _config.Pi07ModelTransformFactory()(model_config)

    transform = _transforms.compose(group.inputs)

    data = transform(
        {
            "prompt": "fold the shirt",
            "subtask": "fold the left sleeve",
            "speed": 2000,
            "quality": 5,
            "mistake": False,
            "control_mode": "joint",
            "state": np.zeros(32, dtype=np.float32),
            "image": {},
            "actions": np.zeros(
                (model_config.action_horizon, model_config.action_dim),
                dtype=np.float32,
            ),
        }
    )

    assert data["tokenized_prompt"].shape == (model_config.max_token_len,)
    assert data["tokenized_prompt_mask"].shape == (model_config.max_token_len,)
