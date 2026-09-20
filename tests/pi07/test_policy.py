import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from openpi.pi07.backbone import Gemma3Backbone
from openpi.pi07.context import EpisodeMetadata
from openpi.pi07.high_level import HighLevelModel
from openpi.pi07.high_level import TextGeneration
from openpi.pi07.model import Pi07
from openpi.pi07.model import Pi07Config
from openpi.pi07.policy import TorchHighLevelPolicy
from openpi.pi07.policy import TorchPolicy
from openpi.pi07.runtime import HistoryFrame
from openpi.pi07.runtime import PolicyRequest
from openpi.pi07.runtime import RuntimeContext


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=50, action_dim=3, state_dim=3, max_delay=12)
        self.backbone = SimpleNamespace(image_size=16, vocab_size=256)
        self.last_call = None

    def sample_actions(self, batch, **kwargs):
        self.last_call = batch, kwargs
        output = torch.zeros(1, 50, 3)
        if kwargs["previous_actions"] is not None:
            output[:, : kwargs["delay"]] = kwargs["previous_actions"]
        return output


def request():
    observation = {"images": np.full((2, 12, 20, 3), 128, dtype=np.uint8), "state": np.arange(3)}
    return PolicyRequest(
        observation=observation,
        history=(HistoryFrame(0.0, observation), HistoryFrame(1.0, observation)),
        context=RuntimeContext(
            "fold towel", "lift edge", None, EpisodeMetadata(speed=2000, quality=5, mistake=False), "joint"
        ),
        subtask_history=("grasp edge", "lift edge"),
        timestep=50,
        timestamp=1.0,
        previous_actions=(),
        inference_delay_steps=0,
    )


def test_history_masks_resize_metadata_only_cfg_and_language_memory():
    model = RecordingModel()
    prompts = []

    def tokenizer(text):
        prompts.append(text)
        return [1, 2, 3]

    policy = TorchPolicy(model, tokenizer, guidance_beta=1.7)
    actions = policy(request())
    assert actions.shape == (50, 3)
    batch, options = model.last_call
    assert batch["images"].shape == (1, 2, 6, 3, 16, 16)
    assert batch["state_mask"].tolist() == [[False, False, False, False, True, True]]
    assert batch["images"][:, :, :4].count_nonzero() == 0
    torch.testing.assert_close(batch["images"][:, :, -1], torch.full((1, 2, 3, 16, 16), 128 / 255))
    assert "Memory: grasp edge." in prompts[0]
    assert "Memory: grasp edge; lift edge" not in prompts[0]
    assert "Quality: 5." in prompts[0]
    assert "Quality:" not in prompts[1]
    for field in ("Task: fold towel.", "Subtask: lift edge.", "Control Mode: joint."):
        assert field in prompts[0]
        assert field in prompts[1]
    assert options["unconditional_batch"]["images"] is batch["images"]
    assert options["unconditional_batch"]["states"] is batch["states"]
    assert options["guidance_beta"] == 1.7


def test_previous_actions_use_only_delay_prefix_and_roundtrip_normalization():
    model = RecordingModel()
    policy = TorchPolicy(
        model,
        lambda text: [1],
        action_normalizer=lambda value: (value - 10) / 2,
        action_denormalizer=lambda value: value * 2 + 10,
    )
    previous = tuple(np.full(3, 12 + index, dtype=np.float32) for index in range(35))
    actions = policy(dataclasses.replace(request(), previous_actions=previous, inference_delay_steps=3))
    _, options = model.last_call
    assert options["previous_actions"].shape == (1, 3, 3)
    np.testing.assert_allclose(actions[:3], np.stack(previous[:3]))
    np.testing.assert_allclose(actions[3:], 10)


def test_goals_and_state_validation():
    policy = TorchPolicy(RecordingModel(), lambda text: [1])
    original = request()
    context = dataclasses.replace(original.context, goal_images=np.ones((3, 9, 9, 3), dtype=np.float32))
    batch, _ = policy.prepare_batch(dataclasses.replace(original, context=context))
    assert batch["goal_images"].shape == (1, 3, 3, 16, 16)
    assert batch["goal_mask"].all()
    bad = {"images": original.observation["images"], "state": np.zeros(4)}
    with pytest.raises(ValueError, match="3 finite real coordinates"):
        policy(dataclasses.replace(original, observation=bad, history=(HistoryFrame(1.0, bad),)))
    context = dataclasses.replace(original.context, goal_images=np.ones((4, 9, 9, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="three goal cameras"):
        policy(dataclasses.replace(original, context=context))


@pytest.mark.parametrize("tokens", [[], [-1], [256], [1.5]])
def test_invalid_tokenizer_output(tokens):
    policy = TorchPolicy(RecordingModel(), lambda text: tokens)
    with pytest.raises(ValueError, match="Tokenizer"):
        policy(request())


def test_prompt_not_silently_truncated_and_transforms_must_be_paired():
    policy = TorchPolicy(RecordingModel(), lambda text: [1, 2, 3], max_text_length=2)
    with pytest.raises(ValueError, match="refusing to truncate"):
        policy(request())
    with pytest.raises(ValueError, match="paired"):
        TorchPolicy(RecordingModel(), lambda text: [1], action_denormalizer=lambda value: value)


def test_actual_tiny_gemma_action_model_runs_through_runtime_bridge():
    torch.manual_seed(3)
    model = Pi07(
        Gemma3Backbone.tiny(),
        Pi07Config(state_dim=3, action_dim=3, expert_width=32, expert_mlp_dim=64),
    )
    policy = TorchPolicy(model, lambda text: [1, 5, 9, 2], num_steps=2, guidance_beta=1.3)
    previous = tuple(np.array([0.1, 0.2, 0.3], dtype=np.float32) for _ in range(3))
    actions = policy(dataclasses.replace(request(), previous_actions=previous, inference_delay_steps=3))
    assert actions.shape == (50, 3)
    assert np.isfinite(actions).all()
    np.testing.assert_allclose(actions[:3], np.stack(previous), rtol=0, atol=0)


class RecordingLanguageModel(torch.nn.Module):
    def __init__(self, *, finished=True):
        super().__init__()
        self.config = SimpleNamespace(state_dim=3)
        self.backbone = SimpleNamespace(image_size=16, vocab_size=256)
        self.finished = finished
        self.last_call = None

    def generate(self, batch, **kwargs):
        self.last_call = batch, kwargs
        return TextGeneration(
            token_ids=torch.tensor([[5, 2, 0]]),
            token_mask=torch.tensor([[True, True, False]]),
            finished=torch.tensor([self.finished]),
        )


def test_high_level_adapter_omits_subtask_goals_and_preserves_issued_history():
    model = RecordingLanguageModel()
    prompts, decoded = [], []

    def tokenizer(text):
        prompts.append(text)
        return [1, 4, 7]

    def decode(ids):
        decoded.append(ids)
        return "  place the edge  "

    policy = TorchHighLevelPolicy(model, tokenizer, decode, max_new_tokens=12)
    original = request()
    original = dataclasses.replace(original, context=dataclasses.replace(original.context, goal_images="ignored"))
    assert policy(original) == "place the edge"
    assert "Subtask:" not in prompts[0]
    assert "Memory: grasp edge; lift edge." in prompts[0]
    assert decoded == [[5, 2]]
    batch, options = model.last_call
    assert "goal_images" not in batch
    assert "goal_mask" not in batch
    assert batch["images"].shape == (1, 2, 6, 3, 16, 16)
    assert options == {"max_new_tokens": 12}


def test_high_level_adapter_rejects_truncated_or_empty_instructions():
    model = RecordingLanguageModel(finished=False)
    policy = TorchHighLevelPolicy(model, lambda text: [1], lambda ids: "unfinished")
    with pytest.raises(ValueError, match="without EOS"):
        policy(request())
    policy = TorchHighLevelPolicy(model, lambda text: [1], lambda ids: "unfinished", require_eos=False)
    assert policy(request()) == "unfinished"
    model.finished = True
    policy = TorchHighLevelPolicy(model, lambda text: [1], lambda ids: "  ")
    with pytest.raises(ValueError, match="empty instruction"):
        policy(request())


def test_actual_tiny_high_level_model_runs_through_language_bridge():
    torch.manual_seed(7)
    model = HighLevelModel(Gemma3Backbone.tiny(), state_dim=3)
    policy = TorchHighLevelPolicy(
        model,
        lambda text: [1, 4, 9],
        lambda ids: " ".join(f"synthetic_token_{value}" for value in ids),
        max_new_tokens=2,
        require_eos=False,
    )
    prediction = policy(request())
    assert prediction.startswith("synthetic_token_")
    assert len(prediction.split()) <= 2
