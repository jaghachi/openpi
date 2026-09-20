import dataclasses

import pytest
import torch
from transformers import Gemma3Config
from transformers import Gemma3TextConfig
from transformers import SiglipVisionConfig

from openpi.pi07.backbone import Gemma3Backbone
from openpi.pi07.model import EncodedContext
from openpi.pi07.model import ExpertLayer
from openpi.pi07.model import Pi07
from openpi.pi07.model import Pi07Config
from openpi.pi07.model import prefix_attention_mask


def make_model():
    torch.manual_seed(11)
    config = Gemma3Config(
        text_config=Gemma3TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            sliding_window=32,
            max_position_embeddings=256,
        ),
        vision_config=SiglipVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
        ),
        mm_tokens_per_image=4,
    )
    backbone = Gemma3Backbone.from_config(config, image_size=16)
    return Pi07(
        backbone,
        Pi07Config(action_dim=3, state_dim=3, action_horizon=5, expert_width=32, expert_mlp_dim=64, max_delay=2),
    )


def make_batch():
    return {
        "images": torch.rand(2, 1, 2, 3, 16, 16),
        "image_mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "states": torch.rand(2, 2, 3),
        "state_mask": torch.ones(2, 2, dtype=torch.bool),
        "goal_images": torch.rand(2, 1, 3, 16, 16),
        "goal_mask": torch.tensor([[True], [False]]),
        "input_ids": torch.tensor([[2, 4, 5, 6], [2, 7, 8, 0]]),
        "text_mask": torch.tensor([[True, True, True, True], [True, True, True, False]]),
        "actions": torch.randn(2, 5, 3),
        "action_mask": torch.ones(2, 5, 3, dtype=torch.bool),
        "fast_token_ids": torch.tensor([[32, 33, 1], [34, 35, 1]]),
        "fast_mask": torch.ones(2, 3, dtype=torch.bool),
    }


def test_attention_follows_figure_19():
    valid = torch.tensor([[True] * 7 + [False]])
    mask = prefix_attention_mask(valid, 2, 3)[0]
    assert mask[0, :2].all()
    assert not mask[:2, 2:].any()
    assert mask[3, :4].all()
    assert not mask[3, 4:].any()
    assert mask[5:7, :7].all()
    assert not mask[:, 7].any()
    assert not mask[7].any()


def test_joint_loss_and_gradient_insulation():
    model, batch = make_model(), make_batch()
    # Zero-initialized residual gates would make this check vacuous.
    with torch.no_grad():
        for layer in model.expert:
            layer.attn_norm.modulation.bias[2 * model.config.expert_width :].fill_(0.5)
    result = model(batch, delays=torch.tensor([0, 2]))
    assert all(torch.isfinite(result[k]) for k in ("loss", "flow_loss", "fast_loss"))
    result["flow_loss"].backward(retain_graph=True)
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.backbone.parameters())
    assert model.state_projection.weight.grad is None
    assert model.action_out.weight.grad.abs().sum() > 0
    assert model.expert[0].q.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    result["fast_loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert model.state_projection.weight.grad.abs().sum() > 0
    assert model.action_out.weight.grad is None


def test_fast_labels_cannot_leak_into_flow_context():
    model, batch = make_model().eval(), make_batch()
    noise, time, delays = torch.randn_like(batch["actions"]), torch.tensor([0.3, 0.7]), torch.tensor([0, 1])
    first = model(batch, noise=noise, time=time, delays=delays)
    changed = {**batch, "fast_token_ids": torch.tensor([[40, 42, 1], [43, 44, 1]])}
    original_context, _ = model.encode_context(
        batch, fast_targets=batch["fast_token_ids"], fast_mask=batch["fast_mask"]
    )
    changed_context, _ = model.encode_context(
        changed, fast_targets=changed["fast_token_ids"], fast_mask=changed["fast_mask"]
    )
    for old_kv, new_kv in zip(original_context.key_values, changed_context.key_values, strict=True):
        for old, new in zip(old_kv, new_kv, strict=True):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
    second = model(changed, noise=noise, time=time, delays=delays)
    torch.testing.assert_close(first["flow_loss"], second["flow_loss"], rtol=0, atol=0)
    assert not torch.isclose(first["fast_loss"], second["fast_loss"])


def test_int32_fast_targets_match_int64_cross_entropy_and_flow():
    model, batch = make_model().eval(), make_batch()
    noise = torch.randn_like(batch["actions"])
    time, delays = torch.tensor([0.3, 0.7]), torch.tensor([0, 1])
    expected = model(batch, noise=noise, time=time, delays=delays)
    integer_batch = {**batch, "fast_token_ids": batch["fast_token_ids"].to(torch.int32)}
    actual = model(integer_batch, noise=noise, time=time, delays=delays)
    for name in ("loss", "fast_loss", "flow_loss"):
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_rtc_mask_and_per_token_time(monkeypatch):
    model, batch = make_model(), make_batch()
    observed = {}

    def velocity(context, noisy, time):
        observed.update(noisy=noisy.detach(), time=time.detach())
        return torch.zeros_like(noisy)

    monkeypatch.setattr(model, "velocity", velocity)
    batch["action_mask"][:, -1] = False
    result = model(
        batch, noise=torch.ones_like(batch["actions"]), time=torch.tensor([0.5, 0.5]), delays=torch.tensor([2, 0])
    )
    torch.testing.assert_close(observed["noisy"][0, :2], batch["actions"][0, :2])
    assert observed["time"][0, :2].eq(0).all()
    assert result["supervised_action_elements"] == 18


def test_sampling_reclamps_prefix_and_cfg_equation(monkeypatch):
    model, batch = make_model().eval(), make_batch()
    prefix = torch.ones(2, 2, 3) * 0.25
    positive, negative = object(), object()
    monkeypatch.setattr(model, "encode_context", lambda value: (positive if value is batch else negative, None))
    # Context shape is read from its validity mask before integration.
    positive = EncodedContext((), torch.ones(2, 1, dtype=torch.bool))
    negative = dataclasses.replace(positive)
    seen = []

    def velocity(context, x, time):
        seen.append(x[:, :2].clone())
        assert time[:, :2].eq(0).all()
        return torch.full_like(x, 2 if context is positive else 1)

    monkeypatch.setattr(model, "velocity", velocity)
    result = model.sample_actions(
        batch,
        noise=torch.zeros(2, 5, 3),
        previous_actions=prefix,
        unconditional_batch={},
        guidance_beta=1.3,
        num_steps=5,
    )
    torch.testing.assert_close(result[:, :2], prefix)
    torch.testing.assert_close(result[:, 2:], torch.full((2, 3, 3), -3.3))
    assert all(torch.equal(value, prefix) for value in seen)


def test_real_sample_and_optimizer_step():
    model, batch = make_model(), make_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.action_out.weight.detach().clone()
    result = model(batch)
    result["loss"].backward()
    optimizer.step()
    assert not torch.equal(before, model.action_out.weight)
    model.eval()
    actions = model.sample_actions(batch, num_steps=2)
    assert actions.shape == (2, 5, 3)
    assert torch.isfinite(actions).all()


def test_missing_fast_targets_fails():
    model, batch = make_model(), make_batch()
    batch["fast_token_ids"] = torch.empty(2, 0, dtype=torch.long)
    batch["fast_mask"] = torch.empty(2, 0, dtype=torch.bool)
    with pytest.raises(ValueError, match="FAST"):
        model(batch)


@pytest.mark.parametrize("bad_time", [torch.tensor([float("nan"), 0.5]), torch.tensor([-0.1, 0.5]), torch.ones(2, 1)])
def test_invalid_flow_times_rejected(bad_time):
    with pytest.raises(ValueError, match="Flow time"):
        make_model()(make_batch(), time=bad_time)


def test_fractional_rtc_delay_rejected():
    with pytest.raises(ValueError, match="RTC delays"):
        make_model()(make_batch(), delays=torch.tensor([0.5, 1.0]))


def test_masked_fast_padding_cannot_index_vocabulary():
    model, batch = make_model(), make_batch()
    batch["fast_mask"][1, -1] = False
    batch["fast_token_ids"][1, -1] = -100
    assert torch.isfinite(model(batch)["loss"])


def test_empty_fast_row_rejected():
    model, batch = make_model(), make_batch()
    batch["fast_mask"][0] = False
    with pytest.raises(ValueError, match="nonempty"):
        model(batch)


def test_default_expert_has_paper_scale_without_allocating_weights():
    config = Pi07Config()
    with torch.device("meta"):
        layer = ExpertLayer(config.expert_width, config.expert_mlp_dim, 8, 4, 256)
    # Gemma 3 4B has 34 decoder layers; projections/time MLP add about 2M.
    transformer_parameters = 34 * sum(parameter.numel() for parameter in layer.parameters())
    assert 850_000_000 < transformer_parameters < 865_000_000
