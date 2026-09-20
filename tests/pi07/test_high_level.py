import pytest
import torch

from openpi.pi07.backbone import Gemma3Backbone
from openpi.pi07.high_level import HighLevelModel
from openpi.pi07.model import Pi07
from openpi.pi07.model import Pi07Config


def make_model():
    torch.manual_seed(23)
    backbone = Gemma3Backbone.tiny(image_size=8)
    with torch.no_grad():
        backbone.hf_model.model.multi_modal_projector.mm_input_projection_weight.normal_(std=0.05)
    return HighLevelModel(backbone, state_dim=3)


def make_batch():
    return {
        "images": torch.rand(2, 1, 2, 3, 8, 8),
        "image_mask": torch.ones(2, 1, 2, dtype=torch.bool),
        "states": torch.rand(2, 2, 3),
        "state_mask": torch.ones(2, 2, dtype=torch.bool),
        "input_ids": torch.tensor([[1, 10, 11], [1, 12, 0]]),
        "text_mask": torch.tensor([[True, True, True], [True, True, False]]),
    }


def test_text_only_supervision_reaches_observation_backbone_and_state_projection():
    model, batch = make_model(), make_batch()
    result = model(batch, torch.tensor([[31, 32, 2], [33, 2, -100]]), torch.tensor([[True] * 3, [True, True, False]]))
    assert result["supervised_tokens"] == 5
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.state_projection.weight.grad.abs().sum() > 0
    grad = model.backbone.hf_model.model.vision_tower.vision_model.embeddings.patch_embedding.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
    assert not any("expert" in name or "action_" in name for name, _ in model.named_parameters())


def test_causal_teacher_forcing_and_padding_are_correct():
    model, batch = make_model().eval(), make_batch()
    targets = torch.tensor([[31, 32, 2], [33, 2, -100]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    result = model(batch, targets, mask)
    altered = targets.clone()
    altered[0, 1] = 51
    altered[1, 2] = 5000
    changed = model(batch, altered, mask)
    # Position 1 must not see its own target (31 is its teacher input).
    torch.testing.assert_close(result["logits"][0, :2], changed["logits"][0, :2])
    assert not torch.allclose(result["logits"][0, 2], changed["logits"][0, 2])
    torch.testing.assert_close(result["logits"][1], changed["logits"][1])
    expected = torch.nn.functional.cross_entropy(result["logits"][mask], targets[mask])
    torch.testing.assert_close(result["loss"], expected)


def test_high_level_ignores_generated_goals():
    model, batch = make_model().eval(), make_batch()
    targets = torch.tensor([[31, 2], [33, 2]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    first = model(batch, targets, mask)
    with_goals = {**batch, "goal_images": torch.randn(2, 1, 3, 8, 8), "goal_mask": torch.ones(2, 1, dtype=torch.bool)}
    second = model(with_goals, targets, mask)
    torch.testing.assert_close(first["loss"], second["loss"], atol=0, rtol=0)


def test_greedy_generation_terminates_each_row_at_eos(monkeypatch):
    model, batch = make_model().eval(), make_batch()
    counts = []
    original_embed = model._embed_context  # noqa: SLF001

    def embed(value):
        counts.append(1)
        return original_embed(value)

    def decode(prefix, valid, allowed, ids, mask):
        logits = torch.zeros(2, ids.shape[1], model.backbone.vocab_size)
        # First row terminates immediately; second terminates on third token.
        logits[0, -1, 2] = 10
        logits[1, -1, 2 if ids.shape[1] == 3 else 40] = 10
        return logits

    monkeypatch.setattr(model, "_embed_context", embed)
    monkeypatch.setattr(model, "_decode_tokens", decode)
    result = model.generate(batch, max_new_tokens=5)
    assert result.token_ids.tolist() == [[2, 0, 0], [40, 40, 2]]
    assert result.token_mask.tolist() == [[True, False, False], [True, True, True]]
    assert result.finished.tolist() == [True, True]
    assert len(counts) == 1


def test_real_generation_is_bounded_and_finite():
    model, batch = make_model().eval(), make_batch()
    result = model.generate(batch, max_new_tokens=3)
    assert 1 <= result.token_ids.shape[1] <= 3
    assert result.token_ids.shape == result.token_mask.shape
    assert result.finished.shape == (2,)
    assert torch.all(result.token_ids >= 0)
    assert torch.all(result.token_ids < model.backbone.vocab_size)


def test_from_vla_shares_or_copies_only_required_modules():
    backbone = Gemma3Backbone.tiny(image_size=8)
    vla = Pi07(
        backbone,
        Pi07Config(state_dim=3, action_dim=3, action_horizon=5, max_delay=2, expert_width=32, expert_mlp_dim=48),
    )
    shared = HighLevelModel.from_vla(vla, share_weights=True)
    assert shared.backbone is vla.backbone
    assert shared.state_projection is vla.state_projection
    copied = HighLevelModel.from_vla(vla, share_weights=False)
    assert copied.backbone is not vla.backbone
    assert copied.state_projection is not vla.state_projection
    torch.testing.assert_close(copied.state_projection.weight, vla.state_projection.weight)
    assert not any("expert" in name for name in shared.state_dict())


def test_bad_or_empty_target_rows_are_rejected():
    model, batch = make_model(), make_batch()
    with pytest.raises(ValueError, match="contiguous"):
        model(batch, torch.tensor([[3, 4], [5, 6]]), torch.tensor([[True, True], [False, True]]))
    with pytest.raises(ValueError, match="vocabulary"):
        model(batch, torch.tensor([[3, 4000], [5, 6]]), torch.ones(2, 2, dtype=torch.bool))
