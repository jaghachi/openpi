"""Offline tests with the real HF Gemma 3 architecture at a tiny scale."""

import pytest
import torch
from transformers import Gemma3Config
from transformers import Gemma3TextConfig
from transformers import SiglipVisionConfig

from openpi.pi07.backbone import Gemma3Backbone
from openpi.pi07.backbone import temporal_position_encoding


def tiny_backbone(*, pretrained_image_size=8):
    torch.manual_seed(17)
    text = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        query_pre_attn_scalar=8,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
    )
    vision = SiglipVisionConfig(
        hidden_size=24,
        intermediate_size=40,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=pretrained_image_size,
        patch_size=2,
        vision_use_head=False,
    )
    config = Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(), mm_tokens_per_image=4)
    model = Gemma3Backbone.from_config(config, image_size=8)
    # HF deliberately initializes the multimodal projection to zero; a random
    # nonzero matrix allows these tests to observe visual dependencies.
    with torch.no_grad():
        model.hf_model.model.multi_modal_projector.mm_input_projection_weight.normal_(std=0.02)
    return model.eval()


def test_single_frame_matches_pretrained_image_path():
    model = tiny_backbone()
    pixels = torch.randn(2, 3, 8, 8)
    actual, mask = model.encode_images(pixels[:, None, None], torch.ones(2, 1, 1, dtype=torch.bool))
    expected = model.hf_model.get_image_features(pixels)
    torch.testing.assert_close(actual, expected)
    assert mask.all()
    assert actual.shape == (2, 4, 32)


def test_temporal_encoding_current_is_exactly_zero():
    encoded = temporal_position_encoding(6, 24, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.count_nonzero(encoded[-1]) == 0
    assert torch.count_nonzero(encoded[:-1]) > 0


def test_memory_changes_current_tokens_and_masked_history_is_inert():
    model = tiny_backbone()
    images = torch.randn(1, 1, 3, 3, 8, 8)
    all_valid = torch.ones(1, 1, 3, dtype=torch.bool)
    baseline, _ = model.encode_images(images, all_valid)
    changed = images.clone()
    changed[:, :, 0] = torch.randn_like(changed[:, :, 0]) * 5
    updated, _ = model.encode_images(changed, all_valid)
    assert not torch.allclose(baseline, updated)
    no_history = torch.tensor([[[False, False, True]]])
    original_masked, _ = model.encode_images(images, no_history)
    changed_masked, _ = model.encode_images(changed, no_history)
    single, _ = model.encode_images(images[:, :, -1:], all_valid[:, :, -1:])
    torch.testing.assert_close(original_masked, changed_masked)
    torch.testing.assert_close(original_masked, single, atol=1e-6, rtol=1e-5)


def test_missing_camera_is_zero_and_position_interpolation_works():
    model = tiny_backbone(pretrained_image_size=16)
    images = torch.randint(0, 256, (1, 2, 2, 3, 10, 12), dtype=torch.uint8)
    tokens, valid = model.encode_images(images, torch.tensor([[[True, True], [False, False]]]))
    assert tokens.shape == (1, 8, 32)
    assert valid.tolist() == [[True] * 4 + [False] * 4]
    assert torch.count_nonzero(tokens[:, 4:]) == 0
    assert torch.isfinite(tokens).all()


def test_custom_mask_prevents_future_and_cross_branch_leakage_and_returns_real_kv():
    model = tiny_backbone()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    # Shared bidirectional observation (0:2), causal text (2:4), then two
    # independent output branches (4 and 5), as in FAST/action or CFG masking.
    allowed = torch.ones(1, 6, 6, dtype=torch.bool).tril()
    allowed[:, :2, :2] = True
    allowed[:, 5, 4] = False
    valid = torch.ones(1, 6, dtype=torch.bool)
    output = model(model.embed_text(ids), allowed, valid)
    changed = ids.clone()
    changed[:, 4] = 13
    modified = model(model.embed_text(changed), allowed, valid)
    torch.testing.assert_close(output.last_hidden_state[:, :4], modified.last_hidden_state[:, :4])
    torch.testing.assert_close(output.last_hidden_state[:, 5], modified.last_hidden_state[:, 5])
    assert not torch.allclose(output.last_hidden_state[:, 4], modified.last_hidden_state[:, 4])
    assert len(output.hidden_states) == model.depth + 1
    assert len(output.key_values) == model.depth
    key, value = output.key_values[0]
    assert key.shape == value.shape == (1, 2, 6, 8)
    assert key.requires_grad
    assert value.requires_grad
    assert output.logits.shape == (1, 6, model.vocab_size)
    torch.testing.assert_close(model.decode(output.last_hidden_state), output.logits)
    cosine, sine = model.rope(torch.arange(6)[None], 0, torch.float32)
    assert cosine.shape == sine.shape == (1, 6, 8)


def test_masked_tokens_are_inert_and_backward_reaches_visual_encoder():
    model = tiny_backbone().train()
    images = torch.randn(1, 1, 2, 3, 8, 8)
    tokens, image_mask = model.encode_images(images, torch.ones(1, 1, 2, dtype=torch.bool))
    embeddings = torch.cat((tokens, model.embed_text(torch.tensor([[1, 2]]))), 1)
    valid = torch.cat((image_mask, torch.tensor([[True, False]])), 1)
    allowed = torch.ones(1, 6, 6, dtype=torch.bool)
    output = model(embeddings, allowed, valid, return_logits=False)
    assert output.logits is None
    assert torch.count_nonzero(output.last_hidden_state[:, -1]) == 0
    model.decode(output.last_hidden_state[:, -2]).square().mean().backward()
    grad = model.hf_model.model.vision_tower.vision_model.embeddings.patch_embedding.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_invalid_attention_query_is_rejected():
    model = tiny_backbone()
    with pytest.raises(ValueError, match="every valid query"):
        model(torch.randn(1, 2, 32), torch.zeros(1, 2, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))
