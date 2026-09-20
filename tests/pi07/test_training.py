"""Offline training/checkpoint integration on the real tiny Gemma modules."""

from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase
import torch
from transformers import PreTrainedTokenizerFast

from openpi.pi07.backbone import Gemma3Backbone
from openpi.pi07.checkpoint import load_checkpoint
from openpi.pi07.checkpoint import save_checkpoint
from openpi.pi07.model import Pi07
from openpi.pi07.model import Pi07Config
import openpi.pi07.train as training
from openpi.pi07.train import train


def smoke_args(output, *, steps=1, resume=None, seed=17):
    return SimpleNamespace(
        smoke=True,
        steps=steps,
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.01,
        seed=seed,
        device="cpu",
        output=str(output),
        resume=None if resume is None else str(resume),
        data=None,
        data_is_normalized=False,
        backbone=None,
        tokenizer=None,
        fast_processor=None,
        fast_codebook_size=None,
        allow_processor_code=False,
        state_dim=3,
        action_dim=3,
        expert_width=32,
        expert_mlp_dim=64,
        generated_goal_probability=0.0,
    )


def test_split_training_matches_uninterrupted_cpu_updates_exactly(tmp_path):
    continuous = tmp_path / "continuous"
    split = tmp_path / "split"
    full_result = train(smoke_args(continuous, steps=3))
    train(smoke_args(split, steps=1))
    resumed_result = train(smoke_args(split, steps=2, resume=split / "checkpoint.pt"))
    full_model, full = load_checkpoint(continuous / "checkpoint.pt")
    resumed_model, resumed = load_checkpoint(split / "checkpoint.pt")
    assert full["step"] == 3
    assert resumed["step"] == 3
    assert resumed_result["loss"] == full_result["loss"]
    assert resumed_result["gradient_norm"] == full_result["gradient_norm"]
    assert full_result["checkpoint_roundtrip"] == "exact"
    assert resumed_result["checkpoint_roundtrip"] == "exact"
    torch.testing.assert_close(full["torch_rng"], resumed["torch_rng"], rtol=0, atol=0)
    for name, value in full_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0, msg=name)
    for index, state in full["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, resumed["optimizer"]["state"][index][key], rtol=0, atol=0)


def test_resume_rejects_different_data_seed(tmp_path):
    output = tmp_path / "run"
    train(smoke_args(output, seed=3))
    with pytest.raises(ValueError, match=r"[Rr]esume|[Ss]eed|metadata"):
        train(smoke_args(output, resume=output / "checkpoint.pt", seed=4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_preserves_tied_embeddings_and_all_buffers_materialized(tmp_path, dtype):
    model = (
        Pi07(
            Gemma3Backbone.tiny(),
            Pi07Config(state_dim=3, action_dim=3, expert_width=32, expert_mlp_dim=64),
        )
        .to(dtype=dtype)
        .eval()
    )
    # Make sampled actions depend on context before any optimizer updates.
    with torch.no_grad():
        for layer in model.expert:
            layer.attn_norm.modulation.bias[-model.config.expert_width :].fill_(0.25)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, step=4, metadata={"note": "local test"})
    restored, payload = load_checkpoint(path)
    assert payload["step"] == 4
    assert payload["metadata"] == {"note": "local test"}
    assert payload["optimizer"] is None
    assert restored.backbone.hf_model.lm_head.weight is restored.backbone.hf_model.get_input_embeddings().weight
    assert all(parameter.device.type == "cpu" for parameter in restored.parameters())
    assert all(buffer.device.type == "cpu" for buffer in restored.buffers())
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, dict(restored.named_buffers())[name], rtol=0, atol=0, msg=name)
    batch = {
        "images": torch.rand(1, 1, 2, 3, 16, 16),
        "image_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "states": torch.randn(1, 2, 3),
        "state_mask": torch.ones(1, 2, dtype=torch.bool),
        "input_ids": torch.tensor([[1, 4, 7]]),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    noise = torch.randn(1, 50, 3)
    expected = model.sample_actions(batch, noise=noise, num_steps=2)
    actual = restored.eval().sample_actions(batch, noise=noise, num_steps=2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not path.with_suffix(".pt.tmp").exists()


def test_checkpoint_rejects_unknown_format(tmp_path):
    path = tmp_path / "unknown.pt"
    torch.save({"format_version": 999}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(path)


def test_bfloat16_backbone_with_float32_new_layers_trains_and_samples():
    model = Pi07(
        Gemma3Backbone.tiny().to(dtype=torch.bfloat16),
        Pi07Config(state_dim=3, action_dim=3, expert_width=32, expert_mlp_dim=64),
    )
    batch = {
        "images": torch.rand(1, 1, 2, 3, 16, 16),
        "image_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "states": torch.randn(1, 2, 3),
        "state_mask": torch.ones(1, 2, dtype=torch.bool),
        "goal_images": torch.rand(1, 1, 3, 16, 16),
        "goal_mask": torch.ones(1, 1, dtype=torch.bool),
        "input_ids": torch.tensor([[1, 4, 7]]),
        "text_mask": torch.ones(1, 3, dtype=torch.bool),
        "actions": torch.randn(1, 50, 3),
        "action_mask": torch.ones(1, 50, 3, dtype=torch.bool),
        "fast_token_ids": torch.tensor([[16, 17, 2]]),
        "fast_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    losses = model(batch, delays=torch.tensor([2]))
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.state_projection.weight.grad is not None
    assert torch.isfinite(model.state_projection.weight.grad).all()
    actions = model.eval().sample_actions(batch, num_steps=1)
    assert actions.dtype == torch.float32
    assert actions.shape == (1, 50, 3)
    assert torch.isfinite(actions).all()


@pytest.mark.parametrize("change", ["data", "normalizer"])
def test_full_training_loads_local_hf_tokenizer_npz_and_rejects_changed_resume_inputs(tmp_path, monkeypatch, change):
    """Exercise the non-smoke path with cached synthetic FAST labels, offline."""
    pretrained = tmp_path / "pretrained"
    backbone = Gemma3Backbone.tiny()
    backbone.hf_model.save_pretrained(pretrained)
    vocabulary = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocabulary.update({f"word{index}": index for index in range(4, 256)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(pretrained)
    episode = training.smoke_dataset(7).episodes[0]
    data_path = tmp_path / "episode.npz"
    episode_arrays = {
        name: getattr(episode, name)
        for name in ("images", "states", "actions", "task", "control_mode", "fps", "fast_token_ids")
    }
    np.savez(data_path, **episode_arrays)
    # Keep this real HF checkpoint/load/training test at the tiny image geometry.
    pretrained_loader = Gemma3Backbone.from_pretrained
    sampling_config = training.SamplingConfig
    monkeypatch.setattr(
        Gemma3Backbone,
        "from_pretrained",
        lambda path, **kwargs: pretrained_loader(path, image_size=16, **kwargs),
    )
    monkeypatch.setattr(training, "SamplingConfig", lambda **kwargs: sampling_config(image_size=16, **kwargs))
    output = tmp_path / "full"
    args = smoke_args(output)
    args.smoke = False
    args.backbone = str(pretrained)
    args.data = [str(data_path)]
    args.data_is_normalized = True
    result = train(args)
    assert np.isfinite(result["loss"])
    _, payload = load_checkpoint(output / "checkpoint.pt")
    assert payload["metadata"]["smoke"] is False
    assert payload["metadata"]["data_is_normalized"] is True
    assert (output / "tokenizer" / "tokenizer.json").exists()
    args.resume = str(output / "checkpoint.pt")
    if change == "data":
        episode_arrays["actions"] = episode_arrays["actions"].copy()
        episode_arrays["actions"][0, 0] += 0.1
        np.savez(data_path, **episode_arrays)
        fingerprint_field = "data_fingerprints"
    else:
        vocabulary_before = tokenizer.get_vocab().copy()
        special_tokens_before = tokenizer.special_tokens_map.copy()
        assert tokenizer.encode("WORD4", add_special_tokens=False) == [3]
        tokenizer.backend_tokenizer.normalizer = Lowercase()
        assert tokenizer.encode("WORD4", add_special_tokens=False) == [4]
        assert tokenizer.get_vocab() == vocabulary_before
        assert tokenizer.special_tokens_map == special_tokens_before
        tokenizer.save_pretrained(pretrained)
        fingerprint_field = "tokenizer_fingerprint"
    with pytest.raises(ValueError, match=f"Resume requires unchanged {fingerprint_field}"):
        train(args)
