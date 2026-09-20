"""CPU contract tests. ContractBagel below is NOT the real BAGEL network.

These tests validate native API packing, multi-camera attention, CFM reduction,
and CFG/integration. They make no claim that official CUDA BAGEL was executed.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional

from openpi.pi07.runtime import GenerationRequest
from openpi.pi07.runtime import RuntimeContext
from openpi.pi07.world_model import BagelGoalPolicy
from openpi.pi07.world_model import BagelMultiviewWorldModel
from openpi.pi07.world_model import WorldModelConfig
from openpi.pi07.world_model import preprocess_views
from openpi.pi07.world_model import segment_end_example
from openpi.pi07.world_model import three_branch_guidance
from openpi.pi07.world_model import world_attention_mask


class ContractVAE(nn.Module):
    def encode(self, images):
        return functional.avg_pool2d(images, 2)

    def decode(self, latents):
        self.last_latents = latents.detach().clone()
        return functional.interpolate(latents, scale_factor=2)


class ContractBagel(nn.Module):
    """Minimal test double for the inspected native method signatures."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.vit_patch_size = self.latent_downsample = 2
        self.latent_patch_size, self.latent_channel = 1, 3
        self.language_model = SimpleNamespace(model=SimpleNamespace())
        self.timestep_shift = 2.0
        self.config = SimpleNamespace(timestep_shift=2.0)
        self.groups, self.velocity_calls = [], []

    def _images(self, images, curr_rope, new_token_ids, kind):
        count = images[0].shape[-2] // 2 * (images[0].shape[-1] // 2)
        length = count + 2
        positions = torch.cat([torch.full((length,), n) for n in curr_rope])
        text_indexes = torch.tensor([index * length + offset for index in range(3) for offset in (0, length - 1)])
        image_indexes = torch.cat([torch.arange(index * length + 1, (index + 1) * length - 1) for index in range(3)])
        pack = {
            "packed_text_ids": torch.tensor([new_token_ids["start_of_image"], new_token_ids["end_of_image"]] * 3),
            "packed_text_indexes": text_indexes,
            "packed_seqlens": torch.tensor([length] * 3, dtype=torch.int32),
            "packed_position_ids": positions,
            "packed_indexes": torch.arange(3 * length),
            "packed_key_value_indexes": torch.empty(0, dtype=torch.long),
            "key_values_lens": torch.tensor([0] * 3),
        }
        if kind == "vit":
            pack.update(
                packed_vit_tokens=torch.zeros(3 * count, 12),
                packed_vit_token_indexes=image_indexes,
                packed_vit_position_ids=torch.arange(count).repeat(3),
                vit_token_seqlens=torch.tensor([count] * 3),
            )
        else:
            pack.update(
                padded_images=torch.stack(images),
                packed_vae_token_indexes=image_indexes,
                packed_vae_position_ids=torch.arange(count).repeat(3),
                packed_timesteps=torch.tensor([0.0]),
                patchified_vae_latent_shapes=[(image.shape[-2] // 2, image.shape[-1] // 2) for image in images],
            )
        return pack, [length] * 3, [rope + 1 for rope in curr_rope]

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        return self._images([transforms(image) for image in images], curr_rope, new_token_ids, "vit")

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        return self._images([transforms(image) for image in images], curr_rope, new_token_ids, "vae")

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        ids = [new_token_ids["bos_token_id"], *tokenizer(prompts[0]), new_token_ids["eos_token_id"]]
        length = len(ids)
        return (
            {
                "packed_text_ids": torch.tensor(ids),
                "packed_text_indexes": torch.arange(length),
                "text_token_lens": torch.tensor([length]),
                "packed_text_position_ids": torch.arange(curr_rope[0], curr_rope[0] + length),
                "packed_key_value_indexes": torch.empty(0, dtype=torch.long),
                "key_values_lens": torch.tensor([0]),
            },
            [length],
            [curr_rope[0] + length],
        )

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        pack = self._images([torch.zeros(3, *size) for size in image_sizes], curr_rope, new_token_ids, "vae")[0]
        count = len(pack["packed_vae_token_indexes"])
        pack["packed_init_noises"] = torch.randn(count, 3)
        for key in ("padded_images", "packed_timesteps", "patchified_vae_latent_shapes"):
            pack.pop(key)
        return pack

    def _cache(self, cache, kind, fields):
        assert len(fields["packed_seqlens"]) == 1  # one joint camera group
        assert fields["packed_seqlens"].item() == len(fields["packed_position_ids"])
        assert fields["key_values_lens"].item() == len(fields["packed_key_value_indexes"])
        self.groups.append((kind, fields))
        cache["images"] = True
        return cache

    def forward_cache_update_vit(self, cache, **fields):
        return self._cache(cache, "vit", fields)

    def forward_cache_update_vae(self, vae, cache, **fields):
        return self._cache(cache, "vae", fields)

    def forward_cache_update_text(self, cache, **fields):
        assert fields["key_values_lens"].item() == len(fields["packed_key_value_indexes"])
        cache["text"] = True
        return cache

    def _forward_flow(self, x_t, timestep, past_key_values, **fields):
        assert len(fields["packed_seqlens"]) == 1
        assert len(fields["packed_vae_token_indexes"]) == x_t.shape[0]
        self.velocity_calls.append((x_t.detach().clone(), timestep[0].item(), dict(past_key_values)))
        value = (1 if past_key_values.get("images") else 0) + (2 if past_key_values.get("text") else 0)
        return torch.full_like(x_t, value)

    def forward(self, **fields):
        self.last_training = fields
        latent = fields["padded_latent"].permute(0, 2, 3, 1).reshape(-1, 3)
        time = fields["packed_timesteps"].sigmoid()
        time = self.timestep_shift * time / (1 + (self.timestep_shift - 1) * time)
        self.last_shifted_times = time.detach().clone()
        target = (torch.ones_like(latent) - latent)[time > 0]
        return {"mse": (self.scale - target).square(), "ce": None}


def make_adapter(**config):
    model = ContractBagel()
    return BagelMultiviewWorldModel(
        model,
        ContractVAE(),
        lambda text: [8, 9],
        {"bos_token_id": 1, "eos_token_id": 2, "start_of_image": 3, "end_of_image": 4},
        dict,
        config=WorldModelConfig(vit_size=(8, 8), vae_size=(8, 8), **config),
    )


def test_paper_resolution_and_segment_end_selection():
    images = torch.randint(0, 256, (3, 3, 9, 12), dtype=torch.uint8)
    config = WorldModelConfig()
    assert preprocess_views(images, config.vit_size).shape == (3, 3, 336, 448)
    assert preprocess_views(images, config.vae_size).shape == (3, 3, 384, 512)
    episode = torch.stack((images, images + 1, images + 2))
    current, goal = segment_end_example(episode, 0, 2)
    assert torch.equal(current, episode[0])
    assert torch.equal(goal, episode[2])
    with pytest.raises(ValueError, match="inside"):
        segment_end_example(episode, 0, 3)


def test_multiview_attention_graph_keeps_all_views_joint_and_goals_causal():
    allowed = torch.isfinite(world_attention_mask((6, 6, 3, 6)))
    assert allowed[:6, :6].all()
    assert not allowed[:6, 6:].any()
    assert allowed[6:12, :12].all()
    assert not allowed[12, 13:].any()
    assert allowed[-6:, :].all()  # all target views see one another
    assert not allowed[:-6, -6:].any()  # targets never leak into conditioning


def test_native_training_pack_and_flow_gradient_contract():
    adapter = make_adapter()
    current, target = torch.rand(3, 3, 8, 8), torch.rand(3, 3, 8, 8)
    result = adapter(current, target, "Subtask: place cup. Quality: 5.", time=0.25)
    fields = adapter.bagel.last_training
    assert fields["sample_lens"] == [fields["sequence_length"]]
    assert fields["padded_latent"].shape[0] == 6
    assert fields["vit_token_seqlens"].shape == (3,)
    times = fields["packed_timesteps"].sigmoid()
    assert times[:48].eq(0).all()
    torch.testing.assert_close(times[48:], torch.full((48,), 0.25))
    assert fields["mse_loss_indexes"].sum() == 48
    assert result["target_latent_elements"] == 144
    result["loss"].backward()
    assert adapter.bagel.scale.grad.abs() > 0


def test_joint_25_step_generation_and_three_branch_cfg():
    adapter = make_adapter(text_guidance=2.0, image_guidance=2.0).eval()
    result = adapter.generate(torch.rand(3, 3, 8, 8), "place cup", noise=torch.zeros(48, 3))
    assert result.shape == (3, 3, 8, 8)
    assert len(adapter.bagel.velocity_calls) == 25 * 3
    # full=3, no-text=1, no-image=2 -> text=5 -> image-guided=8.
    torch.testing.assert_close(adapter.vae.last_latents, torch.full((3, 3, 4, 4), -8.0))
    assert [kind for kind, _ in adapter.bagel.groups] == ["vit", "vae", "vit", "vae"]
    for start in range(0, 75, 3):
        values = adapter.bagel.velocity_calls[start : start + 3]
        # Figure19's bottom branch is (-text,+img), not (-text,-img).
        assert [entry[2] for entry in values] == [
            {"images": True, "text": True},
            {"images": True},
            {"text": True},
        ]
        torch.testing.assert_close(values[0][0], values[1][0])
        torch.testing.assert_close(values[1][0], values[2][0])


def test_training_dropout_matches_compacted_conditioning_positions():
    adapter = make_adapter()
    images = torch.rand(3, 3, 8, 8)
    full = adapter.prepare_training(images, images, "task", time=0.5)
    dropped = adapter.prepare_training(images, images, "task", time=0.5, drop_images=True, drop_text=True)
    # Two source groups of 54 tokens, four text tokens, then 54 target tokens.
    mask = dropped["nested_attention_masks"][0]
    assert torch.isneginf(mask[112:, :112]).all()
    assert torch.isfinite(mask[112:, 112:]).all()
    torch.testing.assert_close(dropped["packed_position_ids"][112:], full["packed_position_ids"][112:] - 10)


def test_guidance_one_is_conditioned_prediction():
    full, no_text, no_image = torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3)
    torch.testing.assert_close(three_branch_guidance(full, no_text, no_image), full)


def test_modes_shapes_and_noise_are_validated():
    adapter = make_adapter()
    images = torch.rand(3, 3, 8, 8)
    with pytest.raises(RuntimeError, match="eval"):
        adapter.generate(images, "task")
    with pytest.raises(ValueError, match="strictly"):
        adapter(images, images, "task", time=0)
    adapter.eval()
    with pytest.raises(RuntimeError, match="train"):
        adapter(images, images, "task")
    with pytest.raises(ValueError, match="noise"):
        adapter.generate(images, "task", noise=torch.zeros(1, 3))
    with pytest.raises(ValueError, match="3 camera"):
        preprocess_views(images[:2], (8, 8))


def test_timestep_shift_applies_to_native_training_and_sampling():
    adapter = make_adapter(timestep_shift=3.0, denoising_steps=4)
    assert adapter.bagel.timestep_shift == adapter.bagel.config.timestep_shift == 3.0
    images = torch.rand(3, 3, 8, 8)
    adapter(images, images, "task", time=0.25)
    assert adapter.bagel.last_shifted_times[:48].eq(0).all()
    torch.testing.assert_close(adapter.bagel.last_shifted_times[48:], torch.full((48,), 0.5))
    adapter.eval().generate(images, "task", noise=torch.zeros(48, 3))
    times = torch.tensor([call[1] for call in adapter.bagel.velocity_calls])
    raw_times = torch.tensor([1.0, 0.75, 0.5, 0.25])
    torch.testing.assert_close(times, 3 * raw_times / (1 + 2 * raw_times))


class ContractGoalModel(nn.Module):
    """Runtime bridge test double; does not pretend to load BAGEL weights."""

    def generate(self, images, prompt):
        assert not self.training
        self.last_images, self.last_prompt = images.clone(), prompt
        return images.float() / 255 if images.dtype == torch.uint8 else images.float()


def make_goal_request(images, *, subtask="fold sleeve", metadata=None, camera_names=None):
    observation = {"images": images, "state": "state must not reach BAGEL"}
    if camera_names is not None:
        observation["camera_names"] = camera_names
    return GenerationRequest(
        observation=observation,
        history=(),
        context=RuntimeContext("overall task", subtask, "old goal must not reach BAGEL", metadata, "joint"),
        subtask_history=("history must not reach BAGEL",),
        timestep=1,
        timestamp=0.02,
    )


def test_goal_policy_runtime_bridge_preserves_views_pixels_and_prompt():
    images = np.arange(4 * 5 * 7 * 3, dtype=np.uint8).reshape(4, 5, 7, 3)
    request = make_goal_request(
        images,
        metadata={"speed": 1260, "quality": 5, "mistake": False},
        camera_names=("front", "left_wrist", "right_wrist", "rear"),
    )
    model = ContractGoalModel()
    result = BagelGoalPolicy(model)(request)
    torch.testing.assert_close(model.last_images, torch.from_numpy(images[:3].transpose(0, 3, 1, 2)))
    assert model.last_prompt == "Subtask: fold sleeve. Speed: 1500. Quality: 5. Mistake: false."
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, images[:3].astype(np.float32) / 255)


def test_goal_policy_missing_subtask_uses_explicit_task_fallback():
    images = np.full((3, 5, 7, 3), 0.25, dtype=np.float32)
    model = ContractGoalModel()
    result = BagelGoalPolicy(model)(make_goal_request(images, subtask=None))
    assert model.last_prompt == "Subtask: overall task. Quality: 5. Mistake: false."
    np.testing.assert_array_equal(result, images)


@pytest.mark.parametrize(
    ("images", "names", "match"),
    [
        (np.zeros((2, 5, 7, 3), dtype=np.uint8), None, "three cameras"),
        (np.zeros((3, 3, 5, 7), dtype=np.uint8), None, "channels last"),
        (np.zeros((3, 5, 7, 3), dtype=np.uint8), ("rear", "left_wrist", "right_wrist"), "ordering"),
        (np.full((3, 5, 7, 3), 2.0), None, "finite and in"),
    ],
)
def test_goal_policy_rejects_invalid_camera_input(images, names, match):
    with pytest.raises(ValueError, match=match):
        BagelGoalPolicy(ContractGoalModel())(make_goal_request(images, camera_names=names))


def test_goal_policy_rejects_invalid_world_output():
    model = ContractGoalModel()
    model.generate = lambda images, prompt: torch.full((3, 3, 5, 7), torch.nan)
    with pytest.raises(ValueError, match="finite floating RGB"):
        BagelGoalPolicy(model)(make_goal_request(np.zeros((3, 5, 7, 3), dtype=np.uint8)))
