"""Three-view BAGEL adapter for pi0.7 subgoal training and generation.

Native API reference: ByteDance-Seed/Bagel commit
a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f, modeling/bagel/bagel.py.
The upstream code is Apache-2.0; this module is an independently written adapter.

Supply an actual constructed BAGEL model, its pretrained AutoEncoder, tokenizer,
special token IDs and official NaiveCache factory. This module neither downloads
weights nor substitutes a smaller network. Stock BAGEL needs its own compatible
CUDA/FlashAttention environment. CPU contract tests do NOT verify that integration.

All three camera views belong to ONE attention group at each modality stage.
Training uses native Bagel.forward's flow objective. Inference repacks the native
cache APIs and calls native _forward_flow, bypassing generate_image's single-image
unpacking assumption. Camera views are never collaged or generated sequentially.
"""

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from openpi.pi07.context import EpisodeMetadata
from openpi.pi07.context import bin_episode_steps
from openpi.pi07.data import _images
from openpi.pi07.runtime import GenerationRequest

BAGEL_API_REVISION = "a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f"


@dataclass(frozen=True)
class WorldModelConfig:
    # Paper dimensions are width x height; tensors use height x width.
    vit_size: tuple[int, int] = (336, 448)
    vae_size: tuple[int, int] = (384, 512)
    denoising_steps: int = 25
    # The paper does not publish timestep shift or guidance strengths.
    timestep_shift: float = 1.0
    text_guidance: float = 1.0
    image_guidance: float = 1.0

    def __post_init__(self):
        for shape in (self.vit_size, self.vae_size):
            if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
                raise ValueError("image sizes must contain positive integer height and width")
        if (
            isinstance(self.denoising_steps, bool)
            or not isinstance(self.denoising_steps, int)
            or self.denoising_steps < 1
        ):
            raise ValueError("denoising_steps must be a positive integer")
        if not math.isfinite(self.timestep_shift) or self.timestep_shift <= 0:
            raise ValueError("timestep_shift must be finite and positive")
        if any(not math.isfinite(x) or x < 1 for x in (self.text_guidance, self.image_guidance)):
            raise ValueError("guidance strengths must be finite and at least one")


def preprocess_views(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Three ordered RGB cameras, uint8 or float [0,1], to normalized CHW."""
    if images.ndim != 4 or images.shape[:2] != (3, 3):
        raise ValueError("images must be [3 camera views, 3 RGB channels, height, width]")
    if images.dtype == torch.uint8:
        pixels = images.float() / 255
    elif images.is_floating_point():
        if not torch.isfinite(images).all() or torch.any((images < 0) | (images > 1)):
            raise ValueError("floating images must be finite and in [0,1]")
        pixels = images.float()
    else:
        raise ValueError("images must be uint8 or floating point")
    # Match the public BAGEL transform's bicubic antialiased resize and 0.5/0.5
    # normalization, using the fixed resolutions given in the pi0.7 appendix.
    resized = functional.interpolate(pixels, size=size, mode="bicubic", align_corners=False, antialias=True)
    return resized.clamp(0, 1) * 2 - 1


def segment_end_example(images: torch.Tensor, timestep: int, segment_end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select current and annotated segment-end views from one episode [N,3,3,H,W]."""
    if images.ndim != 5 or images.shape[1:3] != (3, 3):
        raise ValueError("episode images must be [N,3,3,H,W]")
    if any(isinstance(n, bool) or not isinstance(n, int) for n in (timestep, segment_end)):
        raise ValueError("segment indices must be integers")
    if not 0 <= timestep <= segment_end < images.shape[0]:
        raise ValueError("segment end must follow the current frame inside the same episode")
    return images[timestep], images[segment_end]


def world_attention_mask(lengths: tuple[int, int, int, int], *, device=None) -> torch.Tensor:
    """Fig.19 groups: current ViT, current VAE, causal text, joint noisy goals."""
    if len(lengths) != 4 or any(n < 0 for n in lengths):
        raise ValueError("four nonnegative group lengths are required")
    total = sum(lengths)
    allowed = torch.zeros(total, total, dtype=torch.bool, device=device)
    offset = 0
    for index, length in enumerate(lengths):
        allowed[offset : offset + length, :offset] = True
        block = torch.ones(length, length, dtype=torch.bool, device=device)
        allowed[offset : offset + length, offset : offset + length] = block.tril() if index == 2 else block
        offset += length
    return torch.zeros(total, total, device=device).masked_fill(~allowed, -torch.inf)


def three_branch_guidance(full, without_text, without_images, *, text_scale=1.0, image_scale=1.0):
    """BAGEL's nested text/image guidance, without optional norm rescaling.

    Fig.19 branches are (+text,+image), (-text,+image), (+text,-image).
    The bottom branch retains image attention; it is not fully unconditional.
    Applying
    separate forwards is equivalent to isolated branches of the attention tree.
    Both scales equal one recovers the fully conditioned prediction exactly.
    """
    if full.shape != without_text.shape or full.shape != without_images.shape:
        raise ValueError("all three velocity branches must have identical shapes")
    text_guided = without_text + text_scale * (full - without_text)
    return without_images + image_scale * (text_guided - without_images)


def _identity(value):
    return value


class BagelMultiviewWorldModel(nn.Module):
    """Native BAGEL three-view training/inference adapter; one episode per call.

    ``cache_factory`` is normally
    ``lambda: NaiveCache(bagel.config.llm_config.num_hidden_layers)``. ``prompt``
    must contain the subtask and any desired episode metadata. Supply models in
    training mode for ``forward`` and call ``eval`` before ``generate``. Upstream
    BAGEL dispatches different code paths based on that mode.
    """

    def __init__(
        self,
        bagel: nn.Module,
        vae: nn.Module,
        tokenizer: Any,
        special_token_ids: dict[str, int],
        cache_factory: Callable[[], Any],
        *,
        config: WorldModelConfig | None = None,
    ):
        super().__init__()
        config = config or WorldModelConfig()
        required = (
            "prepare_vit_images",
            "prepare_vae_images",
            "prepare_prompts",
            "prepare_vae_latent",
            "forward_cache_update_vit",
            "forward_cache_update_vae",
            "forward_cache_update_text",
            "_forward_flow",
        )
        if any(not callable(getattr(bagel, name, None)) for name in required):
            raise TypeError("bagel does not implement the pinned official BAGEL module contract")
        if set(special_token_ids) != {"bos_token_id", "eos_token_id", "start_of_image", "end_of_image"}:
            raise ValueError("supply the four token IDs returned by official BAGEL add_special_tokens")
        if any(not isinstance(value, int) or value < 0 for value in special_token_ids.values()):
            raise ValueError("BAGEL special token IDs must be nonnegative integers")
        if not callable(cache_factory):
            raise TypeError("cache_factory must construct a fresh official NaiveCache")
        self.bagel, self.vae, self.tokenizer = bagel, vae, tokenizer
        self.special_token_ids, self.cache_factory, self.config = dict(special_token_ids), cache_factory, config
        for size, stride in ((config.vit_size, bagel.vit_patch_size), (config.vae_size, bagel.latent_downsample)):
            if any(n % stride for n in size):
                raise ValueError("configured sizes must be divisible by their BAGEL patch/downsample strides")
        # Native Bagel.forward shifts training times using this attribute. Apply
        # our explicit setting to both training and the Euler sampling schedule.
        self.bagel.timestep_shift = config.timestep_shift
        if getattr(self.bagel, "config", None) is not None:
            self.bagel.config.timestep_shift = config.timestep_shift
        # Same switch set by upstream generate_image before native _forward_flow.
        self.bagel.language_model.model.enable_taylorseer = False

    @property
    def device(self):
        return next(self.bagel.parameters()).device

    def _move(self, fields):
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in fields.items()
        }

    def _autocast(self):
        dtype = next(self.bagel.parameters()).dtype
        if self.device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=dtype)
        return nullcontext()

    def _views(self, images, *, kind, rope=0):
        size = self.config.vit_size if kind == "vit" else self.config.vae_size
        # Native preparation builds CPU index tensors/padding. Transfer the
        # assembled dictionary together after all three views are packed.
        pixels = preprocess_views(images, size).detach().cpu()
        method = self.bagel.prepare_vit_images if kind == "vit" else self.bagel.prepare_vae_images
        fields, _, _ = method(
            curr_kvlens=[0, 0, 0],
            curr_rope=[rope, rope + 1, rope + 2],
            images=list(pixels),
            transforms=_identity,
            new_token_ids=self.special_token_ids,
        )
        return fields

    def _text(self, prompt, rope):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must contain a nonempty subtask/metadata description")
        fields, _, next_rope = self.bagel.prepare_prompts(
            curr_kvlens=[0],
            curr_rope=[rope],
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.special_token_ids,
        )
        return fields, next_rope[0]

    @staticmethod
    def _merge_group(fields, prefix_length):
        """Convert three independent image packs to one cross-view query group."""
        result = dict(fields)
        length = int(result["packed_seqlens"].sum())
        result["packed_seqlens"] = torch.tensor([length], dtype=torch.int32)
        result["packed_indexes"] = torch.arange(prefix_length, prefix_length + length)
        result["packed_key_value_indexes"] = torch.arange(prefix_length)
        result["key_values_lens"] = torch.tensor([prefix_length], dtype=torch.int32)
        return result

    def prepare_training(
        self,
        current_images,
        target_images,
        prompt,
        *,
        time: float | None = None,
        drop_text: bool = False,
        drop_images: bool = False,
    ):
        """Assemble actual Bagel.forward inputs; clean current latents use -inf.

        BAGEL applies sigmoid to packed_timesteps internally, so supplying zero
        would incorrectly add 50% noise to current observations. Target time is
        uniform in (0,1) by default; this sampling choice is not disclosed in pi0.7.
        """
        if time is None:
            time = float(torch.rand(()).clamp(1e-5, 1 - 1e-5))
        if not isinstance(time, (float, int)) or not math.isfinite(time) or not 0 < time < 1:
            raise ValueError("training flow time must be strictly between zero and one")
        if not isinstance(drop_text, bool) or not isinstance(drop_images, bool):
            raise ValueError("conditioning dropout flags must be boolean")
        vit = self._views(current_images, kind="vit")
        source = self._views(current_images, kind="vae", rope=3)
        text, goal_rope = self._text(prompt, 6)
        target = self._views(target_images, kind="vae", rope=goal_rope)
        lengths = (
            int(vit["packed_seqlens"].sum()),
            int(source["packed_seqlens"].sum()),
            int(text["text_token_lens"].sum()),
            int(target["packed_seqlens"].sum()),
        )
        offsets = (0, lengths[0], sum(lengths[:2]), sum(lengths[:3]))
        packs = (vit, source, text, target)
        source_indexes = source["packed_vae_token_indexes"] + offsets[1]
        target_indexes = target["packed_vae_token_indexes"] + offsets[3]
        mse_indexes = torch.zeros(sum(lengths), dtype=torch.bool)
        mse_indexes[target_indexes] = True
        with torch.no_grad(), self._autocast():
            latent = self.vae.encode(torch.cat((source["padded_images"], target["padded_images"])).to(self.device))
        fields = {
            "sequence_length": sum(lengths),
            "sample_lens": [sum(lengths)],
            "packed_text_ids": torch.cat([pack["packed_text_ids"] for pack in packs]),
            "packed_text_indexes": torch.cat(
                [pack["packed_text_indexes"] + offset for pack, offset in zip(packs, offsets, strict=True)]
            ),
            "packed_position_ids": torch.cat(
                (
                    vit["packed_position_ids"],
                    source["packed_position_ids"],
                    text["packed_text_position_ids"],
                    target["packed_position_ids"],
                )
            ),
            "nested_attention_masks": [world_attention_mask(lengths, device=self.device)],
            "packed_vit_tokens": vit["packed_vit_tokens"],
            "packed_vit_token_indexes": vit["packed_vit_token_indexes"],
            "packed_vit_position_ids": vit["packed_vit_position_ids"],
            "vit_token_seqlens": vit["vit_token_seqlens"],
            "padded_latent": latent,
            "patchified_vae_latent_shapes": source["patchified_vae_latent_shapes"]
            + target["patchified_vae_latent_shapes"],
            "packed_latent_position_ids": torch.cat(
                (source["packed_vae_position_ids"], target["packed_vae_position_ids"])
            ),
            "packed_vae_token_indexes": torch.cat((source_indexes, target_indexes)),
            "packed_timesteps": torch.cat(
                (
                    torch.full((len(source_indexes),), -torch.inf),
                    torch.full((len(target_indexes),), math.log(time / (1 - time))),
                )
            ),
            "mse_loss_indexes": mse_indexes,
        }
        # Explicit per-example dropout supports the three inference branches.
        # Sampling probabilities are left to the data recipe because pi0.7 does
        # not disclose its world-model dropout distribution. Remove dropped
        # conditioning keys and compact retained RoPE positions equivalently to
        # inference, where those blocks are omitted from the cache entirely.
        attention = fields["nested_attention_masks"][0]
        if drop_images:
            attention[offsets[2] :, : offsets[2]] = -torch.inf
            fields["packed_position_ids"][offsets[2] :] -= 6
        if drop_text:
            attention[offsets[3] :, offsets[2] : offsets[3]] = -torch.inf
            fields["packed_position_ids"][offsets[3] :] -= lengths[2]
        return self._move(fields)

    def forward(
        self,
        current_images,
        target_images,
        prompt,
        *,
        time: float | None = None,
        drop_text: bool = False,
        drop_images: bool = False,
    ):
        """Native joint CFM loss over all target-view latent coordinates."""
        if not self.bagel.training:
            raise RuntimeError("call train() before native BAGEL training; its forward dispatch depends on mode")
        fields = self.prepare_training(
            current_images, target_images, prompt, time=time, drop_text=drop_text, drop_images=drop_images
        )
        with self._autocast():
            losses = self.bagel(**fields)
        mse = losses["mse"]
        if mse is None or mse.numel() == 0 or not torch.isfinite(mse).all():
            raise ValueError("native BAGEL returned no finite target flow loss")
        return {"loss": mse.float().mean(), "flow_loss": mse.float().mean(), "target_latent_elements": mse.numel()}

    def _context(self, images, prompt, *, include_images, include_text):
        cache, length, rope = self.cache_factory(), 0, 0
        if include_images:
            for kind in ("vit", "vae"):
                fields = self._merge_group(self._views(images, kind=kind, rope=rope), length)
                fields = self._move(fields)
                if kind == "vit":
                    cache = self.bagel.forward_cache_update_vit(cache, **fields)
                else:
                    cache = self.bagel.forward_cache_update_vae(self.vae, cache, **fields)
                length += int(fields["packed_seqlens"].sum())
                rope += 3
        if include_text:
            fields, rope = self._text(prompt, rope)
            # prepare_prompts used zero cache lengths; add the one shared prefix.
            count = int(fields["text_token_lens"].sum())
            fields["packed_text_indexes"] = torch.arange(length, length + count)
            fields["packed_key_value_indexes"] = torch.arange(length)
            fields["key_values_lens"] = torch.tensor([length], dtype=torch.int32)
            cache = self.bagel.forward_cache_update_text(cache, **self._move(fields))
            length += count
        return cache, length, rope

    def _targets(self, length, rope):
        fields = self.bagel.prepare_vae_latent(
            curr_kvlens=[0, 0, 0],
            curr_rope=[rope, rope + 1, rope + 2],
            image_sizes=[self.config.vae_size] * 3,
            new_token_ids=self.special_token_ids,
        )
        return self._move(self._merge_group(fields, length))

    @torch.no_grad()
    def generate(self, current_images, prompt, *, noise: torch.Tensor | None = None):
        """Jointly generate [3,3,H,W] RGB goals using 25 Euler steps by default.

        Separate full/no-text/no-image forwards implement the three CFG branches.
        Norm rescaling, quantization, tensor parallelism and SageAttention from
        the paper are not implemented. Native CUDA execution remains unverified.
        """
        if self.bagel.training:
            raise RuntimeError("call eval() before native BAGEL image generation")
        # Validate even if a guidance branch would omit its conditioning input.
        preprocess_views(current_images, self.config.vit_size)
        self._text(prompt, 0)
        with self._autocast():
            branches = [self._context(current_images, prompt, include_images=True, include_text=True)]
            if self.config.text_guidance > 1 or self.config.image_guidance > 1:
                branches.extend(
                    (
                        self._context(current_images, prompt, include_images=True, include_text=False),
                        self._context(current_images, prompt, include_images=False, include_text=True),
                    )
                )
            target_packs = [self._targets(length, rope) for _, length, rope in branches]
            shape = target_packs[0]["packed_init_noises"].shape
            if noise is not None and (noise.shape != shape or not torch.isfinite(noise).all()):
                raise ValueError("noise must match the joint three-view packed latent shape and be finite")
            x = target_packs[0]["packed_init_noises"] if noise is None else noise.to(self.device).clone()
            times = torch.linspace(1, 0, self.config.denoising_steps + 1, device=self.device)
            shift = self.config.timestep_shift
            times = shift * times / (1 + (shift - 1) * times)
            for index in range(self.config.denoising_steps):
                velocities = []
                for (cache, _, _), fields in zip(branches, target_packs, strict=True):
                    inputs = {key: value for key, value in fields.items() if key != "packed_init_noises"}
                    velocity = self.bagel._forward_flow(  # noqa: SLF001
                        x_t=x,
                        timestep=times[index].expand(x.shape[0]),
                        past_key_values=cache,
                        cfg_text_scale=1.0,
                        cfg_img_scale=1.0,
                        **inputs,
                    )
                    if velocity.shape != x.shape or not torch.isfinite(velocity).all():
                        raise ValueError("BAGEL returned an invalid joint multiview velocity")
                    velocities.append(velocity)
                velocity = (
                    velocities[0]
                    if len(velocities) == 1
                    else three_branch_guidance(
                        *velocities, text_scale=self.config.text_guidance, image_scale=self.config.image_guidance
                    )
                )
                x = x - (times[index] - times[index + 1]) * velocity
            h, w = (n // self.bagel.latent_downsample for n in self.config.vae_size)
            p, channels = self.bagel.latent_patch_size, self.bagel.latent_channel
            latent = x.reshape(3, h, w, p, p, channels).permute(0, 5, 1, 3, 2, 4).reshape(3, channels, h * p, w * p)
            decoded = self.vae.decode(latent)
        expected = (3, 3, *self.config.vae_size)
        if decoded.shape != expected or not torch.isfinite(decoded).all():
            raise ValueError("VAE did not decode three finite goal images at the configured resolution")
        return (decoded.float() * 0.5 + 0.5).clamp(0, 1)


class BagelGoalPolicy:
    """Bridge runtime requests to a supplied trained BAGEL world model.

    Observations contain ``images[V,H,W,3]`` ordered as front, left wrist,
    right wrist, then optional rear. The first three views are used; the rear
    view is ignored. Optional ``camera_names`` must confirm this exact order.
    Inputs are uint8 or floating RGB [0,1]; goals return float32 NumPy VHWC.

    Prompts contain only the subtask and episode metadata. A missing subtask
    explicitly falls back to the overall task. Missing metadata uses quality 5
    and mistake false, consistent with the action-policy runtime adapter.
    State, history, existing goals and control mode are not world-model inputs.
    Constructing this adapter puts the supplied world model in evaluation mode.
    """

    def __init__(self, world_model: BagelMultiviewWorldModel):
        self.world_model = world_model.eval()

    @torch.inference_mode()
    def __call__(self, request: GenerationRequest) -> np.ndarray:
        observation = request.observation
        if not isinstance(observation, Mapping) or "images" not in observation:
            raise ValueError("World-model observations require 'images' [V,H,W,3]")
        images = _images(observation["images"], "world-model observation images", 4)
        if images.shape[0] not in (3, 4):
            raise ValueError("World-model observations require three cameras and an optional fourth rear camera")
        names = ("front", "left_wrist", "right_wrist", "rear")[: images.shape[0]]
        if "camera_names" in observation and tuple(observation["camera_names"]) != names:
            raise ValueError("camera_names must follow front, left_wrist, right_wrist, optional rear ordering")
        instruction = request.context.subtask
        if instruction is None:
            instruction = request.context.task
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("World-model requests require a nonempty subtask or fallback task")
        metadata = request.context.metadata
        if metadata is None:
            metadata = EpisodeMetadata(quality=5, mistake=False)
        elif isinstance(metadata, Mapping):
            metadata = EpisodeMetadata(**metadata)
        if not isinstance(metadata, EpisodeMetadata):
            raise TypeError("Runtime metadata must be EpisodeMetadata, its field mapping, or None")
        prompt = [f"Subtask: {instruction.strip().rstrip('.')}."]
        if metadata.speed is not None:
            prompt.append(f"Speed: {bin_episode_steps(metadata.speed)}.")
        if metadata.quality is not None:
            prompt.append(f"Quality: {metadata.quality}.")
        if metadata.mistake is not None:
            prompt.append(f"Mistake: {str(metadata.mistake).lower()}.")
        pixels = torch.from_numpy(np.ascontiguousarray(images[:3].transpose(0, 3, 1, 2)))
        goals = self.world_model.generate(pixels, " ".join(prompt))
        if (
            not isinstance(goals, torch.Tensor)
            or goals.ndim != 4
            or goals.shape[:2] != (3, 3)
            or min(goals.shape[2:]) < 1
            or not goals.is_floating_point()
            or not torch.isfinite(goals).all()
            or torch.any((goals < 0) | (goals > 1))
        ):
            raise ValueError("World model must return three finite floating RGB goals [3,3,H,W] in [0,1]")
        return goals.permute(0, 2, 3, 1).float().cpu().numpy().copy()
