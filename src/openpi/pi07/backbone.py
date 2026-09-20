"""Gemma 3 and parameter-sharing visual memory for the pi0.7 reimplementation.

This adapter uses the actual Transformers 4.53.2 Gemma 3 modules and checkpoint
format. No pretrained weights are downloaded by construction from a config.

The MEM paper does not unambiguously specify the fusion of spatial and temporal
attention. Here every fourth vision layer uses one softmax over same-frame
spatial keys and strict-past same-patch temporal keys. This explicit reproduction
choice has separable attention cost, no new learned parameters, and exactly the
original image encoder when there is only one valid frame. It is not a claim to
recover Physical Intelligence's unpublished implementation.
"""

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from transformers import Gemma3Config
from transformers import Gemma3ForConditionalGeneration
from transformers import Gemma3TextConfig
from transformers import SiglipVisionConfig
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class BackboneOutput:
    last_hidden_state: torch.Tensor
    # Input to each decoder layer, followed by the final normalized output.
    hidden_states: tuple[torch.Tensor, ...]
    # Actual normalized, rotary-positioned keys and values, not re-projections.
    # Each tensor is [batch, num_kv_heads, sequence, head_dim].
    key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    logits: torch.Tensor | None = None


def temporal_position_encoding(length: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Fixed sine/cosine features with exactly zero encoding for current time."""
    times = torch.arange(1 - length, 1, dtype=torch.float32, device=device)
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(0, width, 2, device=device) / width)
    angles = times[:, None] * frequencies[None, :]
    # Subtract the t=0 encoding; no learned parameters are introduced.
    encoded = torch.stack((angles.sin(), angles.cos() - 1), dim=-1).flatten(-2)
    return encoded[:, :width].to(dtype)


def _memory_attention(layer: nn.Module, hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """One SigLIP layer with spatial plus causal same-patch temporal attention."""
    batch, frames, patches, width = hidden.shape
    attention = layer.self_attn
    positional = temporal_position_encoding(frames, width, device=hidden.device, dtype=hidden.dtype)
    positioned = hidden + positional[None, :, None, :]
    normalized = layer.layer_norm1(positioned)

    def project(projection: nn.Module) -> torch.Tensor:
        return projection(normalized).view(batch, frames, patches, attention.num_heads, attention.head_dim)

    query, key, value = project(attention.q_proj), project(attention.k_proj), project(attention.v_proj)
    # [batch, frame, head, query_patch, key_patch]
    spatial_scores = torch.einsum("btnhd,btmhd->bthnm", query.float(), key.float()) * attention.scale
    # [batch, query_frame, head, patch, key_frame]
    temporal_scores = torch.einsum("btnhd,bsnhd->bthns", query.float(), key.float()) * attention.scale
    # Current-frame keys already occur in spatial attention, so avoid duplication.
    past = torch.arange(frames, device=hidden.device)[None, :] < torch.arange(frames, device=hidden.device)[:, None]
    temporal_allowed = past[None, :, None, None, :] & valid[:, None, None, None, :]
    temporal_scores = temporal_scores.masked_fill(~temporal_allowed, -torch.inf)
    log_partition = torch.logaddexp(
        spatial_scores.logsumexp(-1, keepdim=True), temporal_scores.logsumexp(-1, keepdim=True)
    )
    spatial_weights = (spatial_scores - log_partition).exp().to(value.dtype)
    temporal_weights = (temporal_scores - log_partition).exp().to(value.dtype)
    spatial_weights = functional.dropout(spatial_weights, p=attention.dropout, training=layer.training)
    temporal_weights = functional.dropout(temporal_weights, p=attention.dropout, training=layer.training)
    attended = torch.einsum("bthnm,btmhd->btnhd", spatial_weights, value)
    attended = attended + torch.einsum("bthns,bsnhd->btnhd", temporal_weights, value)
    hidden = positioned + attention.out_proj(attended.reshape(batch, frames, patches, width))
    hidden = hidden + layer.mlp(layer.layer_norm2(hidden))
    return hidden.masked_fill(~valid[:, :, None, None], 0)


class Gemma3Backbone(nn.Module):
    """Pretrained-compatible visual/text backbone with caller-defined attention.

    Images are channel-first RGB. uint8 inputs are scaled from [0,255] to [-1,1];
    floating inputs must already use SigLIP's [-1,1] normalization. Histories are
    chronological, with the current observation in the final slot. A masked
    current observation masks the entire camera's output. Masked historical
    frames do not influence the current tokens.
    """

    def __init__(self, hf_model: Gemma3ForConditionalGeneration, *, image_size: int = 448, temporal_every: int = 4):
        super().__init__()
        if not isinstance(hf_model, Gemma3ForConditionalGeneration):
            raise TypeError("hf_model must be a Gemma3ForConditionalGeneration instance")
        patch_size = hf_model.config.vision_config.patch_size
        if image_size <= 0 or image_size % patch_size:
            raise ValueError("image_size must be a positive multiple of the vision patch size")
        if temporal_every <= 0:
            raise ValueError("temporal_every must be positive")
        self.hf_model = hf_model
        self.image_size = image_size
        self.temporal_every = temporal_every
        # The pretrained projector contains no learned pooling weights. Preserve
        # its token count while adapting the 896px pretrained patch grid to 448px.
        projector = self.hf_model.model.multi_modal_projector
        if image_size // patch_size < projector.tokens_per_side:
            raise ValueError("image_size produces fewer patches than the configured image token grid")
        # Eager attention obeys arbitrary 4-D masks, including non-causal blocks.
        self.hf_model.config.text_config._attn_implementation = "eager"  # noqa: SLF001
        self.hf_model.config.vision_config._attn_implementation = "eager"  # noqa: SLF001

    @classmethod
    def from_config(cls, config: Gemma3Config | dict[str, Any], **kwargs: Any) -> "Gemma3Backbone":
        if isinstance(config, dict):
            config = Gemma3Config.from_dict(config)
        return cls(Gemma3ForConditionalGeneration(config), **kwargs)

    @classmethod
    def from_pretrained(
        cls, checkpoint: str, *, image_size: int = 448, temporal_every: int = 4, **kwargs: Any
    ) -> "Gemma3Backbone":
        kwargs.setdefault("attn_implementation", "eager")
        model = Gemma3ForConditionalGeneration.from_pretrained(checkpoint, **kwargs)
        return cls(model, image_size=image_size, temporal_every=temporal_every)

    @classmethod
    def tiny(cls, *, image_size: int = 16) -> "Gemma3Backbone":
        """Small randomly initialized real Gemma 3 model for offline smoke tests.

        This is never a fallback for missing pretrained weights. Dimensions are
        intentionally unrelated to the paper's model size.
        """
        text = Gemma3TextConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            query_pre_attn_scalar=8,
            max_position_embeddings=1024,
            sliding_window=128,
            layer_types=["sliding_attention", "full_attention"],
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        vision = SiglipVisionConfig(
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=4,
            image_size=image_size,
            patch_size=2,
            vision_use_head=False,
        )
        config = Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(), mm_tokens_per_image=4)
        return cls.from_config(config, image_size=image_size)

    def resize_token_embeddings(self, size: int) -> nn.Embedding:
        """Expand the real tied Gemma vocabulary for disjoint FAST tokens."""
        return self.hf_model.resize_token_embeddings(size)

    @property
    def width(self) -> int:
        return self.hf_model.config.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.hf_model.config.text_config.vocab_size

    @property
    def depth(self) -> int:
        return self.hf_model.config.text_config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self.hf_model.config.text_config.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.hf_model.config.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.hf_model.config.text_config.head_dim

    @property
    def image_tokens(self) -> int:
        return self.hf_model.config.mm_tokens_per_image

    def embed_text(self, ids: torch.Tensor) -> torch.Tensor:
        # Gemma3TextScaledWordEmbedding applies sqrt(width) exactly once here.
        return self.hf_model.get_input_embeddings()(ids)

    def decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.hf_model.lm_head(hidden_states)
        softcap = self.hf_model.config.text_config.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def rope(
        self, position_ids: torch.Tensor, layer_index: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [B,S,head_dim] rotary cosine/sine for an expert layer."""
        model = self.hf_model.model.language_model
        if not 0 <= layer_index < self.depth:
            raise ValueError("layer_index out of bounds")
        module = model.rotary_emb_local if model.layers[layer_index].self_attn.is_sliding else model.rotary_emb
        dummy = torch.empty((), dtype=dtype, device=position_ids.device)
        return module(dummy, position_ids)

    def encode_images(self, images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError("images must be [batch, views, frames, 3, height, width]")
        if mask.shape != images.shape[:3] or mask.dtype != torch.bool:
            raise ValueError("image mask must be boolean [batch, views, frames]")
        batch, views, frames, channels, height, width = images.shape
        if not views or not frames:
            raise ValueError("images need at least one view and frame")
        vision = self.hf_model.model.vision_tower.vision_model
        dtype = vision.embeddings.patch_embedding.weight.dtype
        pixels = images.reshape(batch * views * frames, channels, height, width)
        pixels = pixels.to(dtype)
        if images.dtype == torch.uint8:
            pixels = pixels / 127.5 - 1
        pixels = functional.interpolate(
            pixels, (self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        hidden = vision.embeddings(pixels, interpolate_pos_encoding=True)
        patches, visual_width = hidden.shape[1:]
        valid = mask.reshape(batch * views, frames)
        hidden = hidden.reshape(batch * views, frames, patches, visual_width)
        hidden = hidden.masked_fill(~valid[:, :, None, None], 0)
        for index, layer in enumerate(vision.encoder.layers):
            if frames > 1 and (index + 1) % self.temporal_every == 0:
                hidden = _memory_attention(layer, hidden, valid)
            else:
                hidden = layer(hidden.flatten(0, 1), attention_mask=None)[0].reshape(
                    batch * views, frames, patches, visual_width
                )
                hidden = hidden.masked_fill(~valid[:, :, None, None], 0)
        hidden = vision.post_layernorm(hidden[:, -1])
        projector = self.hf_model.model.multi_modal_projector
        grid = self.image_size // self.hf_model.config.vision_config.patch_size
        feature_map = hidden.transpose(1, 2).reshape(batch * views, visual_width, grid, grid)
        pooled = functional.adaptive_avg_pool2d(feature_map, (projector.tokens_per_side, projector.tokens_per_side))
        pooled = pooled.flatten(2).transpose(1, 2)
        tokens = projector.mm_soft_emb_norm(pooled) @ projector.mm_input_projection_weight
        tokens = tokens.to(dtype)
        token_mask = valid[:, -1, None].expand(batch * views, self.image_tokens)
        tokens = tokens.masked_fill(~token_mask[:, :, None], 0)
        return tokens.reshape(batch, views * self.image_tokens, self.width), token_mask.reshape(batch, -1)

    def forward(
        self,
        embeddings: torch.Tensor,
        allowed: torch.Tensor,
        valid: torch.Tensor,
        *,
        return_logits: bool = True,
        position_ids: torch.Tensor | None = None,
    ) -> BackboneOutput:
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.width:
            raise ValueError("embeddings must be [batch, sequence, backbone_width]")
        batch, length, _ = embeddings.shape
        if allowed.shape != (batch, length, length) or allowed.dtype != torch.bool:
            raise ValueError("allowed must be boolean [batch, sequence, sequence]")
        if valid.shape != (batch, length) or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [batch, sequence]")
        if not length:
            raise ValueError("the prefix cannot be empty")
        permitted = allowed & valid[:, :, None] & valid[:, None, :]
        if torch.any(valid & ~permitted.any(-1)):
            raise ValueError("every valid query needs at least one allowed valid key")
        # Padded queries attend only themselves to avoid undefined all-masked
        # softmax rows. Their activations and K/V are masked before returning.
        permitted = permitted | (~valid[:, :, None] & torch.eye(length, dtype=torch.bool, device=valid.device)[None])
        mask = torch.zeros((batch, 1, length, length), dtype=embeddings.dtype, device=embeddings.device)
        mask.masked_fill_(~permitted[:, None], torch.finfo(embeddings.dtype).min)
        if position_ids is None:
            position_ids = (valid.long().cumsum(-1) - 1).clamp_min(0)
        if position_ids.shape != (batch, length):
            raise ValueError("position_ids must be [batch, sequence]")
        cache = DynamicCache()
        output = self.hf_model.model.language_model(
            inputs_embeds=embeddings.masked_fill(~valid[:, :, None], 0),
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        states = tuple(state.masked_fill(~valid[:, :, None], 0) for state in output.hidden_states)
        key_values = tuple(
            (key.masked_fill(~valid[:, None, :, None], 0), value.masked_fill(~valid[:, None, :, None], 0))
            for key, value in cache
        )
        return BackboneOutput(
            last_hidden_state=states[-1],
            hidden_states=states,
            key_values=key_values,
            logits=self.decode(states[-1]) if return_logits else None,
        )
