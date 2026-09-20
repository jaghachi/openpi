"""Knowledge-insulated Gemma 3 VLA with flow matching and training-time RTC.

The paper does not publish the expert dimensions. The default 34-layer, 1024-wide
expert below has approximately 858M parameters; these dimensions are an explicit
reimplementation choice, not a claim of checkpoint compatibility.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from openpi.pi07.backbone import Gemma3Backbone


@dataclass(frozen=True)
class Pi07Config:
    action_dim: int = 32
    state_dim: int = 32
    action_horizon: int = 50
    expert_width: int = 1024
    expert_mlp_dim: int = 4096
    max_delay: int = 12
    denoising_steps: int = 5
    fast_bos_id: int = 2
    fast_loss_weight: float = 1.0
    flow_loss_weight: float = 1.0

    def __post_init__(self):
        if min(self.action_dim, self.state_dim, self.action_horizon, self.expert_width, self.expert_mlp_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.expert_width % 2 or not 0 <= self.max_delay < self.action_horizon:
            raise ValueError("Expert width must be even and delay must be shorter than the horizon")
        if self.denoising_steps < 1 or min(self.fast_loss_weight, self.flow_loss_weight) < 0:
            raise ValueError("Denoising steps must be positive and loss weights nonnegative")


def prefix_attention_mask(valid: torch.Tensor, observation_length: int, text_length: int) -> torch.Tensor:
    """Figure 19: observations -> causal text -> bidirectional goals.

    State history is included in the observation block. FAST labels are appended
    separately so no action target can leak into the conditioning prefix.
    """
    length = valid.shape[1]
    pos = torch.arange(length, device=valid.device)
    groups = torch.where(pos < observation_length, 0, pos - observation_length + 1)
    groups = torch.where(pos >= observation_length + text_length, text_length + 1, groups)
    allowed = groups[None, :] <= groups[:, None]
    return allowed[None] & valid[:, :, None] & valid[:, None, :]


def time_embedding(time: torch.Tensor, width: int) -> torch.Tensor:
    periods = torch.logspace(-3, math.log10(4.0), width // 2, device=time.device, dtype=torch.float32)
    angles = time.float()[..., None] * (2 * math.pi / periods)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.modulation = nn.Linear(width, 3 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, condition):
        scale, shift, gate = self.modulation(condition).chunk(3, dim=-1)
        normalized = F.rms_norm(x.float(), (x.shape[-1],), eps=1e-6).to(x.dtype)
        return normalized * (1 + scale) + shift, gate


def _rotate(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return x * cos[:, None] + torch.cat((-second, first), dim=-1) * sin[:, None]


class ExpertLayer(nn.Module):
    def __init__(self, width, mlp_dim, heads, kv_heads, head_dim):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.attn_norm = AdaptiveRMSNorm(width)
        self.ffn_norm = AdaptiveRMSNorm(width)
        self.q = nn.Linear(width, heads * head_dim, bias=False)
        self.k = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.v = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, width, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.gate = nn.Linear(width, mlp_dim, bias=False)
        self.up = nn.Linear(width, mlp_dim, bias=False)
        self.down = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x, condition, prefix_kv, prefix_valid, rope):
        h, attn_gate = self.attn_norm(x, condition)
        batch, length, _ = h.shape
        q = self.q_norm(self.q(h).view(batch, length, self.heads, self.head_dim).transpose(1, 2))
        k = self.k_norm(self.k(h).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2))
        v = self.v(h).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = rope
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        # KI boundary: no flow gradient reaches any backbone/vision/state weights.
        prefix_k, prefix_v = (value.detach().to(dtype=k.dtype) for value in prefix_kv)
        k = torch.cat((prefix_k, k), dim=-2)
        v = torch.cat((prefix_v, v), dim=-2)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        mask = torch.cat((prefix_valid, torch.ones(batch, length, dtype=torch.bool, device=x.device)), dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None])
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        x = x + attn_gate * self.out(attended)
        h, ffn_gate = self.ffn_norm(x, condition)
        return x + ffn_gate * self.down(F.gelu(self.gate(h), approximate="tanh") * self.up(h))


@dataclass
class EncodedContext:
    key_values: tuple
    valid: torch.Tensor


class Pi07(nn.Module):
    def __init__(self, backbone: Gemma3Backbone, config: Pi07Config | None = None):
        super().__init__()
        config = config or Pi07Config()
        self.backbone, self.config = backbone, config
        if not 0 <= config.fast_bos_id < backbone.vocab_size:
            raise ValueError("FAST beginning token is outside the backbone vocabulary")
        self.state_projection = nn.Linear(config.state_dim, backbone.width)
        self.action_in = nn.Linear(config.action_dim, config.expert_width)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.expert_width, config.expert_width),
            nn.SiLU(),
            nn.Linear(config.expert_width, config.expert_width),
            nn.SiLU(),
        )
        self.expert = nn.ModuleList(
            [
                ExpertLayer(
                    config.expert_width,
                    config.expert_mlp_dim,
                    backbone.num_heads,
                    backbone.num_kv_heads,
                    backbone.head_dim,
                )
                for _ in range(backbone.depth)
            ]
        )
        self.final_norm = nn.RMSNorm(config.expert_width, eps=1e-6)
        self.action_out = nn.Linear(config.expert_width, config.action_dim)

    def _embed_context(self, batch):
        images = batch["images"]
        # EpisodeDataset and policy inputs use RGB [0,1]; SigLIP uses [-1,1].
        images = images * 2 - 1 if images.is_floating_point() else images
        tokens, image_valid = self.backbone.encode_images(images, batch["image_mask"])
        states = batch["states"]
        if states.shape[-1] != self.config.state_dim:
            raise ValueError("State dimension does not match Pi07Config")
        state_tokens = self.state_projection(states.to(self.state_projection.weight.dtype)).to(tokens.dtype)
        observation = torch.cat((tokens, state_tokens), dim=1)
        valid = torch.cat((image_valid, batch["state_mask"].bool()), dim=1)
        text = self.backbone.embed_text(batch["input_ids"])
        embeddings = [observation, text]
        masks = [valid, batch["text_mask"].bool()]
        if "goal_images" in batch and batch["goal_images"].shape[1]:
            goal_images = batch["goal_images"]
            goal_images = goal_images * 2 - 1 if goal_images.is_floating_point() else goal_images
            goals, goal_valid = self.backbone.encode_images(goal_images.unsqueeze(2), batch["goal_mask"].unsqueeze(2))
            embeddings.append(goals)
            masks.append(goal_valid)
        valid = torch.cat(masks, dim=1)
        allowed = prefix_attention_mask(valid, observation.shape[1], text.shape[1])
        return torch.cat(embeddings, dim=1), valid, allowed

    def encode_context(self, batch, *, fast_targets=None, fast_mask=None):
        embeddings, valid, allowed = self._embed_context(batch)
        prefix_length = embeddings.shape[1]
        if fast_targets is not None:
            if fast_targets.ndim != 2 or fast_targets.shape[1] == 0:
                raise ValueError("KI training requires nonempty FAST target tokens")
            if fast_mask is None or fast_mask.shape != fast_targets.shape:
                raise ValueError("FAST mask must match target shape")
            if fast_mask.dtype != torch.bool or fast_targets.dtype not in (torch.int32, torch.int64):
                raise ValueError("FAST targets must be integer IDs with a boolean mask")
            if not fast_mask.any(dim=1).all() or torch.any(fast_mask[:, 1:] & ~fast_mask[:, :-1]):
                raise ValueError("Every FAST sequence must be nonempty and right padded")
            if torch.any((fast_targets[fast_mask] < 0) | (fast_targets[fast_mask] >= self.backbone.vocab_size)):
                raise ValueError("FAST target ID outside the backbone vocabulary")
            # Embeddings accept int32 indices, but cross_entropy requires long
            # class labels. Normalize the supported integer inputs once here.
            fast_targets = fast_targets.to(torch.long).masked_fill(~fast_mask, 0)
            # BOS predicts the first FAST token; all later positions see only prior labels.
            teacher = torch.cat(
                (torch.full_like(fast_targets[:, :1], self.config.fast_bos_id), fast_targets[:, :-1]), dim=1
            )
            embeddings = torch.cat((embeddings, self.backbone.embed_text(teacher)), dim=1)
            full_valid = torch.cat((valid, fast_mask.bool()), dim=1)
            total, count = embeddings.shape[1], fast_targets.shape[1]
            expanded = torch.zeros(embeddings.shape[0], total, total, dtype=torch.bool, device=embeddings.device)
            expanded[:, :prefix_length, :prefix_length] = allowed
            expanded[:, prefix_length:, :prefix_length] = valid[:, None]
            expanded[:, prefix_length:, prefix_length:] = (
                torch.ones(count, count, device=embeddings.device).tril().bool()
            )
            allowed = expanded & full_valid[:, :, None] & full_valid[:, None, :]
        else:
            full_valid = valid
        result = self.backbone(embeddings, allowed, full_valid, return_logits=False)
        # Discard teacher-forced action tokens before constructing the flow context.
        context = EncodedContext(
            tuple((k[:, :, :prefix_length], v[:, :, :prefix_length]) for k, v in result.key_values), valid
        )
        ce = embeddings.new_zeros(())
        if fast_targets is not None:
            logits = self.backbone.decode(result.last_hidden_state[:, prefix_length:])
            targets = fast_targets.masked_fill(~fast_mask.bool(), -100)
            ce_sum = F.cross_entropy(
                logits.float().flatten(0, 1), targets.flatten(), ignore_index=-100, reduction="sum"
            )
            ce = ce_sum / fast_mask.sum().clamp_min(1)
        return context, ce

    def velocity(self, context: EncodedContext, noisy_actions, time):
        if time.ndim == 1:
            time = time[:, None].expand(noisy_actions.shape[:2])
        condition = self.time_mlp(time_embedding(time, self.config.expert_width).to(self.action_in.weight.dtype))
        x = self.action_in(noisy_actions.to(self.action_in.weight.dtype))
        positions = context.valid.sum(dim=-1)[:, None] + torch.arange(x.shape[1], device=x.device)[None]
        for index, layer in enumerate(self.expert):
            x = layer(
                x, condition, context.key_values[index], context.valid, self.backbone.rope(positions, index, x.dtype)
            )
        return self.action_out(self.final_norm(x))

    def forward(self, batch, *, noise=None, time=None, delays=None):
        """Joint FAST CE + flow loss; noise endpoint t=1, clean data endpoint t=0."""
        actions = batch["actions"]
        if actions.shape[1:] != (self.config.action_horizon, self.config.action_dim):
            raise ValueError("Action shape does not match configured horizon/dimension")
        context, ce = self.encode_context(batch, fast_targets=batch["fast_token_ids"], fast_mask=batch["fast_mask"])
        batch_size, horizon, _ = actions.shape
        noise = torch.randn_like(actions) if noise is None else noise
        if noise.shape != actions.shape or not torch.isfinite(noise).all():
            raise ValueError("Noise must be finite and match the action shape")
        # This beta schedule comes from upstream OpenPI; pi0.7 does not publish one.
        if time is None:
            time = torch.distributions.Beta(1.5, 1.0).sample((batch_size,)).to(actions.device) * 0.999 + 0.001
        if time.shape != (batch_size,) or not torch.isfinite(time).all() or torch.any((time < 0) | (time > 1)):
            raise ValueError("Flow time must be finite, shape [batch], and in [0,1]")
        if delays is None:
            delays = torch.randint(self.config.max_delay + 1, (batch_size,), device=actions.device)
        if (
            delays.dtype not in (torch.int32, torch.int64)
            or delays.shape != (batch_size,)
            or torch.any(delays < 0)
            or torch.any(delays > self.config.max_delay)
        ):
            raise ValueError("RTC delays must lie between zero and max_delay")
        prefix = torch.arange(horizon, device=actions.device)[None] < delays[:, None]
        per_token_time = time[:, None].expand(batch_size, horizon).masked_fill(prefix, 0)
        noisy = per_token_time[..., None] * noise + (1 - per_token_time[..., None]) * actions
        predicted = self.velocity(context, noisy, per_token_time)
        valid = batch.get("action_mask", torch.ones_like(actions, dtype=torch.bool)).bool() & ~prefix[..., None]
        squared = (predicted.float() - (noise - actions).float()).square()
        flow = (squared * valid).sum() / valid.sum().clamp_min(1)
        loss = self.config.flow_loss_weight * flow + self.config.fast_loss_weight * ce
        return {"loss": loss, "flow_loss": flow, "fast_loss": ce, "supervised_action_elements": valid.sum()}

    @torch.no_grad()
    def sample_actions(
        self,
        batch,
        *,
        noise=None,
        num_steps=None,
        previous_actions=None,
        delay=None,
        unconditional_batch=None,
        guidance_beta=0.0,
    ):
        """Euler flow integration with optional metadata CFG and clean RTC prefix."""
        steps = self.config.denoising_steps if num_steps is None else num_steps
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 1
            or not math.isfinite(guidance_beta)
            or guidance_beta < 0
        ):
            raise ValueError("Steps must be positive and guidance_beta finite/nonnegative")
        if guidance_beta and unconditional_batch is None:
            raise ValueError("CFG requires a matching context with metadata removed")
        context, _ = self.encode_context(batch)
        uncond = self.encode_context(unconditional_batch)[0] if guidance_beta else None
        shape = (context.valid.shape[0], self.config.action_horizon, self.config.action_dim)
        x = (
            torch.randn(shape, device=context.valid.device, dtype=self.action_in.weight.dtype)
            if noise is None
            else noise.clone()
        )
        if tuple(x.shape) != shape or not torch.isfinite(x).all():
            raise ValueError("Noise must be finite and match the action shape")
        fixed = torch.zeros(shape[:2], device=x.device, dtype=torch.bool)
        if previous_actions is not None:
            if (
                previous_actions.ndim != 3
                or previous_actions.shape[0] != shape[0]
                or previous_actions.shape[2] != shape[2]
                or not torch.isfinite(previous_actions).all()
            ):
                raise ValueError("Previous actions must have shape [batch, prefix, action_dim]")
            count = previous_actions.shape[1] if delay is None else delay
            if not isinstance(count, int) or not 0 <= count <= min(self.config.max_delay, previous_actions.shape[1]):
                raise ValueError("RTC prefix length must lie between zero and max_delay")
            fixed[:, :count] = True
            x[:, :count] = previous_actions[:, :count]
        elif delay is not None:
            raise ValueError("delay requires previous_actions")
        initial = x.clone()
        for step in range(steps):
            time = torch.full(shape[:2], 1 - step / steps, device=x.device).masked_fill(fixed, 0)
            velocity = self.velocity(context, x, time)
            if uncond is not None:
                negative = self.velocity(uncond, x, time)
                velocity = velocity + guidance_beta * (velocity - negative)
            x = x - velocity / steps
            x = torch.where(fixed[..., None], initial, x)
        return x
