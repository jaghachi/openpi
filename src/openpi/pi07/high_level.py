"""Native Gemma 3 language policy for coaching and semantic subtask prediction.

Training needs observations, robot states, task/history prompt tokens, and the
desired next subtask as text tokens. No action or FAST targets are required.
Prompt wording, coaching dataset composition, and decoding settings are explicit
reimplementation choices; pi0.7 does not disclose a complete high-level recipe.
"""

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional

from openpi.pi07.backbone import Gemma3Backbone
from openpi.pi07.model import Pi07


@dataclass(frozen=True)
class TextGeneration:
    """Generated text IDs, including EOS but excluding the initial BOS.

    ``token_mask`` excludes padding after EOS. ``finished`` distinguishes EOS
    termination from a generation truncated by ``max_new_tokens``.
    """

    token_ids: torch.Tensor
    token_mask: torch.Tensor
    finished: torch.Tensor


class HighLevelModel(nn.Module):
    """Observation-conditioned text model with no action-expert parameters.

    Input batches use Pi07's image/state/text tensor schema. Their prompt text
    should contain the overall task and prior semantic instruction history.
    Goal-image keys are deliberately ignored: this policy produces the subtask
    used by the downstream goal generator.

    ``from_vla`` can either share the VLA's backbone/state projection, meaning
    language training also updates that VLA, or copy only those two modules for
    an independently trained high-level policy. A fresh constructor initializes
    its state projection randomly and requires training before meaningful use.
    """

    def __init__(
        self,
        backbone: Gemma3Backbone,
        state_dim: int,
        *,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ):
        super().__init__()
        if isinstance(state_dim, bool) or not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        self.backbone = backbone
        self.state_projection = nn.Linear(state_dim, backbone.width)
        # Pi07._embed_context only requires state_dim, backbone and projection;
        # reuse that implementation without constructing/registering an expert.
        self.config = SimpleNamespace(state_dim=state_dim)
        text_config = backbone.hf_model.config.text_config
        self.bos_token_id = text_config.bos_token_id if bos_token_id is None else bos_token_id
        self.eos_token_id = text_config.eos_token_id if eos_token_id is None else eos_token_id
        self.pad_token_id = text_config.pad_token_id if pad_token_id is None else pad_token_id
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < backbone.vocab_size:
                raise ValueError(f"{name} must be an integer in the backbone vocabulary")

    @classmethod
    def from_vla(cls, model: Pi07, *, share_weights: bool = True, **kwargs) -> "HighLevelModel":
        """Share weights explicitly, or copy the backbone/projection only.

        With ``share_weights=True`` an optimizer step on this module changes
        ``model`` and should be saved through its VLA checkpoint as well. With
        ``False`` it is a separate model and requires its own checkpoint.
        """
        backbone = model.backbone if share_weights else copy.deepcopy(model.backbone)
        result = cls(backbone, model.config.state_dim, **kwargs)
        result.state_projection = model.state_projection if share_weights else copy.deepcopy(model.state_projection)
        return result

    def _embed_context(self, batch):
        context_batch = {key: value for key, value in batch.items() if key not in ("goal_images", "goal_mask")}
        return Pi07._embed_context(self, context_batch)  # noqa: SLF001

    def _decode_tokens(self, prefix, valid, allowed, token_ids, token_mask):
        """Append a causal text-output branch to the fixed multimodal prefix."""
        batch_size, prefix_length = valid.shape
        target_length = token_ids.shape[1]
        safe_ids = token_ids.masked_fill(~token_mask, self.pad_token_id)
        embeddings = torch.cat((prefix, self.backbone.embed_text(safe_ids)), dim=1)
        total_valid = torch.cat((valid, token_mask), dim=1)
        total_length = prefix_length + target_length
        full_mask = torch.zeros(batch_size, total_length, total_length, device=valid.device, dtype=torch.bool)
        full_mask[:, :prefix_length, :prefix_length] = allowed
        full_mask[:, prefix_length:, :prefix_length] = valid[:, None, :]
        full_mask[:, prefix_length:, prefix_length:] = torch.ones(
            target_length, target_length, dtype=torch.bool, device=valid.device
        ).tril()
        full_mask &= total_valid[:, :, None] & total_valid[:, None, :]
        result = self.backbone(embeddings, full_mask, total_valid, return_logits=False)
        return self.backbone.decode(result.last_hidden_state[:, prefix_length:])

    def forward(self, batch, target_ids: torch.Tensor, target_mask: torch.Tensor):
        """Teacher-forced next-subtask CE; targets start after BOS and include EOS.

        Masked target values may be any integer, including -100. Each row must
        have a nonempty contiguous valid prefix, followed by optional padding.
        """
        if target_ids.ndim != 2 or target_ids.shape[1] == 0 or target_ids.dtype != torch.long:
            raise ValueError("target_ids must be nonempty long [batch, tokens]")
        if target_mask.shape != target_ids.shape or target_mask.dtype != torch.bool:
            raise ValueError("target_mask must be boolean with the target_ids shape")
        if torch.any(~target_mask[:, 0]) or torch.any(target_mask[:, 1:] & ~target_mask[:, :-1]):
            raise ValueError("each target row must contain a nonempty contiguous valid prefix")
        if torch.any(((target_ids < 0) | (target_ids >= self.backbone.vocab_size)) & target_mask):
            raise ValueError("valid target IDs must lie inside the backbone vocabulary")
        prefix, valid, allowed = self._embed_context(batch)
        if target_ids.shape[0] != prefix.shape[0]:
            raise ValueError("target batch size must match observation batch size")
        safe_targets = target_ids.masked_fill(~target_mask, self.pad_token_id)
        teacher = torch.cat((torch.full_like(safe_targets[:, :1], self.bos_token_id), safe_targets[:, :-1]), dim=1)
        logits = self._decode_tokens(prefix, valid, allowed, teacher, target_mask)
        labels = target_ids.masked_fill(~target_mask, -100)
        summed = functional.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum"
        )
        loss = summed / target_mask.sum()
        return {"loss": loss, "text_loss": loss, "supervised_tokens": target_mask.sum(), "logits": logits}

    @torch.no_grad()
    def generate(self, batch, *, max_new_tokens: int = 64) -> TextGeneration:
        """Greedy, EOS-aware batched decoding using the native Gemma LM head.

        The multimodal input is embedded once; decoder layers are recomputed at
        each token for a transparent reference implementation. This does not
        claim the paper's optimized high-level inference latency. Call ``eval``
        first to disable training dropout.
        """
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        prefix, valid, allowed = self._embed_context(batch)
        batch_size = prefix.shape[0]
        device = prefix.device
        inputs = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        input_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated, generated_masks = [], []
        for _ in range(max_new_tokens):
            logits = self._decode_tokens(prefix, valid, allowed, inputs, input_mask)
            active = ~finished
            next_token = logits[:, -1].argmax(-1).masked_fill(~active, self.pad_token_id)
            generated.append(next_token)
            generated_masks.append(active)
            finished = finished | (active & next_token.eq(self.eos_token_id))
            if finished.all():
                break
            inputs = torch.cat((inputs, next_token[:, None]), dim=1)
            input_mask = torch.cat((input_mask, (~finished)[:, None]), dim=1)
        return TextGeneration(torch.stack(generated, dim=1), torch.stack(generated_masks, dim=1), finished)
