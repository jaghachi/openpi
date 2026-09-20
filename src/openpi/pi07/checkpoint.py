"""Self-describing pi0.7 checkpoints; loading never fetches model weights."""

import copy
from dataclasses import asdict
from pathlib import Path

import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

from openpi.pi07.backbone import Gemma3Backbone
from openpi.pi07.model import Pi07
from openpi.pi07.model import Pi07Config


def save_checkpoint(path, model, *, optimizer=None, step=0, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_config": asdict(model.config),
        "backbone_config": model.backbone.hf_model.config.to_dict(),
        "image_size": model.backbone.image_size,
        "temporal_every": model.backbone.temporal_every,
        "state_dict": model.state_dict(),
        # HF omits rotary frequencies/embedding scales from its state_dict.
        # Saving their actual dtype/value preserves mixed-dtype inference too.
        "buffers": {name: value.detach().clone() for name, value in model.named_buffers()},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "metadata": metadata or {},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 2:
        raise ValueError("Unsupported pi0.7 checkpoint format")
    # Meta construction avoids initializing a second full-sized model in memory.
    with torch.device("meta"):
        backbone = Gemma3Backbone.from_config(
            payload["backbone_config"], image_size=payload["image_size"], temporal_every=payload["temporal_every"]
        )
        model = Pi07(backbone, Pi07Config(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    backbone.hf_model.tie_weights()
    # Rotary/nonpersistent buffers are created on meta; rebuild them from config.
    text = backbone.hf_model.model.language_model
    text.embed_tokens.embed_scale = torch.tensor(backbone.width**0.5, device=device)
    text.rotary_emb = Gemma3RotaryEmbedding(config=backbone.hf_model.config.text_config, device=device)
    # The local rotary configuration uses a different theta.
    local_config = copy.deepcopy(backbone.hf_model.config.text_config)
    local_config.rope_theta = local_config.rope_local_base_freq
    local_config.rope_scaling = {"rope_type": "default"}
    text.rotary_emb_local = Gemma3RotaryEmbedding(config=local_config, device=device)
    # SigLIP position IDs are also a nonpersistent buffer.
    embeddings = backbone.hf_model.model.vision_tower.vision_model.embeddings
    embeddings.position_ids = torch.arange(embeddings.num_positions, device=device).expand((1, -1))
    for name, buffer in payload["buffers"].items():
        module_name, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(module_name), attribute, buffer.to(device))
    for rotary in (text.rotary_emb, text.rotary_emb_local):
        rotary.original_inv_freq = rotary.inv_freq
    return model.to(device), payload
