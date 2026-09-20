"""Small, reproducible pi0.7 trainer with an offline end-to-end smoke mode.

Run ``python -m openpi.pi07.train --help`` with src on PYTHONPATH. Full training
requires locally supplied Gemma 3 weights, a tokenizer, normalized episodes and
FAST labels/processor. No large checkpoint is downloaded implicitly.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from transformers import AutoTokenizer

from openpi.pi07.backbone import Gemma3Backbone
from openpi.pi07.checkpoint import load_checkpoint
from openpi.pi07.checkpoint import save_checkpoint
from openpi.pi07.data import Episode
from openpi.pi07.data import EpisodeDataset
from openpi.pi07.data import SamplingConfig
from openpi.pi07.data import collate_examples
from openpi.pi07.model import Pi07
from openpi.pi07.model import Pi07Config
from openpi.pi07.tokenization import FASTCodec


def fingerprint_directory(path):
    """Hash supplied local processor artifacts, including weights and custom code."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError("Processor assets must be a local directory")
    files = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or any(
            part in {".git", ".cache", "__pycache__"} for part in item.relative_to(root).parts
        ):
            continue
        with item.open("rb") as stream:
            files.append((item.relative_to(root).as_posix(), hashlib.file_digest(stream, "sha256").hexdigest()))
    return hashlib.sha256(json.dumps(files).encode()).hexdigest()


def smoke_dataset(seed=0):
    """Synthetic tensors and synthetic token IDs, never a FAST efficacy test."""
    rng = np.random.default_rng(seed)
    length = 80
    episode = Episode(
        images=rng.integers(0, 256, (length, 2, 16, 16, 3), dtype=np.uint8),
        states=rng.uniform(-1, 1, (length, 3)).astype(np.float32),
        actions=rng.uniform(-1, 1, (length, 3)).astype(np.float32),
        task="move the test object",
        subtask="reach the test object",
        control_mode="joint",
        quality=5,
        mistake=False,
        fps=2,
        fast_token_ids=rng.integers(16, 32, (length, 4), dtype=np.int64),
    )
    return EpisodeDataset([episode], config=SamplingConfig(image_size=16), seed=seed)


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Steps, batch size and learning rate must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype_name = getattr(args, "dtype", "float32")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise ValueError("Output directory is nonempty; use a new directory or --resume")
    output.mkdir(parents=True, exist_ok=True)
    action_tokenizer = None
    text_tokenizer = None
    payload = None
    if args.smoke:
        dataset = smoke_dataset(args.seed)

        def tokenizer(text):
            return [1, 3, 4, 5]  # Explicitly synthetic smoke-only text labels.

        backbone = Gemma3Backbone.tiny()
        # HF config initialization zeros this normally pretrained projection.
        torch.nn.init.normal_(backbone.hf_model.model.multi_modal_projector.mm_input_projection_weight, std=0.02)
        model = Pi07(backbone, Pi07Config(action_dim=3, state_dim=3, expert_width=32, expert_mlp_dim=64, fast_bos_id=1))
        pad_token_id = 0
    else:
        if not args.data or not args.backbone or not args.data_is_normalized:
            raise ValueError("Training requires --data, local --backbone and --data-is-normalized")
        text_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.backbone, local_files_only=True)

        def tokenizer(text):
            return text_tokenizer.encode(text, add_special_tokens=True)

        pad_token_id = text_tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Text tokenizer needs a padding token")
        backbone = (
            None
            if args.resume
            else Gemma3Backbone.from_pretrained(args.backbone, local_files_only=True, torch_dtype=dtype)
        )
        if args.fast_processor:
            if args.fast_codebook_size is None:
                raise ValueError("--fast-codebook-size is required with --fast-processor")
            processor = AutoProcessor.from_pretrained(
                args.fast_processor, local_files_only=True, trust_remote_code=args.allow_processor_code
            )
            action_tokenizer = FASTCodec(processor, text_tokenizer, codebook_size=args.fast_codebook_size)
        if backbone is not None:
            if len(text_tokenizer) != backbone.vocab_size:
                backbone.resize_token_embeddings(len(text_tokenizer))
            model = Pi07(
                backbone,
                Pi07Config(
                    action_dim=args.action_dim,
                    state_dim=args.state_dim,
                    expert_width=args.expert_width,
                    expert_mlp_dim=args.expert_mlp_dim,
                    fast_bos_id=text_tokenizer.bos_token_id,
                ),
            )
        dataset = EpisodeDataset.from_npz(
            args.data, config=SamplingConfig(generated_goal_probability=args.generated_goal_probability), seed=args.seed
        )
    fingerprints = []
    for path in args.data or []:
        with Path(path).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        fingerprints.append({"path": str(Path(path).resolve()), "sha256": digest})
    tokenizer_fingerprint = "synthetic-smoke-tokenizer-v1"
    if text_tokenizer is not None:
        backend = getattr(text_tokenizer, "backend_tokenizer", None)
        sentencepiece = getattr(text_tokenizer, "sp_model", None)
        if backend is not None:
            tokenization_rules = backend.to_str()
        elif sentencepiece is not None:
            tokenization_rules = hashlib.sha256(sentencepiece.serialized_model_proto()).hexdigest()
        else:
            raise ValueError("Cannot fingerprint this tokenizer's complete encoding rules")
        serialized = json.dumps(
            {
                "vocab": text_tokenizer.get_vocab(),
                "special_tokens": text_tokenizer.special_tokens_map,
                "tokenization_rules": tokenization_rules,
                "init_kwargs": text_tokenizer.init_kwargs,
            },
            sort_keys=True,
            default=str,
        )
        tokenizer_fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
    metadata = {
        "smoke": args.smoke,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "data_fingerprints": fingerprints,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "fast_processor_fingerprint": fingerprint_directory(args.fast_processor) if args.fast_processor else None,
        "fast_codebook_size": args.fast_codebook_size,
        "allow_processor_code": args.allow_processor_code,
        "data_is_normalized": args.data_is_normalized,
        "dtype": dtype_name,
        "backbone_source": args.backbone,
        "sampling": asdict(dataset.config),
        "optimizer_recipe": "AdamW; implementation choice, not disclosed in paper",
    }
    if args.resume:
        model, payload = load_checkpoint(args.resume, device=device)
        for key in (
            "smoke",
            "seed",
            "batch_size",
            "data_fingerprints",
            "tokenizer_fingerprint",
            "fast_processor_fingerprint",
            "fast_codebook_size",
            "allow_processor_code",
            "sampling",
            "data_is_normalized",
            "dtype",
        ):
            if payload["metadata"].get(key) != metadata[key]:
                raise ValueError(f"Resume requires unchanged {key}")
        if text_tokenizer is not None and len(text_tokenizer) != model.backbone.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match resumed model")
    if text_tokenizer is not None:
        text_tokenizer.save_pretrained(output / "tokenizer")
    model.to(device=device, dtype=dtype).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = 0
    if payload is not None:
        if payload["optimizer"] is None:
            raise ValueError("Resume requires optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
        start_step = payload["step"]
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    batches_per_epoch = (len(dataset) + args.batch_size - 1) // args.batch_size
    last = None
    for step in range(start_step, start_step + args.steps):
        epoch, offset = divmod(step, batches_per_epoch)
        dataset.set_epoch(epoch)
        order = np.random.default_rng(np.random.SeedSequence([args.seed, epoch])).permutation(len(dataset))
        selected = order[offset * args.batch_size : (offset + 1) * args.batch_size]
        batch = collate_examples(
            [dataset[int(index)] for index in selected],
            tokenizer,
            action_tokenizer=action_tokenizer,
            pad_token_id=pad_token_id,
            target_state_dim=model.config.state_dim,
            target_action_dim=model.config.action_dim,
            as_torch=True,
        )
        batch = {name: value.to(device) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        losses = model(batch)
        if not torch.isfinite(losses["loss"]):
            raise FloatingPointError("Nonfinite training loss")
        losses["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        last = {key: float(value.detach()) for key, value in losses.items()}
        last.update(step=step + 1, gradient_norm=float(norm))
        print(json.dumps(last), flush=True)
    checkpoint = output / "checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer=optimizer, step=start_step + args.steps, metadata=metadata)
    if args.smoke:
        model.eval()
        noise = torch.randn_like(batch["actions"])
        expected = model.sample_actions(batch, noise=noise)
        restored, _ = load_checkpoint(checkpoint, device=device)
        restored.eval()
        actual = restored.sample_actions(batch, noise=noise)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if not torch.isfinite(actual).all():
            raise FloatingPointError("Nonfinite sampled actions")
        last.update(
            sample_shape=list(actual.shape),
            checkpoint_roundtrip="exact",
            scope="synthetic code validation; no pretrained weights, real FAST codec, or robot evaluation",
        )
    (output / "run.json").write_text(json.dumps({"metadata": metadata, "result": last}, indent=2), encoding="utf-8")
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny offline model and synthetic data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10, help="additional optimizer steps (including on resume)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--resume")
    parser.add_argument("--data", nargs="+")
    parser.add_argument("--backbone", help="local Gemma3ForConditionalGeneration checkpoint")
    parser.add_argument("--tokenizer", help="local extended Gemma text tokenizer; defaults to backbone")
    parser.add_argument(
        "--data-is-normalized", action="store_true", help="confirm shared normalized state/action layout"
    )
    parser.add_argument("--fast-processor", help="local pretrained FAST processor (or provide cached NPZ FAST IDs)")
    parser.add_argument("--fast-codebook-size", type=int)
    parser.add_argument(
        "--allow-processor-code", action="store_true", help="allow custom code from the supplied local processor"
    )
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--expert-width", type=int, default=1024)
    parser.add_argument("--expert-mlp-dim", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--generated-goal-probability",
        type=float,
        default=0.0,
        help="explicit undisclosed mixture ratio for supplied generated goal images",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
