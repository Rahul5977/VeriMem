"""QLoRA SFT for the Qwen2.5-3B verifier -- P2.8 (smoke), P2.9 (plain), P2.10 (retrieval).

Runs on GPU-A (RTX 4070, 8 GB) or GPU-B (RTX 4050, 6 GB) under Linux or WSL2. Never on
the Mac: torch, bitsandbytes and Unsloth are imported lazily inside `train()` so that
`--help`, `--dry-run` and the whole data path still work there.

Defaults come from `core.config.RunConfig` and match the 8 GB VRAM budget:
4-bit base, bf16 compute, gradient checkpointing, batch 1, accumulation 16, seq <= 4096,
a checkpoint every 200 steps, and `--resume` to pick a killed run back up.

    # smoke test first -- 50 examples, 10 steps, proves the environment (P2.8)
    python -m training.sft --data training/fixtures/sample_trajectories.jsonl \\
        --name smoke --smoke

    # the real runs, once trajectories exist (P2.4)
    python -m training.sft --data data/trajectories.jsonl --name lora-sft --mode plain
    python -m training.sft --data data/trajectories.jsonl --name rag-sft  --mode retrieval

Output is a LoRA adapter (tens of MB), not a merged model -- small enough to attach to a
release or push to the Hub for Teesha to pull onto the Mac. See training/README.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import RunConfig
from core.paths import run_dir
from experience.format import read_jsonl
from training.data import build_dataset


@dataclass
class TrainArgs:
    data: Path
    name: str
    mode: str = "plain"
    model: str | None = None
    out: Path | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    learning_rate: float = 2e-4
    epochs: float = 2.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    batch_size: int = 1
    grad_accum: int | None = None
    max_seq_length: int | None = None
    save_steps: int | None = None
    backend: str = "auto"  # auto | unsloth | peft
    load_in_4bit: bool = True
    keep_incorrect: bool = False
    seed: int = 13
    limit: int | None = None
    resume: bool = False
    dry_run: bool = False


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    p = argparse.ArgumentParser(
        prog="python -m training.sft",
        description="QLoRA SFT for the VeriMem verifier (GPU only).",
    )
    p.add_argument("--data", type=Path, required=True, help="trajectories .jsonl")
    p.add_argument("--name", required=True, help="run name; output goes to results/<date>_<name>/")
    p.add_argument("--mode", choices=["plain", "retrieval"], default="plain")
    p.add_argument("--model", default=None, help="base model (default: RunConfig.verifier_model)")
    p.add_argument(
        "--out", type=Path, default=None, help="adapter dir (default: inside the run dir)"
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--save-steps", type=int, default=None)
    p.add_argument("--backend", choices=["auto", "unsloth", "peft"], default="auto")
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument(
        "--keep-incorrect", action="store_true", help="train on failed verifications too"
    )
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit", type=int, default=None, help="cap the number of trajectories")
    p.add_argument("--resume", action="store_true", help="resume from the latest checkpoint")
    p.add_argument("--dry-run", action="store_true", help="build the dataset, print stats, stop")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="P2.8: 50 examples, 10 steps -- proves the environment without a real run",
    )
    ns = p.parse_args(argv)

    if ns.smoke:
        ns.limit = ns.limit or 50
        ns.max_steps = 10 if ns.max_steps < 0 else ns.max_steps
        ns.save_steps = ns.save_steps or 5
    del ns.smoke
    return TrainArgs(**vars(ns))


def prepare(args: TrainArgs) -> tuple[list[dict[str, Any]], dict[str, Any], RunConfig]:
    """Load trajectories and build the SFT dataset. No GPU, no torch -- runs anywhere."""
    cfg = RunConfig(
        verifier="finetuned",
        experience_retrieval=(args.mode == "retrieval"),
        seed=args.seed,
        limit=args.limit,
    )
    if args.model:
        cfg.verifier_model = args.model
    if args.max_seq_length:
        cfg.max_seq_length = args.max_seq_length
    if args.grad_accum:
        cfg.grad_accum_steps = args.grad_accum
    if args.save_steps:
        cfg.checkpoint_every_steps = args.save_steps

    trajectories = read_jsonl(args.data)
    random.Random(args.seed).shuffle(trajectories)
    if args.limit:
        trajectories = trajectories[: args.limit]

    examples, stats = build_dataset(
        trajectories,
        mode=args.mode,
        cfg=cfg,
        keep_incorrect=args.keep_incorrect,
    )
    if not examples:
        raise SystemExit(
            f"No trainable examples in {args.data}. All {stats.n_skipped} trajectories were "
            "incorrect verifications -- pass --keep-incorrect to train on them anyway."
        )
    if stats.max_tokens > cfg.max_seq_length:
        print(
            f"warning: longest example is ~{stats.max_tokens} tokens, over max_seq_length "
            f"{cfg.max_seq_length}; it will be truncated by the trainer",
            file=sys.stderr,
        )
    return [e.to_dict() for e in examples], stats.as_dict(), cfg


def train(args: TrainArgs) -> Path:
    records, stats, cfg = prepare(args)
    out_dir = run_dir(args.name)
    adapter_dir = args.out or (out_dir / "adapter")
    cfg.save(out_dir)
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_dataset_preview(records, out_dir)

    print(f"[verimem] {stats['n_examples']} examples, mean ~{stats['mean_tokens']} tokens")
    print(f"[verimem] mode={args.mode} base={cfg.verifier_model} -> {adapter_dir}")
    if args.dry_run:
        print("[verimem] --dry-run: dataset built, stopping before training.")
        return out_dir

    backend = _resolve_backend(args.backend)
    print(f"[verimem] backend={backend}")

    from datasets import Dataset  # noqa: PLC0415  -- GPU-only import

    dataset = Dataset.from_list(records)

    if backend == "unsloth":
        model, tokenizer = _load_unsloth(args, cfg)
    else:
        model, tokenizer = _load_peft(args, cfg)

    import torch  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        max_length=cfg.max_seq_length,
        logging_steps=10,
        save_steps=cfg.checkpoint_every_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="adamw_8bit" if args.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=args.seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer
    )

    result = trainer.train(resume_from_checkpoint=args.resume or None)

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = {
        **result.metrics,
        "dataset": stats,
        "mode": args.mode,
        "base_model": cfg.verifier_model,
        "backend": backend,
        "lora_r": args.lora_r,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_notes(out_dir, args, cfg, stats, adapter_dir)
    print(f"[verimem] adapter written to {adapter_dir}")
    return out_dir


def _resolve_backend(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import unsloth  # noqa: F401, PLC0415

        return "unsloth"
    except ImportError:
        return "peft"


def _load_unsloth(args: TrainArgs, cfg: RunConfig):
    # Unsloth patches transformers on import, so it must come first.
    from unsloth import FastLanguageModel  # noqa: PLC0415

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.verifier_model,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=None,  # Unsloth picks bf16 on Ada, fp16 on a T4
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    return model, tokenizer


def _load_peft(args: TrainArgs, cfg: RunConfig):
    import torch  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    quant_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415  -- CUDA only

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.verifier_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.verifier_model,
        quantization_config=quant_config,
        dtype="auto",
        device_map={"": 0},
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    return model, tokenizer


def write_dataset_preview(records: list[dict[str, Any]], out_dir: Path, n: int = 3) -> None:
    """First few examples, so a reviewer can read what the model actually saw."""
    preview = out_dir / "dataset_preview.jsonl"
    with preview.open("w", encoding="utf-8") as fh:
        for rec in records[:n]:
            fh.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")


def _write_notes(
    out_dir: Path, args: TrainArgs, cfg: RunConfig, stats: dict[str, Any], adapter: Path
) -> None:
    (out_dir / "notes.md").write_text(
        f"""# {out_dir.name}

Mode **{args.mode}** · base `{cfg.verifier_model}` · LoRA r={args.lora_r}, alpha={args.lora_alpha}
seq<={cfg.max_seq_length} · batch {args.batch_size} x accum {cfg.grad_accum_steps} · seed {args.seed}

- Trajectories: `{args.data}`
- Examples: {stats["n_examples"]} ({stats["n_skipped"]} skipped as incorrect verifications)
- With retrieved context: {stats["n_with_retrieval"]} · trimmed to fit: {stats["n_trimmed"]}
- Mean ~{stats["mean_tokens"]} tokens, max ~{stats["max_tokens"]}
- Adapter: `{adapter}`

Fill in after evaluating: held-out-topic macro-F1, in-distribution macro-F1, and how this
compares with the previous run.
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
