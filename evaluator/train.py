"""Fine-tune the CRAG relevance evaluator (P1.3).

Flan-T5 scores how relevant a retrieved passage is to a claim. CRAG then compares that
score against two thresholds to pick an action -- Correct, Incorrect, or Ambiguous -- so
the evaluator must emit a *graded* score, not just a label. T5 is a text-to-text model,
so the score comes from the probability it assigns to the token "yes" against "no" at
the first decoded position, mapped to [-1, 1]:

    score = 2 * P(yes) / (P(yes) + P(no)) - 1

That is cheap (one forward pass, one position) and monotone in the model's confidence,
which is all the thresholds need.

Runs on a GPU: Kaggle T4, or GPU-B. Flan-T5-base is 250M parameters and trains on ~7k
pairs in minutes. T5-large only with bf16 + Adafactor + batch 1-2, per the hardware
rules, and a T4 has no bf16 -- `core.device` handles the fp16 fallback.

    python -m evaluator.train --data data/evaluator --name t5-evaluator
    python -m evaluator.train --data data/evaluator --name smoke --smoke
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.paths import DATA_DIR, run_dir

# The two tokens whose relative probability becomes the score.
YES, NO = "yes", "no"
PROMPT = "Is this passage relevant to judging the claim?\nClaim: {claim}\nPassage: {passage}"


@dataclass
class TrainArgs:
    data: Path
    name: str
    model: str = "google/flan-t5-base"
    epochs: float = 3.0
    batch_size: int = 8
    grad_accum: int = 2
    learning_rate: float = 3e-4
    max_input_tokens: int = 512
    seed: int = 13
    limit: int | None = None
    dry_run: bool = False


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    p = argparse.ArgumentParser(prog="python -m evaluator.train")
    p.add_argument("--data", type=Path, default=DATA_DIR / "evaluator")
    p.add_argument("--name", default="t5-evaluator")
    p.add_argument("--model", default="google/flan-t5-base")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="load and report, train nothing")
    p.add_argument("--smoke", action="store_true", help="200 pairs, 1 epoch: proves the setup")
    ns = p.parse_args(argv)
    if ns.smoke:
        ns.limit, ns.epochs = ns.limit or 200, 1.0
    del ns.smoke
    return TrainArgs(**vars(ns))


def load_pairs(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def format_example(row: dict[str, Any]) -> dict[str, str]:
    return {
        "input": PROMPT.format(claim=row["claim"], passage=row["passage"]),
        "target": YES if row["label"] == 1 else NO,
    }


def class_balance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pos = sum(1 for r in rows if r["label"] == 1)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {"n": len(rows), "positive": pos, "negative": len(rows) - pos, "by_kind": kinds}


def describe(args: TrainArgs) -> dict[str, Any]:
    """Everything knowable without a GPU, so --dry-run is useful on the Mac."""
    train = load_pairs(args.data / "relevance_train.jsonl", args.limit)
    dev = load_pairs(args.data / "relevance_dev.jsonl")
    if not train:
        raise SystemExit(f"No training pairs in {args.data}. Run evaluator.build_data first.")
    overlap = {r["claim_id"] for r in train} & {r["claim_id"] for r in dev}
    if overlap:
        raise SystemExit(
            f"{len(overlap)} claim ids appear in both train and dev. The split must be "
            "by claim (rule 5) -- rebuild with evaluator.build_data."
        )
    return {"train": class_balance(train), "dev": class_balance(dev)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = describe(args)
    print(
        f"[verimem] train {stats['train']['n']} pairs "
        f"({stats['train']['positive']} pos / {stats['train']['negative']} neg)"
    )
    print(f"[verimem] dev   {stats['dev']['n']} pairs   by kind: {stats['dev']['by_kind']}")
    print(f"[verimem] base model {args.model}")

    example = format_example(load_pairs(args.data / "relevance_train.jsonl", 1)[0])
    print(f"\n[verimem] example input:\n{example['input'][:300]}...\n -> {example['target']}\n")

    if args.dry_run:
        print("[verimem] --dry-run: nothing trained.")
        return 0

    out = run_dir(args.name)
    _train(args, out, stats)
    return 0


def _train(args: TrainArgs, out: Path, stats: dict[str, Any]) -> None:
    import numpy as np  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from core.device import get_device, supports_bf16  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def encode(rows: list[dict[str, Any]]) -> Dataset:
        formatted = [format_example(r) for r in rows]
        ds = Dataset.from_list(formatted)

        def tok(batch):
            enc = tokenizer(batch["input"], max_length=args.max_input_tokens, truncation=True)
            enc["labels"] = tokenizer(batch["target"], max_length=4, truncation=True)["input_ids"]
            return enc

        return ds.map(tok, batched=True, remove_columns=["input", "target"])

    train_ds = encode(load_pairs(args.data / "relevance_train.jsonl", args.limit))
    # Dev is scored by hand in _score_dev, since the metric that matters is a ranking
    # over the relevance score rather than token-level seq2seq loss.
    dev_rows = load_pairs(args.data / "relevance_dev.jsonl")

    cuda = get_device() == "cuda"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_steps=500,
        save_total_limit=1,
        seed=args.seed,
        bf16=cuda and supports_bf16(),
        fp16=cuda and not supports_bf16(),  # T4 has no bf16
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    result = trainer.train()

    adapter_dir = out / "model"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    scores, labels = _score_dev(model, tokenizer, dev_rows, args)
    metrics = {
        **result.metrics,
        "dev": _relevance_metrics(np.array(scores), np.array(labels)),
        "data": stats,
        "base_model": args.model,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n[verimem] dev: {metrics['dev']}")
    print(f"[verimem] wrote {out}")
    _write_notes(out, args, metrics)


def _score_dev(model, tokenizer, rows, args) -> tuple[list[float], list[int]]:
    """Relevance score per dev pair, from P(yes) against P(no) at the first position."""
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    yes_id = tokenizer(YES, add_special_tokens=False)["input_ids"][0]
    no_id = tokenizer(NO, add_special_tokens=False)["input_ids"][0]
    scores, labels = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), 16):
            batch = rows[i : i + 16]
            enc = tokenizer(
                [format_example(r)["input"] for r in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(device)
            start = torch.full((len(batch), 1), model.config.decoder_start_token_id).to(device)
            logits = model(**enc, decoder_input_ids=start).logits[:, 0, :]
            probs = torch.softmax(logits[:, [yes_id, no_id]].float(), dim=-1)[:, 0]
            scores += (probs * 2 - 1).cpu().tolist()  # map to [-1, 1]
            labels += [r["label"] for r in batch]
    return scores, labels


def _relevance_metrics(scores, labels) -> dict[str, float]:
    """Accuracy at the natural threshold, plus AUC, which the thresholds depend on."""
    predicted = (scores > 0).astype(int)
    accuracy = float((predicted == labels).mean())

    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        auc = float("nan")
    else:
        # AUC as the probability a random positive outranks a random negative.
        wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        auc = float(wins / (len(pos) * len(neg)))
    return {
        "accuracy": round(accuracy, 4),
        "auc": round(auc, 4),
        "mean_score_positive": round(float(pos.mean()) if len(pos) else float("nan"), 4),
        "mean_score_negative": round(float(neg.mean()) if len(neg) else float("nan"), 4),
        "n": int(len(labels)),
    }


def _write_notes(out: Path, args: TrainArgs, metrics: dict[str, Any]) -> None:
    d = metrics["dev"]
    (out / "notes.md").write_text(
        f"""# {out.name}

Flan-T5 relevance evaluator. Base `{args.model}`, {args.epochs} epochs,
batch {args.batch_size} x accum {args.grad_accum}, lr {args.learning_rate}, seed {args.seed}.

| metric | value |
|---|---|
| dev accuracy | {d["accuracy"]} |
| dev AUC | {d["auc"]} |
| mean score, relevant | {d["mean_score_positive"]} |
| mean score, irrelevant | {d["mean_score_negative"]} |
| dev pairs | {d["n"]} |

AUC matters more than accuracy here: CRAG picks its action from two thresholds on the
score, so what counts is whether relevant passages are ranked above irrelevant ones,
not whether they land on the right side of zero.

Next: P1.4 compares this against a cross-encoder baseline, and P1.7 tunes the two
thresholds on dev and logs the resulting action shares.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
