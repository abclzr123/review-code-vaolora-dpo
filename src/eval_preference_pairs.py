import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset, load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig

from train_dpo import (
    VAOLoRADPOTrainer,
    build_prompt_text,
    compute_value_alignment_loss,
    extract_assistant_text,
    get_torch_dtype,
    load_yaml,
    maybe_limit_dataset,
    safe_normalize,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--field-prompt", default="prompt")
    parser.add_argument("--field-chosen", default="chosen")
    parser.add_argument("--field-rejected", default="rejected")
    parser.add_argument("--dataset-tag", default="uf_test")
    parser.add_argument("--top-k", type=int, default=64)
    return parser.parse_args()


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(f"Unsupported json dataset format: {path}")


def load_pair_dataset(args, cfg: Dict[str, Any]) -> Dataset:
    data_cfg = cfg["data"]
    split = args.split or data_cfg.get("eval_split", "test")
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
        if dataset_path.is_file():
            return maybe_limit_dataset(Dataset.from_list(load_json_records(dataset_path)), args.limit)
        loaded = load_from_disk(args.dataset_path)
    elif args.dataset_repo:
        loaded = load_dataset(args.dataset_repo)
    else:
        dataset_path = data_cfg.get("dataset_path")
        dataset_repo = data_cfg.get("dataset_repo")
        if dataset_path and Path(dataset_path).is_file():
            return maybe_limit_dataset(Dataset.from_list(load_json_records(Path(dataset_path))), args.limit)
        if dataset_path and Path(dataset_path).exists():
            loaded = load_from_disk(dataset_path)
        elif dataset_repo:
            loaded = load_dataset(dataset_repo)
        else:
            raise ValueError("Need a dataset path or dataset repo")

    if isinstance(loaded, Dataset):
        dataset = loaded
    else:
        dataset = loaded[split]
    return maybe_limit_dataset(dataset, args.limit)


def stringify_prompt(prompt_value: Any) -> str:
    if prompt_value is None:
        return ""
    if isinstance(prompt_value, str):
        return prompt_value
    if isinstance(prompt_value, list):
        parts = []
        for item in prompt_value:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(str(seg.get("text", "")) if isinstance(seg, dict) else str(seg) for seg in content)
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(prompt_value)


def prepare_example(row: Dict[str, Any], tokenizer, args) -> Dict[str, Any]:
    chosen_value = row.get(args.field_chosen)
    rejected_value = row.get(args.field_rejected)
    if chosen_value is None or rejected_value is None:
        raise KeyError(
            f"dataset row must contain `{args.field_chosen}` and `{args.field_rejected}`; got {sorted(row.keys())}"
        )

    prompt_value = row.get(args.field_prompt)
    if prompt_value is None and isinstance(chosen_value, list):
        prompt_value = chosen_value[:-1]

    prompt_text = build_prompt_text(tokenizer, prompt_value) if prompt_value else ""
    return {
        "prompt": prompt_text,
        "chosen": extract_assistant_text(chosen_value),
        "rejected": extract_assistant_text(rejected_value),
        "raw_prompt": stringify_prompt(prompt_value),
    }


def build_eval_dataset(dataset: Dataset, tokenizer, args) -> Dataset:
    rows = [prepare_example(dataset[idx], tokenizer, args) for idx in range(len(dataset))]
    return Dataset.from_list(rows)


def build_trainer(cfg: Dict[str, Any], eval_dataset: Dataset, batch_size: int, adapter_path: str):
    model_cfg = cfg["model"]
    peft_cfg = cfg["peft"]
    trainer_cfg = cfg["trainer"]
    tokenizer_name = cfg["tokenizer"]["tokenizer_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_torch_dtype(model_cfg.get("torch_dtype", "bf16"))
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_cfg["model_name_or_path"], **model_kwargs)
    except ImportError as exc:
        attn_impl = model_kwargs.get("attn_implementation")
        if attn_impl != "flash_attention_2":
            raise
        model_kwargs.pop("attn_implementation", None)
        print(
            "[warn] flash_attention_2 unavailable in eval env; retrying with default attention implementation",
            flush=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(model_cfg["model_name_or_path"], **model_kwargs)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    args_cfg = DPOConfig(
        output_dir=str(Path(adapter_path).parent / "tmp_eval_output"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1.0e-5,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        num_train_epochs=1,
        max_steps=1,
        logging_steps=1,
        save_steps=1000000,
        eval_steps=1000000,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        bf16=bool(trainer_cfg.get("bf16", False)),
        fp16=bool(trainer_cfg.get("fp16", False)),
        gradient_checkpointing=False,
        max_prompt_length=int(trainer_cfg["max_prompt_length"]),
        max_completion_length=int(trainer_cfg["max_completion_length"]),
        remove_unused_columns=False,
        report_to=[],
        beta=float(trainer_cfg.get("beta", 0.1)),
        loss_type=trainer_cfg.get("loss_type", "sigmoid"),
        seed=int(cfg["run"]["seed"]),
    )

    trainer = VAOLoRADPOTrainer(
        model=model,
        args=args_cfg,
        method_cfg=cfg.get("method", {}),
        processing_class=tokenizer,
        train_dataset=eval_dataset.select([]),
        eval_dataset=eval_dataset,
    )
    return trainer, tokenizer


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["tokenizer_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    raw_dataset = load_pair_dataset(args, cfg)
    eval_dataset = build_eval_dataset(raw_dataset, tokenizer, args)
    trainer, _ = build_trainer(cfg, eval_dataset, args.batch_size, args.adapter_path)

    dataloader = trainer.get_eval_dataloader()
    sample_rows: List[Dict[str, Any]] = []
    all_margins: List[float] = []
    all_correct: List[float] = []
    all_losses: List[float] = []
    all_value_cosine: List[float] = []

    offset = 0
    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = trainer._prepare_inputs(batch)
            for key, value in list(batch.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                if key.endswith("input_ids") or key.endswith("labels"):
                    batch[key] = value.long()
                elif key.endswith("attention_mask"):
                    batch[key] = value.long()
            batch_size = batch["prompt_input_ids"].shape[0]
            model_output = trainer.concatenated_forward(trainer.model, batch)
            chosen_rewards = model_output["chosen_logps"].detach().cpu()
            rejected_rewards = model_output["rejected_logps"].detach().cpu()
            margins = (chosen_rewards - rejected_rewards).detach().cpu()
            losses = torch.nn.functional.softplus(-margins)
            correct = (margins > 0).float()

            if trainer.enable_value_alignment and trainer.lambda_val > 0:
                pooled_hidden = trainer._compute_pooled_hidden_states(trainer.model, batch)
                value_direction = trainer._load_value_direction(
                    pooled_hidden["chosen_pooled"].device,
                    pooled_hidden["chosen_pooled"].dtype,
                )
                value_stats = compute_value_alignment_loss(
                    pooled_hidden["chosen_pooled"],
                    pooled_hidden["rejected_pooled"],
                    safe_normalize(value_direction, dim=0),
                    trainer.value_loss_type,
                    trainer.value_margin,
                )
                value_cosine = value_stats["value_cosine"].detach().cpu()
            else:
                value_cosine = None

            for idx in range(batch_size):
                row = eval_dataset[offset + idx]
                sample = {
                    "dataset_tag": args.dataset_tag,
                    "sample_index": offset + idx,
                    "prompt": row["raw_prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                    "chosen_reward": float(chosen_rewards[idx].item()),
                    "rejected_reward": float(rejected_rewards[idx].item()),
                    "reward_margin": float(margins[idx].item()),
                    "reward_correct": bool(correct[idx].item() > 0.5),
                    "dpo_loss": float(losses[idx].item()),
                }
                if value_cosine is not None:
                    sample["value_cosine"] = float(value_cosine[idx].item())
                    all_value_cosine.append(float(value_cosine[idx].item()))
                sample_rows.append(sample)
                all_margins.append(sample["reward_margin"])
                all_correct.append(1.0 if sample["reward_correct"] else 0.0)
                all_losses.append(sample["dpo_loss"])
            offset += batch_size

    sample_rows.sort(key=lambda item: item["reward_margin"])
    failures = [row for row in sample_rows if not row["reward_correct"]]
    summary = {
        "dataset_tag": args.dataset_tag,
        "num_examples": len(sample_rows),
        "accuracy": sum(all_correct) / max(len(all_correct), 1),
        "mean_reward_margin": sum(all_margins) / max(len(all_margins), 1),
        "mean_dpo_loss": sum(all_losses) / max(len(all_losses), 1),
        "num_failures": len(failures),
        "adapter_path": args.adapter_path,
        "config_path": args.config,
        "dataset_path": args.dataset_path,
        "dataset_repo": args.dataset_repo,
    }
    if all_value_cosine:
        summary["mean_value_cosine"] = sum(all_value_cosine) / len(all_value_cosine)

    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "worst_margin_samples.json", sample_rows[: args.top_k])
    write_json(output_dir / "failure_samples.json", failures[: args.top_k])
    write_jsonl(output_dir / "sample_scores.jsonl", sample_rows)


if __name__ == "__main__":
    main()
