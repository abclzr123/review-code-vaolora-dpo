import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train_dpo import format_pair, get_torch_dtype, load_pref_dataset, load_yaml, maybe_limit_dataset, masked_mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def encode_response_batch(tokenizer, prompt_texts, response_texts, max_length):
    texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
    prompt_only = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    full_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    completion_mask = torch.zeros_like(full_inputs["attention_mask"])
    prompt_lengths = prompt_only["attention_mask"].sum(dim=1)
    full_lengths = full_inputs["attention_mask"].sum(dim=1)
    for idx, (prompt_len, full_len) in enumerate(zip(prompt_lengths.tolist(), full_lengths.tolist())):
        start = min(prompt_len, full_inputs["input_ids"].shape[1])
        end = min(full_len, full_inputs["input_ids"].shape[1])
        if end > start:
            completion_mask[idx, start:end] = 1
    full_inputs["completion_mask"] = completion_mask
    return full_inputs


def mean_pool_last_hidden(model, tokenizer_inputs):
    model_inputs = {k: v for k, v in tokenizer_inputs.items() if k != "completion_mask"}
    outputs = model(**model_inputs, use_cache=False, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    completion_mask = tokenizer_inputs["completion_mask"].bool()
    return masked_mean(hidden_states, completion_mask)


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = dict(cfg["data"])
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"]["tokenizer_name_or_path"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = get_torch_dtype(model_cfg.get("torch_dtype", "bf16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name_or_path"],
        torch_dtype=dtype,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    model.to(args.device)
    model.eval()

    dataset = load_pref_dataset(data_cfg)[args.split]
    dataset = maybe_limit_dataset(dataset, args.max_samples)
    dataset = dataset.map(
        lambda example: format_pair(example, tokenizer),
        remove_columns=dataset.column_names,
        desc="format anchor dataset",
    )

    prompt_texts = []
    chosen_texts = []
    rejected_texts = []
    for row in dataset:
        prompt_texts.append(row["prompt"])
        chosen_texts.append(row["chosen"])
        rejected_texts.append(row["rejected"])

    chosen_pooled_batches = []
    rejected_pooled_batches = []
    max_length = int(trainer_cfg["max_prompt_length"]) + int(trainer_cfg["max_completion_length"])

    with torch.no_grad():
        for start in range(0, len(prompt_texts), args.batch_size):
            end = start + args.batch_size
            chosen_inputs = encode_response_batch(
                tokenizer,
                prompt_texts[start:end],
                chosen_texts[start:end],
                max_length=max_length,
            )
            rejected_inputs = encode_response_batch(
                tokenizer,
                prompt_texts[start:end],
                rejected_texts[start:end],
                max_length=max_length,
            )
            chosen_inputs = {k: v.to(args.device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(args.device) for k, v in rejected_inputs.items()}
            chosen_pooled_batches.append(mean_pool_last_hidden(model, chosen_inputs).cpu())
            rejected_pooled_batches.append(mean_pool_last_hidden(model, rejected_inputs).cpu())

    chosen_pooled = torch.cat(chosen_pooled_batches, dim=0)
    rejected_pooled = torch.cat(rejected_pooled_batches, dim=0)
    mu_pos = chosen_pooled.mean(dim=0)
    mu_neg = rejected_pooled.mean(dim=0)
    value_direction = mu_pos - mu_neg
    value_direction = value_direction / value_direction.norm().clamp_min(1.0e-12)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "value_direction": value_direction,
            "mu_pos": mu_pos,
            "mu_neg": mu_neg,
            "max_samples": len(dataset),
            "split": args.split,
                "pooling": "completion_mean",
            "model_name_or_path": model_cfg["model_name_or_path"],
            "dataset_path": data_cfg.get("dataset_path"),
        },
        output_path,
    )
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_path": str(output_path),
                "max_samples": len(dataset),
                "split": args.split,
                "device": args.device,
                "model_name_or_path": model_cfg["model_name_or_path"],
                "dataset_path": data_cfg.get("dataset_path"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
