import argparse
from contextlib import nullcontext
import json
import os
import shutil
import socket
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import flush_left, flush_right
from trl.trainer.utils import selective_log_softmax


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, list):
                    parts.append(normalize_content(text))
                else:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return normalize_content(value.get("content") or value.get("text") or "")
    return str(value)


def ensure_messages(value: Any, default_role: str) -> Iterable[Dict[str, str]]:
    if isinstance(value, list):
        messages = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role", default_role)
                content = normalize_content(item.get("content"))
                messages.append({"role": role, "content": content})
        if messages:
            return messages
    if isinstance(value, dict):
        role = value.get("role", default_role)
        content = normalize_content(value.get("content"))
        return [{"role": role, "content": content}]
    if isinstance(value, str):
        return [{"role": default_role, "content": value}]
    return [{"role": default_role, "content": normalize_content(value)}]


def extract_assistant_text(value: Any) -> str:
    if isinstance(value, list):
        assistant_segments = []
        for item in value:
            if isinstance(item, dict) and item.get("role") == "assistant":
                assistant_segments.append(normalize_content(item.get("content")))
        if assistant_segments:
            return "\n".join(segment for segment in assistant_segments if segment)
    if isinstance(value, dict) and value.get("role") == "assistant":
        return normalize_content(value.get("content"))
    return normalize_content(value)


def build_prompt_text(tokenizer, prompt_value: Any) -> str:
    messages = list(ensure_messages(prompt_value, "user"))
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def format_pair(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    prompt_value = example.get("prompt")
    if prompt_value is None and "chosen" in example and isinstance(example["chosen"], list):
        chosen_messages = example["chosen"]
        prompt_value = chosen_messages[:-1]
    prompt_text = build_prompt_text(tokenizer, prompt_value)
    chosen_text = extract_assistant_text(example.get("chosen"))
    rejected_text = extract_assistant_text(example.get("rejected"))
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def load_pref_dataset(data_cfg: Dict[str, Any]) -> DatasetDict:
    dataset_path = data_cfg.get("dataset_path")
    dataset_repo = data_cfg.get("dataset_repo")
    if dataset_path and Path(dataset_path).exists():
        loaded = load_from_disk(dataset_path)
    elif dataset_repo:
        loaded = load_dataset(dataset_repo)
    else:
        raise ValueError("Need either data.dataset_path or data.dataset_repo")

    if isinstance(loaded, DatasetDict):
        return loaded
    raise ValueError("Expected a DatasetDict with train/eval splits")


def maybe_limit_dataset(dataset: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None or limit <= 0:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def get_torch_dtype(name: str):
    import torch

    normalized = name.lower()
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    return torch.float32


def expand_template(value: str, context: Dict[str, str]) -> str:
    return value.format(**context) if "{" in value else value


@dataclass
class RunPaths:
    log_dir: str
    checkpoint_dir: str
    eval_dir: str
    final_dir: str
    best_dir: str
    metrics_path: str
    env_path: str
    launch_path: str


def resolve_paths(cfg: Dict[str, Any], run_name: str) -> RunPaths:
    context = {"run_name": run_name}
    paths_cfg = cfg["paths"]
    log_dir = ensure_dir(expand_template(paths_cfg["log_dir_template"], context))
    checkpoint_dir = ensure_dir(expand_template(paths_cfg["checkpoint_dir_template"], context))
    eval_dir = ensure_dir(expand_template(paths_cfg["eval_dir_template"], context))
    final_dir = ensure_dir(os.path.join(checkpoint_dir, "final"))
    best_dir = os.path.join(checkpoint_dir, "best")
    return RunPaths(
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        eval_dir=eval_dir,
        final_dir=final_dir,
        best_dir=best_dir,
        metrics_path=os.path.join(log_dir, "metrics.jsonl"),
        env_path=os.path.join(log_dir, "env.txt"),
        launch_path=os.path.join(log_dir, "launch_cmd.txt"),
    )


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: str):
        self.metrics_path = metrics_path
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = dict(logs)
        payload["step"] = state.global_step
        payload["epoch"] = state.epoch
        payload["timestamp"] = datetime.utcnow().isoformat() + "Z"
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def safe_normalize(tensor: torch.Tensor, dim: int, eps: float = 1.0e-12) -> torch.Tensor:
    norm = tensor.norm(dim=dim, keepdim=True).clamp_min(eps)
    return tensor / norm


def masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden_states.dtype)
    denom = weights.sum(dim=1).clamp_min(eps)
    return (hidden_states * weights).sum(dim=1) / denom


def canonicalize_module_name(module_name: str) -> str:
    parts = module_name.split(".")
    while parts and parts[0] == "module":
        parts = parts[1:]
    while parts and parts[0] == "base_model":
        parts = parts[1:]
    while len(parts) > 1 and parts[0] == "model" and parts[1] == "model":
        parts = parts[1:]
    return ".".join(parts)


def iter_lora_factors(model) -> Iterable[Dict[str, torch.Tensor]]:
    for module_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        if not hasattr(lora_a, "keys") or not hasattr(lora_b, "keys"):
            continue
        for adapter_name in lora_a.keys():
            if adapter_name not in lora_b:
                continue
            a_layer = lora_a[adapter_name]
            b_layer = lora_b[adapter_name]
            if not hasattr(a_layer, "weight") or not hasattr(b_layer, "weight"):
                continue
            yield {
                "module_name": module_name,
                "canonical_module_name": canonicalize_module_name(module_name),
                "adapter_name": adapter_name,
                "A": a_layer.weight,
                "B": b_layer.weight,
            }


def compute_factor_orthogonal_regularizer(model) -> Dict[str, torch.Tensor]:
    orth_error_a_terms = []
    orth_error_b_terms = []
    delta_w_norm_terms = []

    for factor in iter_lora_factors(model):
        a_weight = factor["A"]
        b_weight = factor["B"]
        rank = a_weight.shape[0]
        eye = torch.eye(rank, device=a_weight.device, dtype=a_weight.dtype)
        orth_error_a_terms.append(((a_weight @ a_weight.transpose(0, 1)) - eye).pow(2).mean())
        orth_error_b_terms.append(((b_weight.transpose(0, 1) @ b_weight) - eye).pow(2).mean())
        delta_w_norm_terms.append(torch.matmul(b_weight, a_weight).norm(p="fro"))

    if not orth_error_a_terms:
        zero = torch.zeros((), dtype=torch.float32)
        return {
            "orth_loss": zero,
            "orth_error_A": zero,
            "orth_error_B": zero,
            "delta_w_fro_norm": zero,
            "num_lora_factors": zero,
        }

    orth_error_a = torch.stack(orth_error_a_terms).mean()
    orth_error_b = torch.stack(orth_error_b_terms).mean()
    delta_w_fro_norm = torch.stack(delta_w_norm_terms).mean()
    orth_loss = orth_error_a + orth_error_b
    return {
        "orth_loss": orth_loss,
        "orth_error_A": orth_error_a,
        "orth_error_B": orth_error_b,
        "delta_w_fro_norm": delta_w_fro_norm,
        "num_lora_factors": torch.tensor(float(len(orth_error_a_terms)), device=orth_loss.device),
    }


def compute_subspace_projection_regularizer(
    model,
    orth_basis_map: Dict[str, Dict[str, torch.Tensor]],
    projection_mode: str,
) -> Dict[str, torch.Tensor]:
    orth_loss_terms = []
    input_ratio_terms = []
    output_ratio_terms = []
    delta_w_norm_terms = []
    matched_factors = 0
    missing_factors = 0

    for factor in iter_lora_factors(model):
        canonical_name = factor["canonical_module_name"]
        basis = orth_basis_map.get(canonical_name)
        if basis is None:
            missing_factors += 1
            continue

        matched_factors += 1
        a_weight = factor["A"]
        b_weight = factor["B"]
        delta_w = torch.matmul(b_weight, a_weight)
        delta_w_sq = delta_w.pow(2).sum().clamp_min(1.0e-12)
        delta_w_norm_terms.append(delta_w.norm(p="fro"))

        input_ratio = torch.zeros((), device=delta_w.device, dtype=delta_w.dtype)
        output_ratio = torch.zeros((), device=delta_w.device, dtype=delta_w.dtype)
        orth_term = torch.zeros((), device=delta_w.device, dtype=delta_w.dtype)

        if projection_mode in {"input", "both"}:
            right_basis = basis["right_basis"].to(device=delta_w.device, dtype=delta_w.dtype)
            input_ratio = torch.matmul(delta_w, right_basis).pow(2).sum() / delta_w_sq
            orth_term = orth_term + input_ratio

        if projection_mode in {"output", "both"}:
            left_basis = basis["left_basis"].to(device=delta_w.device, dtype=delta_w.dtype)
            output_ratio = torch.matmul(left_basis.transpose(0, 1), delta_w).pow(2).sum() / delta_w_sq
            orth_term = orth_term + output_ratio

        orth_loss_terms.append(orth_term)
        input_ratio_terms.append(input_ratio)
        output_ratio_terms.append(output_ratio)

    if not orth_loss_terms:
        zero = torch.zeros((), dtype=torch.float32)
        return {
            "orth_loss": zero,
            "orth_subspace_input_ratio": zero,
            "orth_subspace_output_ratio": zero,
            "delta_w_fro_norm": zero,
            "matched_factors": zero,
            "missing_factors": torch.tensor(float(missing_factors), dtype=torch.float32),
        }

    orth_loss = torch.stack(orth_loss_terms).mean()
    input_ratio = torch.stack(input_ratio_terms).mean()
    output_ratio = torch.stack(output_ratio_terms).mean()
    delta_w_fro_norm = torch.stack(delta_w_norm_terms).mean()
    return {
        "orth_loss": orth_loss,
        "orth_subspace_input_ratio": input_ratio,
        "orth_subspace_output_ratio": output_ratio,
        "delta_w_fro_norm": delta_w_fro_norm,
        "matched_factors": torch.tensor(float(matched_factors), device=orth_loss.device),
        "missing_factors": torch.tensor(float(missing_factors), device=orth_loss.device),
    }


def compute_value_alignment_loss(
    chosen_pooled: torch.Tensor,
    rejected_pooled: torch.Tensor,
    value_direction: torch.Tensor,
    loss_type: str,
    margin: float,
) -> Dict[str, torch.Tensor]:
    direction = safe_normalize(chosen_pooled - rejected_pooled, dim=-1)
    anchor = safe_normalize(value_direction, dim=0).unsqueeze(0).to(direction)
    cosine = F.cosine_similarity(direction, anchor.expand_as(direction), dim=-1)
    if loss_type == "margin":
        value_loss = torch.clamp_min(margin - cosine, 0.0)
    else:
        value_loss = 1.0 - cosine
    return {
        "value_loss": value_loss,
        "value_cosine": cosine,
        "chosen_hidden_norm": chosen_pooled.norm(dim=-1),
        "rejected_hidden_norm": rejected_pooled.norm(dim=-1),
    }


def compute_margin_penalty(
    score: torch.Tensor,
    margin: torch.Tensor,
    penalty_type: str,
    temperature: float,
) -> torch.Tensor:
    gap = margin - score
    if penalty_type == "hinge":
        return torch.clamp_min(gap, 0.0)
    if penalty_type in {"softplus", "logistic"}:
        if temperature <= 0:
            raise ValueError("value_loss_temperature must be > 0 for smooth value losses")
        return F.softplus(gap / temperature) * temperature
    raise ValueError(f"Unsupported penalty type: {penalty_type}")


def compute_value_alignment_v2_loss(
    chosen_pooled: torch.Tensor,
    rejected_pooled: torch.Tensor,
    chosen_ref_pooled: torch.Tensor,
    rejected_ref_pooled: torch.Tensor,
    loss_type: str,
    pos_margin: float,
    pair_margin: float,
    pair_loss_alpha: float,
    pair_similarity_max: float,
    diff_norm_min: float,
    gate_strategy: str,
    tight_pos_margin: float,
    tight_pair_margin: float,
    loss_temperature: float,
) -> Dict[str, torch.Tensor]:
    ref_direction = safe_normalize(chosen_ref_pooled - rejected_ref_pooled, dim=-1)
    current_direction = safe_normalize(chosen_pooled - rejected_pooled, dim=-1)

    chosen_delta = safe_normalize(chosen_pooled - rejected_ref_pooled, dim=-1)
    rejected_delta = safe_normalize(rejected_pooled - rejected_ref_pooled, dim=-1)

    chosen_score = F.cosine_similarity(chosen_delta, ref_direction, dim=-1)
    rejected_score = F.cosine_similarity(rejected_delta, ref_direction, dim=-1)
    pair_score = chosen_score - rejected_score
    direction_cosine = F.cosine_similarity(current_direction, ref_direction, dim=-1)

    if loss_type not in {
        "chosen_pair_margin",
        "chosen_margin",
        "chosen_pair_softplus",
        "chosen_softplus",
        "chosen_pair_logistic",
        "chosen_logistic",
    }:
        raise ValueError(f"Unsupported value v2 loss type: {loss_type}")

    pair_enabled = loss_type.startswith("chosen_pair")
    if loss_type.endswith("_margin"):
        penalty_type = "hinge"
    elif loss_type.endswith("_softplus"):
        penalty_type = "softplus"
    elif loss_type.endswith("_logistic"):
        penalty_type = "logistic"
    else:
        raise ValueError(f"Unable to infer penalty type from value v2 loss type: {loss_type}")

    ref_pair_cosine = F.cosine_similarity(chosen_ref_pooled, rejected_ref_pooled, dim=-1)
    ref_diff_norm = (chosen_ref_pooled - rejected_ref_pooled).norm(dim=-1)

    gate = torch.ones_like(chosen_score)
    if pair_similarity_max < 1.0:
        gate = gate * (ref_pair_cosine <= pair_similarity_max).to(chosen_score.dtype)
    if diff_norm_min > 0:
        gate = gate * (ref_diff_norm >= diff_norm_min).to(chosen_score.dtype)

    similarity_weight = torch.ones_like(chosen_score)
    if pair_similarity_max < 1.0:
        similarity_denom = max(1.0 - pair_similarity_max, 1.0e-6)
        similarity_weight = torch.clamp((1.0 - ref_pair_cosine) / similarity_denom, 0.0, 1.0)

    diff_weight = torch.ones_like(chosen_score)
    if diff_norm_min > 0:
        diff_weight = torch.clamp(ref_diff_norm / max(diff_norm_min, 1.0e-6), 0.0, 1.0)

    soft_weight = similarity_weight * diff_weight

    effective_pos_margin = torch.full_like(chosen_score, pos_margin)
    effective_pair_margin = torch.full_like(chosen_score, pair_margin)
    effective_weight = gate

    if gate_strategy == "hard":
        pass
    elif gate_strategy == "soft_weight":
        effective_weight = soft_weight
    elif gate_strategy == "tight_margin":
        active_mask = gate > 0
        effective_pos_margin = torch.where(
            active_mask,
            torch.full_like(chosen_score, tight_pos_margin),
            effective_pos_margin,
        )
        effective_pair_margin = torch.where(
            active_mask,
            torch.full_like(chosen_score, tight_pair_margin),
            effective_pair_margin,
        )
    else:
        raise ValueError(f"Unsupported value gate strategy: {gate_strategy}")

    positive_loss = compute_margin_penalty(
        chosen_score,
        effective_pos_margin,
        penalty_type,
        loss_temperature,
    )
    if pair_enabled:
        pair_loss = compute_margin_penalty(
            pair_score,
            effective_pair_margin,
            penalty_type,
            loss_temperature,
        )
    else:
        pair_loss = torch.zeros_like(positive_loss)
    value_loss = positive_loss + pair_loss_alpha * pair_loss

    return {
        "value_loss": value_loss,
        "value_raw_loss": value_loss,
        "value_effective_loss": value_loss * effective_weight,
        "value_effective_positive_loss": positive_loss * effective_weight,
        "value_effective_pair_loss": pair_loss * effective_weight,
        "value_gate": gate,
        "value_soft_weight": soft_weight,
        "value_effective_weight": effective_weight,
        "value_similarity_weight": similarity_weight,
        "value_diff_weight": diff_weight,
        "value_direction_cosine": direction_cosine,
        "value_chosen_score": chosen_score,
        "value_rejected_score": rejected_score,
        "value_pair_score": pair_score,
        "value_positive_loss": positive_loss,
        "value_pair_loss": pair_loss,
        "value_ref_pair_cosine": ref_pair_cosine,
        "value_ref_diff_norm": ref_diff_norm,
        "value_effective_pos_margin": effective_pos_margin,
        "value_effective_pair_margin": effective_pair_margin,
        "value_positive_gap": chosen_score - effective_pos_margin,
        "value_pair_gap": pair_score - effective_pair_margin,
        "chosen_hidden_norm": chosen_pooled.norm(dim=-1),
        "rejected_hidden_norm": rejected_pooled.norm(dim=-1),
    }


class VAOLoRADPOTrainer(DPOTrainer):
    def __init__(self, *args, method_cfg: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        method_cfg = method_cfg or {}
        self.method_cfg = method_cfg
        self.enable_orth = bool(method_cfg.get("enable_orth", False))
        self.enable_value_alignment = bool(method_cfg.get("enable_value_alignment", False))
        self.lambda_orth = float(method_cfg.get("lambda_orth", 0.0))
        self.lambda_val = float(method_cfg.get("lambda_val", 0.0))
        self.orth_strategy = method_cfg.get("orth_strategy", "soft_lora_factor")
        self.orth_basis_path = method_cfg.get("orth_basis_path")
        self.orth_projection_mode = method_cfg.get("orth_projection_mode", "both")
        self.value_anchor_path = method_cfg.get("value_anchor_path")
        self.value_pooling = method_cfg.get("value_pooling", "completion_mean")
        self.value_loss_type = method_cfg.get("value_loss_type", "cosine")
        self.value_margin = float(method_cfg.get("value_margin", 0.2))
        self.value_direction_mode = method_cfg.get("value_direction_mode", "global_anchor")
        self.value_target_layers_cfg = method_cfg.get("value_target_layers")
        self.value_target_layer_cfg = method_cfg.get("value_target_layer")
        self.value_pos_margin = float(method_cfg.get("value_pos_margin", 0.10))
        self.value_pair_margin = float(method_cfg.get("value_pair_margin", self.value_margin))
        self.value_pair_loss_alpha = float(method_cfg.get("value_pair_loss_alpha", 1.0))
        self.value_pair_similarity_max = float(method_cfg.get("value_pair_similarity_max", 1.0))
        self.value_diff_norm_min = float(method_cfg.get("value_diff_norm_min", 0.0))
        self.value_gate_strategy = method_cfg.get("value_gate_strategy", "hard")
        self.value_tight_pos_margin = float(method_cfg.get("value_tight_pos_margin", self.value_pos_margin))
        self.value_tight_pair_margin = float(method_cfg.get("value_tight_pair_margin", self.value_pair_margin))
        self.value_loss_temperature = float(method_cfg.get("value_loss_temperature", 0.10))
        self.value_warmup_ratio = float(method_cfg.get("value_warmup_ratio", 0.0))
        self.value_warmup_steps = int(method_cfg.get("value_warmup_steps", 0))
        self._orth_basis_cache = None
        self._value_direction_cache = None
        self._value_target_layers_cache = None
        if self.enable_orth and self.orth_strategy not in {"soft_lora_factor", "subspace_projection"}:
            raise ValueError(f"Unsupported orth_strategy: {self.orth_strategy}")
        if self.enable_orth and self.orth_strategy == "subspace_projection":
            if not self.orth_basis_path:
                raise ValueError("method.orth_basis_path is required when orth_strategy=subspace_projection")
            if self.orth_projection_mode not in {"input", "output", "both"}:
                raise ValueError("method.orth_projection_mode must be one of: input, output, both")
        if self.enable_value_alignment and self.value_direction_mode == "per_pair_reference" and bool(
            getattr(self.args, "gradient_checkpointing", False)
        ):
            raise ValueError(
                "value_direction_mode=per_pair_reference is currently incompatible with gradient_checkpointing=true; "
                "set trainer.gradient_checkpointing=false for value v2 smoke runs."
            )

    def _unwrap_model_for_value(self, model):
        current = model
        while hasattr(current, "module"):
            current = current.module
        return current

    def _load_value_direction(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.value_anchor_path:
            raise ValueError("method.value_anchor_path is required when enable_value_alignment=true")
        if self._value_direction_cache is None:
            anchor_path = Path(self.value_anchor_path)
            if not anchor_path.exists():
                raise FileNotFoundError(f"value anchor file not found: {anchor_path}")
            payload = torch.load(anchor_path, map_location="cpu")
            if isinstance(payload, dict):
                if "value_direction" in payload:
                    vector = payload["value_direction"]
                elif "vector" in payload:
                    vector = payload["vector"]
                else:
                    raise KeyError("value anchor payload must contain `value_direction` or `vector`")
            else:
                vector = payload
            self._value_direction_cache = safe_normalize(vector.float(), dim=0)
        return self._value_direction_cache.to(device=device, dtype=dtype)

    def _load_orth_basis_map(self) -> Dict[str, Dict[str, torch.Tensor]]:
        if self._orth_basis_cache is not None:
            return self._orth_basis_cache

        if not self.orth_basis_path:
            raise ValueError("method.orth_basis_path is required when orth_strategy=subspace_projection")

        basis_path = Path(self.orth_basis_path)
        if not basis_path.exists():
            raise FileNotFoundError(f"orth basis file not found: {basis_path}")

        payload = torch.load(basis_path, map_location="cpu")
        module_bases = payload.get("module_bases") if isinstance(payload, dict) else None
        if not isinstance(module_bases, dict):
            raise ValueError("orth basis payload must be a dict containing `module_bases`")

        normalized = {}
        for module_name, basis in module_bases.items():
            if not isinstance(basis, dict):
                continue
            left_basis = basis.get("left_basis")
            right_basis = basis.get("right_basis")
            if left_basis is None or right_basis is None:
                continue
            normalized[canonicalize_module_name(module_name)] = {
                "left_basis": left_basis.float(),
                "right_basis": right_basis.float(),
            }

        if not normalized:
            raise ValueError(f"No valid module bases found in orth basis payload: {basis_path}")

        self._orth_basis_cache = normalized
        return self._orth_basis_cache

    def _resolve_value_target_layers(self, model) -> List[int]:
        if self._value_target_layers_cache is not None:
            return self._value_target_layers_cache

        base_model = self._unwrap_model_for_value(model)
        num_hidden_layers = int(getattr(base_model.config, "num_hidden_layers"))
        raw_layers = self.value_target_layers_cfg
        if raw_layers is None:
            raw_layers = self.value_target_layer_cfg

        if raw_layers is None:
            layers = [num_hidden_layers]
        elif isinstance(raw_layers, int):
            layers = [raw_layers]
        elif isinstance(raw_layers, str):
            normalized = raw_layers.strip().lower()
            if normalized == "last":
                layers = [num_hidden_layers]
            else:
                layers = [int(part.strip()) for part in raw_layers.split(",") if part.strip()]
        elif isinstance(raw_layers, list):
            layers = [int(item) for item in raw_layers]
        else:
            raise ValueError(f"Unsupported value target layers config: {raw_layers}")

        resolved = []
        for layer in layers:
            if layer < 0:
                layer = num_hidden_layers + 1 + layer
            if layer < 1 or layer > num_hidden_layers:
                raise ValueError(f"value target layer {layer} out of range 1..{num_hidden_layers}")
            resolved.append(layer)

        self._value_target_layers_cache = sorted(set(resolved))
        return self._value_target_layers_cache

    def _build_concatenated_sequence_tensors(self, batch: Dict[str, torch.Tensor]):
        if self.is_encoder_decoder:
            raise NotImplementedError("value alignment v1 currently supports decoder-only models only")
        if self.padding_free:
            raise NotImplementedError("value alignment v1 does not support padding_free mode")

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        completion_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
            dim=1,
        )

        if self.max_length is not None and self.max_length < attention_mask.size(1):
            if self.truncation_mode == "keep_start":
                attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)
                attention_mask = attention_mask[:, : self.max_length]
                input_ids = input_ids[:, : self.max_length]
                completion_mask = completion_mask[:, : self.max_length]
            elif self.truncation_mode == "keep_end":
                attention_mask, input_ids, completion_mask = flush_right(attention_mask, input_ids, completion_mask)
                input_ids = input_ids[:, -self.max_length :]
                attention_mask = attention_mask[:, -self.max_length :]
                completion_mask = completion_mask[:, -self.max_length :]
                attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
        else:
            attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        return input_ids, attention_mask, completion_mask.bool(), batch["prompt_input_ids"].shape[0]

    def _compute_pooled_hidden_states(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        disable_adapter: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if self.value_pooling != "completion_mean":
            raise ValueError(f"Unsupported value_pooling: {self.value_pooling}")

        input_ids, attention_mask, completion_mask, num_examples = self._build_concatenated_sequence_tensors(batch)
        base_model = self._unwrap_model_for_value(model)
        execution_model = base_model if disable_adapter else model
        adapter_ctx = base_model.disable_adapter() if disable_adapter and hasattr(base_model, "disable_adapter") else nullcontext()
        grad_ctx = torch.no_grad() if disable_adapter else nullcontext()
        with adapter_ctx, grad_ctx:
            outputs = execution_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )

        selected_layers = self._resolve_value_target_layers(model)
        pooled_per_layer = []
        for layer_idx in selected_layers:
            hidden_states = outputs.hidden_states[layer_idx]
            pooled_per_layer.append(masked_mean(hidden_states, completion_mask))

        pooled = torch.stack(pooled_per_layer, dim=0).mean(dim=0)
        return {
            "chosen_pooled": pooled[:num_examples],
            "rejected_pooled": pooled[num_examples:],
            "num_value_layers": float(len(selected_layers)),
        }

    def _pool_hidden_states_from_outputs(
        self,
        model,
        hidden_states: tuple[torch.Tensor, ...],
        completion_mask: torch.Tensor,
        num_examples: int,
    ) -> Dict[str, torch.Tensor]:
        selected_layers = self._resolve_value_target_layers(model)
        pooled_per_layer = []
        for layer_idx in selected_layers:
            pooled_per_layer.append(masked_mean(hidden_states[layer_idx], completion_mask))

        pooled = torch.stack(pooled_per_layer, dim=0).mean(dim=0)
        return {
            "chosen_pooled": pooled[:num_examples],
            "rejected_pooled": pooled[num_examples:],
            "num_value_layers": float(len(selected_layers)),
        }

    def _get_value_warmup_scale(self, train_eval: Literal["train", "eval"]) -> float:
        if train_eval != "train":
            return 1.0

        warmup_steps = self.value_warmup_steps
        if warmup_steps <= 0 and self.value_warmup_ratio > 0:
            total_steps = int(getattr(self.state, "max_steps", 0) or self.args.max_steps or 0)
            if total_steps > 0:
                warmup_steps = max(1, int(total_steps * self.value_warmup_ratio))

        if warmup_steps <= 0:
            return 1.0

        current_step = int(getattr(self.state, "global_step", 0)) + 1
        return min(1.0, current_step / float(warmup_steps))

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[list, torch.LongTensor]], is_ref_model: bool = False
    ):
        num_examples = batch["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            return super().concatenated_forward(model, batch, is_ref_model=is_ref_model)

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
            dim=1,
        )

        if self.max_length is not None and self.max_length < attention_mask.size(1):
            if self.truncation_mode == "keep_start":
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                attention_mask = attention_mask[:, : self.max_length]
                input_ids = input_ids[:, : self.max_length]
                loss_mask = loss_mask[:, : self.max_length]
            elif self.truncation_mode == "keep_end":
                attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                input_ids = input_ids[:, -self.max_length :]
                attention_mask = attention_mask[:, -self.max_length :]
                loss_mask = loss_mask[:, -self.max_length :]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            else:
                raise ValueError(
                    f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', 'keep_start']."
                )
        else:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        if self.use_logits_to_keep:
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
            model_kwargs["logits_to_keep"] = logits_to_keep

        if self.padding_free:
            raise NotImplementedError("value alignment path does not support padding_free mode")

        completion_mask = loss_mask.bool()
        model_kwargs["attention_mask"] = attention_mask
        model_kwargs["output_hidden_states"] = True

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if self.use_logits_to_keep:
            labels = labels[:, -logits_to_keep:]
            loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None:
            chosen_logits = logits[:num_examples, :-1]
            chosen_labels = labels[:num_examples, :-1]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )

        if self.loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)

        if self.args.ld_alpha is not None and not is_ref_model:
            completion_lengths = loss_mask.sum(dim=1)
            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)
            public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)
            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)
            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
            front_logps = (per_token_logps * front_mask).sum(dim=1)
            rear_logps = (per_token_logps * rear_mask).sum(dim=1)
            all_logps = front_logps + self.args.ld_alpha * rear_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["mean_chosen_logits"] = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

        if self.enable_value_alignment and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            pooled_hidden = self._pool_hidden_states_from_outputs(
                model=model,
                hidden_states=outputs.hidden_states,
                completion_mask=completion_mask,
                num_examples=num_examples,
            )
            output["value_chosen_pooled"] = pooled_hidden["chosen_pooled"]
            output["value_rejected_pooled"] = pooled_hidden["rejected_pooled"]
            output["value_num_layers"] = pooled_hidden["num_value_layers"]

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ):
        if self.args.use_liger_loss and (self.enable_orth or self.enable_value_alignment):
            raise NotImplementedError("orth/value regularization v1 does not support use_liger_loss")

        metrics = {}
        model_output = self.concatenated_forward(model, batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )

        pref_loss = losses.mean()
        total_loss = losses
        metrics[f"{prefix}loss/pref"] = pref_loss.detach().item()

        if self.enable_orth and self.lambda_orth > 0:
            if self.orth_strategy == "soft_lora_factor":
                orth_stats = compute_factor_orthogonal_regularizer(model)
                metrics[f"{prefix}orth/error_A"] = orth_stats["orth_error_A"].detach().item()
                metrics[f"{prefix}orth/error_B"] = orth_stats["orth_error_B"].detach().item()
                metrics[f"{prefix}orth/subspace_input_ratio"] = 0.0
                metrics[f"{prefix}orth/subspace_output_ratio"] = 0.0
                metrics[f"{prefix}orth/matched_factors"] = orth_stats["num_lora_factors"].detach().item()
                metrics[f"{prefix}orth/missing_factors"] = 0.0
            elif self.orth_strategy == "subspace_projection":
                orth_stats = compute_subspace_projection_regularizer(
                    model,
                    self._load_orth_basis_map(),
                    self.orth_projection_mode,
                )
                metrics[f"{prefix}orth/error_A"] = 0.0
                metrics[f"{prefix}orth/error_B"] = 0.0
                metrics[f"{prefix}orth/subspace_input_ratio"] = (
                    orth_stats["orth_subspace_input_ratio"].detach().item()
                )
                metrics[f"{prefix}orth/subspace_output_ratio"] = (
                    orth_stats["orth_subspace_output_ratio"].detach().item()
                )
                metrics[f"{prefix}orth/matched_factors"] = orth_stats["matched_factors"].detach().item()
                metrics[f"{prefix}orth/missing_factors"] = orth_stats["missing_factors"].detach().item()
            else:
                raise ValueError(f"Unsupported orth_strategy: {self.orth_strategy}")

            total_loss = total_loss + self.lambda_orth * orth_stats["orth_loss"]
            metrics[f"{prefix}orth/loss"] = orth_stats["orth_loss"].detach().item()
            metrics[f"{prefix}orth/delta_w_fro_norm"] = orth_stats["delta_w_fro_norm"].detach().item()
            metrics[f"{prefix}orth/num_lora_factors"] = metrics[f"{prefix}orth/matched_factors"]
        else:
            metrics[f"{prefix}orth/loss"] = 0.0
            metrics[f"{prefix}orth/error_A"] = 0.0
            metrics[f"{prefix}orth/error_B"] = 0.0
            metrics[f"{prefix}orth/subspace_input_ratio"] = 0.0
            metrics[f"{prefix}orth/subspace_output_ratio"] = 0.0
            metrics[f"{prefix}orth/delta_w_fro_norm"] = 0.0
            metrics[f"{prefix}orth/num_lora_factors"] = 0.0
            metrics[f"{prefix}orth/matched_factors"] = 0.0
            metrics[f"{prefix}orth/missing_factors"] = 0.0

        if self.enable_value_alignment and self.lambda_val > 0:
            if "value_chosen_pooled" in model_output and "value_rejected_pooled" in model_output:
                pooled_hidden = {
                    "chosen_pooled": model_output["value_chosen_pooled"],
                    "rejected_pooled": model_output["value_rejected_pooled"],
                    "num_value_layers": float(model_output.get("value_num_layers", 0.0)),
                }
            else:
                pooled_hidden = self._compute_pooled_hidden_states(model, batch)
            warmup_scale = self._get_value_warmup_scale(train_eval)

            if self.value_direction_mode == "per_pair_reference":
                ref_pooled_hidden = self._compute_pooled_hidden_states(model, batch, disable_adapter=True)
                value_stats = compute_value_alignment_v2_loss(
                    pooled_hidden["chosen_pooled"],
                    pooled_hidden["rejected_pooled"],
                    ref_pooled_hidden["chosen_pooled"],
                    ref_pooled_hidden["rejected_pooled"],
                    self.value_loss_type,
                    self.value_pos_margin,
                    self.value_pair_margin,
                    self.value_pair_loss_alpha,
                    self.value_pair_similarity_max,
                    self.value_diff_norm_min,
                    self.value_gate_strategy,
                    self.value_tight_pos_margin,
                    self.value_tight_pair_margin,
                    self.value_loss_temperature,
                )
                total_loss = total_loss + (self.lambda_val * warmup_scale) * value_stats["value_effective_loss"]

                gate = value_stats["value_gate"].detach()
                effective_weight = value_stats["value_effective_weight"].detach()
                active_weight = max(float(effective_weight.sum().item()), 1.0)
                metrics[f"{prefix}value/loss"] = (
                    value_stats["value_effective_loss"].detach().sum().item() / active_weight
                )
                metrics[f"{prefix}value/raw_loss"] = value_stats["value_raw_loss"].detach().mean().item()
                metrics[f"{prefix}value/direction_cosine"] = value_stats["value_direction_cosine"].detach().mean().item()
                metrics[f"{prefix}value/chosen_score"] = value_stats["value_chosen_score"].detach().mean().item()
                metrics[f"{prefix}value/rejected_score"] = value_stats["value_rejected_score"].detach().mean().item()
                metrics[f"{prefix}value/pair_score"] = value_stats["value_pair_score"].detach().mean().item()
                metrics[f"{prefix}value/positive_loss"] = value_stats["value_positive_loss"].detach().mean().item()
                metrics[f"{prefix}value/pair_loss"] = value_stats["value_pair_loss"].detach().mean().item()
                metrics[f"{prefix}value/weighted_positive_loss"] = (
                    value_stats["value_effective_positive_loss"].detach().sum().item() / active_weight
                )
                metrics[f"{prefix}value/weighted_pair_loss"] = (
                    value_stats["value_effective_pair_loss"].detach().sum().item() / active_weight
                )
                metrics[f"{prefix}value/gate_mean"] = gate.mean().item()
                metrics[f"{prefix}value/soft_weight_mean"] = (
                    value_stats["value_soft_weight"].detach().mean().item()
                )
                metrics[f"{prefix}value/effective_weight_mean"] = effective_weight.mean().item()
                metrics[f"{prefix}value/similarity_weight_mean"] = (
                    value_stats["value_similarity_weight"].detach().mean().item()
                )
                metrics[f"{prefix}value/diff_weight_mean"] = (
                    value_stats["value_diff_weight"].detach().mean().item()
                )
                metrics[f"{prefix}value/ref_pair_cosine"] = value_stats["value_ref_pair_cosine"].detach().mean().item()
                metrics[f"{prefix}value/ref_diff_norm"] = value_stats["value_ref_diff_norm"].detach().mean().item()
                metrics[f"{prefix}value/effective_pos_margin"] = (
                    value_stats["value_effective_pos_margin"].detach().mean().item()
                )
                metrics[f"{prefix}value/effective_pair_margin"] = (
                    value_stats["value_effective_pair_margin"].detach().mean().item()
                )
                metrics[f"{prefix}value/positive_gap"] = value_stats["value_positive_gap"].detach().mean().item()
                metrics[f"{prefix}value/pair_gap"] = value_stats["value_pair_gap"].detach().mean().item()
                metrics[f"{prefix}value/loss_temperature"] = self.value_loss_temperature
            else:
                value_direction = self._load_value_direction(
                    pooled_hidden["chosen_pooled"].device,
                    pooled_hidden["chosen_pooled"].dtype,
                )
                value_stats = compute_value_alignment_loss(
                    pooled_hidden["chosen_pooled"],
                    pooled_hidden["rejected_pooled"],
                    value_direction,
                    self.value_loss_type,
                    self.value_margin,
                )
                total_loss = total_loss + (self.lambda_val * warmup_scale) * value_stats["value_loss"]
                metrics[f"{prefix}value/loss"] = value_stats["value_loss"].detach().mean().item()
                metrics[f"{prefix}value/cosine"] = value_stats["value_cosine"].detach().mean().item()

            metrics[f"{prefix}value/chosen_hidden_norm"] = value_stats["chosen_hidden_norm"].detach().mean().item()
            metrics[f"{prefix}value/rejected_hidden_norm"] = (
                value_stats["rejected_hidden_norm"].detach().mean().item()
            )
            metrics[f"{prefix}value/num_layers"] = pooled_hidden["num_value_layers"]
            metrics[f"{prefix}value/warmup_scale"] = warmup_scale
        else:
            metrics[f"{prefix}value/loss"] = 0.0

        total_loss_mean = total_loss.mean()
        metrics[f"{prefix}loss/total"] = total_loss_mean.detach().item()

        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return total_loss_mean, metrics


def write_env_report(path: str, cfg_path: str, run_name: str):
    lines = [
        f"timestamp={datetime.utcnow().isoformat()}Z",
        f"hostname={socket.gethostname()}",
        f"python={sys.executable}",
        f"config_path={cfg_path}",
        f"run_name={run_name}",
    ]
    for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE"):
        lines.append(f"{key}={os.environ.get(key, '')}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_launch_command(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(sys.argv) + "\n")


def copy_best_checkpoint(best_checkpoint: Optional[str], best_dir: str):
    if not best_checkpoint or not Path(best_checkpoint).exists():
        return
    if Path(best_dir).exists():
        shutil.rmtree(best_dir, ignore_errors=True)
    shutil.copytree(best_checkpoint, best_dir)


def resolve_load_best_model_policy(trainer_cfg: Dict[str, Any]) -> Dict[str, bool]:
    requested = bool(trainer_cfg.get("load_best_model_at_end", False))
    safe_skip = bool(trainer_cfg.get("safe_skip_best_model_reload", True))
    effective = requested and not safe_skip
    if requested and safe_skip:
        print(
            "[warn] safe_skip_best_model_reload=true: disabling load_best_model_at_end to avoid end-of-run OOM; best checkpoint will be tracked from trainer_state and copied on disk instead.",
            file=sys.stderr,
        )
    return {
        "requested": requested,
        "safe_skip": safe_skip,
        "effective": effective,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    run_cfg = cfg["run"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]
    peft_cfg = cfg["peft"]

    run_name = args.run_name or run_cfg["run_name"]
    run_paths = resolve_paths(cfg, run_name)
    write_env_report(run_paths.env_path, args.config, run_name)
    write_launch_command(run_paths.launch_path)
    set_seed(int(run_cfg["seed"]))

    tokenizer_name = cfg["tokenizer"]["tokenizer_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset_path:
        data_cfg["dataset_path"] = args.dataset_path
    dataset_dict = load_pref_dataset(data_cfg)
    train_split = data_cfg.get("train_split", "train")
    eval_split = data_cfg.get("eval_split", "test")
    train_dataset = maybe_limit_dataset(dataset_dict[train_split], args.train_limit or data_cfg.get("train_limit"))
    eval_dataset = maybe_limit_dataset(dataset_dict[eval_split], args.eval_limit or data_cfg.get("eval_limit"))

    train_dataset = train_dataset.map(
        lambda example: format_pair(example, tokenizer),
        remove_columns=train_dataset.column_names,
        desc="format train dataset",
    )
    eval_dataset = eval_dataset.map(
        lambda example: format_pair(example, tokenizer),
        remove_columns=eval_dataset.column_names,
        desc="format eval dataset",
    )

    dtype = get_torch_dtype(model_cfg.get("torch_dtype", "bf16"))
    model_path = args.model_path or model_cfg["model_name_or_path"]
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except Exception:
        if "attn_implementation" not in model_kwargs:
            raise
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs.pop("attn_implementation", None)
        print(
            "[warn] failed to load model with requested attn_implementation; falling back to default attention",
            file=sys.stderr,
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, **fallback_kwargs)
    model.config.use_cache = False
    if trainer_cfg.get("gradient_checkpointing", True) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=int(peft_cfg["r"]),
        lora_alpha=int(peft_cfg["lora_alpha"]),
        lora_dropout=float(peft_cfg["lora_dropout"]),
        target_modules=list(peft_cfg["target_modules"]),
        bias=peft_cfg.get("bias", "none"),
        task_type=peft_cfg.get("task_type", "CAUSAL_LM"),
    )

    max_steps = args.max_steps if args.max_steps is not None else int(trainer_cfg.get("max_steps", -1))
    num_train_epochs = (
        args.num_train_epochs if args.num_train_epochs is not None else float(trainer_cfg.get("num_train_epochs", 1))
    )
    best_model_policy = resolve_load_best_model_policy(trainer_cfg)

    dpo_kwargs = dict(
        output_dir=run_paths.checkpoint_dir,
        logging_dir=run_paths.log_dir,
        per_device_train_batch_size=int(trainer_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(trainer_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(trainer_cfg["gradient_accumulation_steps"]),
        learning_rate=float(trainer_cfg["learning_rate"]),
        lr_scheduler_type=trainer_cfg["lr_scheduler_type"],
        warmup_ratio=float(trainer_cfg["warmup_ratio"]),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        logging_steps=int(trainer_cfg["logging_steps"]),
        save_steps=int(trainer_cfg["save_steps"]),
        eval_steps=int(trainer_cfg["eval_steps"]),
        eval_strategy=trainer_cfg["eval_strategy"],
        save_strategy=trainer_cfg["save_strategy"],
        logging_strategy=trainer_cfg["logging_strategy"],
        bf16=bool(trainer_cfg.get("bf16", False)),
        fp16=bool(trainer_cfg.get("fp16", False)),
        gradient_checkpointing=bool(trainer_cfg.get("gradient_checkpointing", True)),
        max_prompt_length=int(trainer_cfg["max_prompt_length"]),
        max_completion_length=int(trainer_cfg["max_completion_length"]),
        remove_unused_columns=False,
        report_to=[],
        save_total_limit=int(trainer_cfg.get("save_total_limit", 2)),
        load_best_model_at_end=best_model_policy["effective"],
        metric_for_best_model=trainer_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(trainer_cfg.get("greater_is_better", False)),
        seed=int(run_cfg["seed"]),
        beta=float(trainer_cfg.get("beta", 0.1)),
        loss_type=trainer_cfg.get("loss_type", "sigmoid"),
        dataset_num_proc=int(trainer_cfg.get("dataset_num_proc", 1)),
    )
    if trainer_cfg.get("gradient_checkpointing", True):
        dpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = DPOConfig(**dpo_kwargs)

    trainer = VAOLoRADPOTrainer(
        model=model,
        args=training_args,
        method_cfg=cfg.get("method", {}),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[JsonlMetricsCallback(run_paths.metrics_path)],
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(run_paths.final_dir)

    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if hasattr(trainer, "accelerator"):
        trainer.accelerator.wait_for_everyone()

    summary = {
        "run_name": run_name,
        "smoke": args.smoke,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "requested_load_best_model_at_end": best_model_policy["requested"],
        "effective_load_best_model_at_end": best_model_policy["effective"],
        "safe_skip_best_model_reload": best_model_policy["safe_skip"],
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "final_dir": run_paths.final_dir,
        "checkpoint_dir": run_paths.checkpoint_dir,
        "log_dir": run_paths.log_dir,
    }
    if trainer.is_world_process_zero():
        with open(os.path.join(run_paths.eval_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        copy_best_checkpoint(trainer.state.best_model_checkpoint, run_paths.best_dir)


if __name__ == "__main__":
    main()
