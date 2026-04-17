import argparse
import json
import logging
import os
import re
import time
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def configure_runtime() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass


configure_runtime()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NullSVDLogger:
    enabled = False
    to_stdout = False
    singular_values = ""
    scores = ""
    lam = ""
    gam = ""
    fisher_norm = ""
    ratio = ""
    importance = ""

    def log(self, file_name: str, **kv) -> None:
        return


class SVDLogger:
    def __init__(self) -> None:
        self.enabled = os.environ.get("SVD_LOG_ENABLE", "1") != "0"
        self.to_stdout = False
        self.log_dir = os.environ.get("SVD_LOG_FILE", "./logs/svd")

        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            self.to_stdout = os.environ.get("SVD_LOG_STDOUT", "1") == "1"

        self._reset_log_files()

    def _reset_log_files(self) -> None:
        self.singular_values = os.path.join(self.log_dir, "singular_values.jsonl")
        self.scores = os.path.join(self.log_dir, "scores.jsonl")
        self.lam = os.path.join(self.log_dir, "lam.jsonl")
        self.gam = os.path.join(self.log_dir, "gam.jsonl")
        self.fisher_norm = os.path.join(self.log_dir, "fisher_norm.jsonl")
        self.ratio = os.path.join(self.log_dir, "ratio.jsonl")
        self.importance = os.path.join(self.log_dir, "importance.jsonl")

        for path in (
            self.singular_values,
            self.scores,
            self.lam,
            self.gam,
            self.fisher_norm,
            self.ratio,
            self.importance,
        ):
            if os.path.exists(path):
                os.remove(path)

    def log(self, file_name: str, **kv) -> None:
        if not self.enabled:
            return
        line = json.dumps(kv, ensure_ascii=False)
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.to_stdout:
            print(line, flush=True)


def build_svd_logger():
    if "grad-kfac" in os.environ.get("SVD_LOG_FILE", ""):
        return SVDLogger()
    return NullSVDLogger()


SVD_LOGGER = build_svd_logger()


@torch.no_grad()
def _check_reconstruction(
    parent_module,
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float,
) -> Tuple[float, float]:
    W = parent_module.base_layer.weight.data.to(torch.float32)
    delta = B @ A
    if getattr(parent_module, "fan_in_fan_out", False):
        delta = delta.t()
    delta = (scaling * delta).to(W.dtype)

    W_minus = (W - delta).to(torch.float32)
    W_recon = (W_minus + delta).to(torch.float32)
    err = (W_recon - W).abs()
    return float(err.max().cpu()), float(err.mean().cpu())


def _tokenize_sft(
    tokenizer: AutoTokenizer,
    instruction: str,
    response: Optional[str],
    max_len: int,
    add_eos: bool = True,
) -> Dict[str, torch.Tensor]:
    source = PROMPT.format_map({"instruction": instruction})
    target = (response or "").rstrip()
    if add_eos and tokenizer.eos_token is not None:
        target = target + tokenizer.eos_token

    tok_full = tokenizer(
        source + target,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    tok_src = tokenizer(
        source,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = tok_full["input_ids"][0]
    attention_mask = tok_full["attention_mask"][0]
    labels = input_ids.clone()

    src_len = int(tok_src["attention_mask"][0].sum().item())
    labels[:src_len] = IGNORE_INDEX
    labels[attention_mask == 0] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _pick_first(record: Dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in record and record[key] is not None:
            value = record[key]
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
            return str(value)
    return None


class SFTDataset(Dataset):
    QUESTION_KEYS = ["query", "question", "problem", "instruction", "prompt", "input"]
    ANSWER_KEYS = ["response", "output", "answer", "completion"]

    def __init__(self, tokenizer: AutoTokenizer, hf_path: str, hf_split: str, max_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len

        if ".json" in hf_path:
            dataset = load_dataset("json", data_files=hf_path, split=hf_split)
        else:
            dataset = load_dataset(hf_path, split="train")
            dataset = dataset.shuffle(seed=99991).select(range(256))

        if "conversation" in dataset[0]:
            dataset = dataset.map(
                lambda x: {
                    "instruction": x["conversation"][0]["human"],
                    "response": x["conversation"][0]["assistant"],
                }
            )

        self.items = []
        for example in dataset:
            question = _pick_first(example, self.QUESTION_KEYS)
            answer = _pick_first(example, self.ANSWER_KEYS)
            if question is None:
                question = str(example)
            self.items.append(_tokenize_sft(tokenizer, question, answer, max_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.items[index]


def build_loader_from_hf(
    tokenizer: AutoTokenizer,
    hf_path: str,
    hf_split: str,
    batch_size: int,
    max_len: int,
) -> DataLoader:
    dataset = SFTDataset(tokenizer, hf_path, hf_split, max_len)
    if len(dataset) == 0:
        raise RuntimeError(f"HF dataset is empty: {hf_path} {hf_split}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        drop_last=False,
    )


@torch.no_grad()
def _svd_w(
    weight: torch.Tensor,
    layer_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.to(torch.float32)
    U, S, Vh = torch.linalg.svd(weight_fp32, full_matrices=False)

    layer_tag = layer_name or "<unnamed>"
    logger.info(
        "[SVD] layer=%s, U=%s, S=%s, Vh=%s",
        layer_tag,
        tuple(U.shape),
        tuple(S.shape),
        tuple(Vh.shape),
    )

    topk = min(3, S.numel())
    if topk > 0:
        head = ", ".join(f"{x:.4e}" for x in S[:topk].tolist())
        tail = ", ".join(f"{x:.4e}" for x in S[-topk:].tolist())
        logger.info("[SVD] %s head[%d]: [%s] tail[%d]: [%s]", layer_tag, topk, head, topk, tail)

    return U, S, Vh


@torch.no_grad()
def _build_ab_from_indices(
    U: torch.Tensor,
    S: torch.Tensor,
    Vh: torch.Tensor,
    idx: torch.Tensor,
    *,
    scaling: float = 1.0,
    scale_compensation: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    S_sel = S[idx].clone()
    if scale_compensation and scaling not in (0.0, 1.0):
        S_sel = S_sel / scaling

    sqrt_s = torch.sqrt(S_sel)
    diag_s = torch.diag(sqrt_s)
    B = U[:, idx] @ diag_s
    A = diag_s @ Vh[idx, :]
    return A.contiguous(), B.contiguous()


def _get_peft_scaling(module, default_alpha: int, default_rank: int) -> float:
    try:
        if hasattr(module, "scaling"):
            scaling = module.scaling
            if isinstance(scaling, dict):
                return float(scaling.get("default", default_alpha / default_rank))
            return float(scaling)

        if hasattr(module, "lora_alpha") and hasattr(module, "r"):
            alpha, rank = module.lora_alpha, module.r
            if isinstance(alpha, dict) and isinstance(rank, dict):
                return float(alpha.get("default", default_alpha)) / float(rank.get("default", default_rank))
            return float(alpha) / float(rank)
    except Exception:
        pass
    return float(default_alpha) / float(default_rank)


@torch.no_grad()
def initialize_lora_layer(
    W: torch.Tensor,
    rank: int,
    method: str = "solar",
    grad: Optional[torch.Tensor] = None,
    lora_alpha: Optional[int] = None,
    layer_name: Optional[str] = None,
    cov_h: Optional[torch.Tensor] = None,
    cov_delta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    m, n = W.shape
    r = max(1, min(rank, m, n))
    if lora_alpha is None:
        lora_alpha = r

    eff_rank = int(r)
    grad_norm = None
    singular_values_log = None
    selected_idx_log = None
    lam_log = None
    gam_log = None
    score_log = None
    fisher_norm = None

    if grad is not None:
        try:
            grad_norm = float(grad.float().pow(2).sum().sqrt().cpu())
        except Exception:
            grad_norm = None

    U, S, Vh = _svd_w(W, layer_name=layer_name)
    singular_values_log = S.detach().cpu()

    if cov_h is None or cov_delta is None:
        raise ValueError("[grad-kfac] cov_h & cov_delta are required for this method.")

    device = U.device
    cov_h32 = cov_h.to(device=device, dtype=torch.float32)
    cov_d32 = cov_delta.to(device=device, dtype=torch.float32)

    cov_h32 = 0.5 * (cov_h32 + cov_h32.t())
    cov_d32 = 0.5 * (cov_d32 + cov_d32.t())

    V = Vh.transpose(0, 1)
    U_m = U

    tmp_h = cov_h32 @ V
    lam = (V * tmp_h).sum(dim=0)

    tmp_d = cov_d32 @ U_m
    gam = (U_m * tmp_d).sum(dim=0)

    eps = 1e-20
    lam = lam.clamp_min(0.0)
    gam = gam.clamp_min(0.0)

    S2 = S.detach().to(torch.float32) ** 2
    score = S2 * lam * gam

    lam_log = lam.detach().cpu()
    gam_log = gam.detach().cpu()
    score_log = score.detach().cpu()

    try:
        norm_cov_h = torch.linalg.norm(cov_h32, ord="fro")
        norm_cov_d = torch.linalg.norm(cov_d32, ord="fro")
        fisher_norm = float((norm_cov_h * norm_cov_d).cpu())
    except Exception:
        fisher_norm = None

    if torch.all(score <= eps) or torch.isnan(score).all():
        raise ValueError("[grad-kfac] all scores ~ 0, cannot select rank based on K-FAC scores.")

    k_eff = min(r, score.numel())
    if r < 128:
        _, idx = torch.topk(score, k=k_eff, largest=False, sorted=False)
    else:
        _, idx = torch.topk(score, k=k_eff, largest=True, sorted=False)
    idx, _ = torch.sort(idx)
    selected_idx_log = idx.detach().cpu()

    A, B = _build_ab_from_indices(
        U,
        S,
        Vh,
        idx,
        scaling=1.0,
        scale_compensation=True,
    )

    if layer_name is not None and method == "grad-kfac":
        payload = {"layer": layer_name, "method": method, "rank": int(eff_rank)}
        if singular_values_log is not None:
            payload["singular_values"] = [float(x) for x in singular_values_log.tolist()]
        if selected_idx_log is not None:
            payload["selected_idx"] = [int(i) for i in selected_idx_log.tolist()]
        if grad_norm is not None:
            payload["grad_norm"] = float(grad_norm)
        SVD_LOGGER.log(SVD_LOGGER.singular_values, **payload)

        payload = {"layer": layer_name, "method": method, "rank": int(eff_rank)}
        if lam_log is not None:
            payload["lam"] = [float(x) for x in lam_log.tolist()]
            payload["selected_idx"] = [int(i) for i in selected_idx_log.tolist()]
        SVD_LOGGER.log(SVD_LOGGER.lam, **payload)

        payload = {"layer": layer_name, "method": method, "rank": int(eff_rank)}
        if gam_log is not None:
            payload["gam"] = [float(x) for x in gam_log.tolist()]
            payload["selected_idx"] = [int(i) for i in selected_idx_log.tolist()]
        SVD_LOGGER.log(SVD_LOGGER.gam, **payload)

        payload = {"layer": layer_name, "method": method, "rank": int(eff_rank)}
        if score_log is not None:
            payload["score"] = [float(x) for x in score_log.tolist()]
            payload["selected_idx"] = [int(i) for i in selected_idx_log.tolist()]
        SVD_LOGGER.log(SVD_LOGGER.scores, **payload)

        payload = {"layer": layer_name, "method": method, "rank": int(eff_rank)}
        if fisher_norm is not None:
            payload["fisher_norm_fro_kfac"] = float(fisher_norm)
        SVD_LOGGER.log(SVD_LOGGER.fisher_norm, **payload)

    return A.to(torch.float32), B.to(torch.float32), eff_rank


def is_target_weight(param_name: str, target_modules: List[str]) -> bool:
    for module_name in target_modules:
        if param_name.endswith(f".{module_name}.weight") or param_name.endswith(f".{module_name}.base_layer.weight"):
            return True
    return False


def make_grad_hook(name: str, grad_bucket: Dict[str, torch.Tensor]):
    def _hook(grad: torch.Tensor):
        with torch.no_grad():
            grad_cpu = grad.detach().to(torch.float16).to("cpu", non_blocking=True)
            if name in grad_bucket:
                grad_bucket[name].add_(grad_cpu.float())
            else:
                grad_bucket[name] = grad_cpu.float().clone()

    return _hook


def build_module_rank_map(layer_rank_map: Dict[str, int]) -> Dict[str, int]:
    return {f"^{param_name}": int(rank) for param_name, rank in layer_rank_map.items()}


@contextmanager
def temp_enable_base_grads(model: torch.nn.Module, target_modules: List[str]):
    toggled = []
    for name, param in model.named_parameters():
        if is_target_weight(name, target_modules) and not param.requires_grad:
            param.requires_grad_(True)
            toggled.append(param)
    try:
        yield
    finally:
        for param in toggled:
            param.requires_grad_(False)


def make_kfac_forward_hook(layer_name: str, kfac_stats: Dict[str, Dict], max_samples: int = 1024):
    def _hook(module, inputs, outputs):
        if not inputs:
            return
        x = inputs[0]
        if isinstance(x, (tuple, list)):
            x = x[0]
        if x is None or not torch.is_tensor(x):
            return

        x_flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32)
        if max_samples > 0 and x_flat.size(0) > max_samples:
            idx = torch.randperm(x_flat.size(0), device=x_flat.device)[:max_samples]
            x_flat = x_flat[idx]

        stat = kfac_stats.setdefault(layer_name, {})
        if "sum_hhT" not in stat:
            d_in = x_flat.size(1)
            stat["sum_hhT"] = x_flat.new_zeros(d_in, d_in)
            stat["count_h"] = 0
        stat["sum_hhT"] += x_flat.t().matmul(x_flat)
        stat["count_h"] += x_flat.size(0)

    return _hook


def make_kfac_backward_hook(layer_name: str, kfac_stats: Dict[str, Dict], max_samples: int = 1024):
    def _hook(module, grad_input, grad_output):
        if not grad_output or grad_output[0] is None:
            return
        gy = grad_output[0]
        if isinstance(gy, (tuple, list)):
            gy = gy[0]
        if gy is None or not torch.is_tensor(gy):
            return

        gy_flat = gy.detach().reshape(-1, gy.shape[-1]).to(torch.float32)
        if max_samples > 0 and gy_flat.size(0) > max_samples:
            idx = torch.randperm(gy_flat.size(0), device=gy_flat.device)[:max_samples]
            gy_flat = gy_flat[idx]

        stat = kfac_stats.setdefault(layer_name, {})
        if "sum_ddT" not in stat:
            d_out = gy_flat.size(1)
            stat["sum_ddT"] = gy_flat.new_zeros(d_out, d_out)
            stat["count_d"] = 0
        stat["sum_ddT"] += gy_flat.t().matmul(gy_flat)
        stat["count_d"] += gy_flat.size(0)

    return _hook


def estimate_gradient(
    model: torch.nn.Module,
    dataloader: DataLoader,
    target_modules: List[str],
    device: Optional[torch.device] = None,
    amp_dtype: Optional[str] = "bf16",
    max_grad_steps: int = 8,
    collect_kfac: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
    if device is None:
        device = next(model.parameters()).device

    model.train()
    named_grads: Dict[str, torch.Tensor] = {}
    hooks = []

    kfac_stats: Dict[str, Dict[str, torch.Tensor]] = {}
    kfac_hooks = []
    module_dict = {module_name: module for module_name, module in model.named_modules()}

    for name, param in model.named_parameters():
        if param.requires_grad and is_target_weight(name, target_modules):
            hooks.append(param.register_hook(make_grad_hook(name, named_grads)))

            if collect_kfac:
                if ".base_layer.weight" in name:
                    parent_name = name.split(".base_layer.weight")[0]
                else:
                    parent_name = name.rsplit(".weight", 1)[0]

                if parent_name in module_dict and parent_name not in kfac_stats:
                    kfac_stats[parent_name] = {}
                    module = module_dict[parent_name]
                    fwd_hook = module.register_forward_hook(make_kfac_forward_hook(parent_name, kfac_stats))
                    bwd_hook = module.register_full_backward_hook(make_kfac_backward_hook(parent_name, kfac_stats))
                    kfac_hooks.extend([fwd_hook, bwd_hook])

    try:
        num_batches = len(dataloader)
    except Exception:
        num_batches = -1

    logger.info("[GradEst] batches=%s, amp=%s, max_grad_steps=%s", num_batches, amp_dtype, max_grad_steps)

    if amp_dtype == "bf16":
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    elif amp_dtype == "fp16":
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.float16)
    else:
        amp_ctx = nullcontext()

    steps = 0
    with amp_ctx:
        for batch in dataloader:
            steps += 1
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            if steps >= max_grad_steps:
                break

    if steps == 0:
        raise RuntimeError("No steps were run; dataloader empty?")

    inv_steps = 1.0 / float(steps)
    for name in list(named_grads.keys()):
        named_grads[name].mul_(inv_steps)
        named_grads[name] = named_grads[name].cpu()

    for layer_name, stat in kfac_stats.items():
        if "sum_hhT" in stat and stat.get("count_h", 0) > 0:
            stat["cov_h"] = stat["sum_hhT"] / float(stat["count_h"])
        if "sum_ddT" in stat and stat.get("count_d", 0) > 0:
            stat["cov_delta"] = stat["sum_ddT"] / float(stat["count_d"])

    clean_grads = {}
    for name, grad in named_grads.items():
        if ".weight" in name:
            clean_name = name.rsplit(".weight", 1)[0]
            clean_grads[clean_name] = grad
        else:
            clean_grads[name] = grad

    for hook in hooks:
        hook.remove()
    for hook in kfac_hooks:
        hook.remove()

    torch.cuda.empty_cache()
    return clean_grads, kfac_stats


def _extract_layer_idx_from_name(param_name: str) -> int:
    match = re.search(r"layers\.(\d+)", param_name)
    if match:
        return int(match.group(1))
    return -1


def _fisher_importance_for_weight(
    W: torch.Tensor,
    cov_h: Optional[torch.Tensor],
    cov_delta: Optional[torch.Tensor],
) -> Tuple[float, float]:
    W32 = W.detach().to(torch.float32)
    m, n = W32.shape
    fallback = float(W32.pow(2).mean().cpu())

    if cov_h is None or cov_delta is None:
        return fallback, fallback

    cov_h32 = cov_h.to(dtype=torch.float32, device=W32.device)
    cov_d32 = cov_delta.to(dtype=torch.float32, device=W32.device)

    diag_h = torch.diag(cov_h32)
    diag_d = torch.diag(cov_d32)

    if diag_h.numel() == 0 or diag_d.numel() == 0:
        score1 = fallback
    else:
        fisher_diag = diag_d.unsqueeze(1) * diag_h.unsqueeze(0)
        if fisher_diag.shape != W32.shape:
            if fisher_diag.t().shape == W32.shape:
                fisher_diag = fisher_diag.t()
            else:
                fisher_diag = torch.ones_like(W32) * float(diag_d.mean() * diag_h.mean())
        score1 = float((fisher_diag * (W32 ** 2)).mean().cpu())

    cov_h_sym = 0.5 * (cov_h32 + cov_h32.t())
    cov_d_sym = 0.5 * (cov_d32 + cov_d32.t())

    if cov_h_sym.shape != (n, n) or cov_d_sym.shape != (m, m):
        score2 = fallback
    else:
        val = torch.trace((cov_d_sym @ W32 @ cov_h_sym) @ W32.t())
        denom = float(m * n) if (m * n) > 0 else 1.0
        score2 = float((val / denom).cpu())

    return score1, score2


def allocate_layer_ranks_from_fisher(
    targets: List[Tuple[str, torch.nn.Parameter]],
    kfac_stats: Dict[str, Dict[str, torch.Tensor]],
    base_rank: int,
    min_rank: int,
    max_rank: int,
    avg_rank: int,
    start_layer: int,
) -> Dict[str, int]:
    name_to_layer_idx: Dict[str, int] = {}
    layer_to_names: Dict[int, List[str]] = {}

    for name, _ in targets:
        idx = _extract_layer_idx_from_name(name)
        name_to_layer_idx[name] = idx
        if idx >= 0:
            layer_to_names.setdefault(idx, []).append(name)

    unique_layers = sorted(layer_to_names.keys())
    if not unique_layers:
        return {name: base_rank for name, _ in targets}

    if start_layer is not None and start_layer >= 0:
        early_layer_set = {idx for idx in unique_layers if 0 <= idx <= start_layer}
    else:
        early_layers_count = max(1, len(unique_layers) // 6)
        early_layer_set = set(unique_layers[:early_layers_count])

    early_param_names = [
        name for name, _ in targets if name_to_layer_idx.get(name, -1) in early_layer_set
    ]
    if not early_param_names:
        return {name: base_rank for name, _ in targets}

    importance_map: Dict[str, float] = {}
    for name, param in targets:
        if name not in early_param_names:
            continue

        parent_name = name.split(".base_layer.weight")[0] if ".base_layer.weight" in name else name.rsplit(".weight", 1)[0]
        stat = kfac_stats.get(parent_name, {})
        _, score2 = _fisher_importance_for_weight(
            W=param.data,
            cov_h=stat.get("cov_h"),
            cov_delta=stat.get("cov_delta"),
        )
        importance_map[name] = score2

    imp_list = np.array([importance_map[name] for name in early_param_names], dtype=np.float64)
    eps = 1e-12
    total_imp = float(imp_list.sum())
    if not np.isfinite(total_imp) or total_imp <= eps:
        advantages = np.ones_like(imp_list) / float(len(imp_list))
    else:
        advantages = imp_list / total_imp

    inv = 1.0 / (advantages + eps)
    inv_sum = float(inv.sum())
    if not np.isfinite(inv_sum) or inv_sum <= eps:
        inv = np.ones_like(inv)
        inv_sum = float(inv.sum())

    early_max_rank = min(max_rank, base_rank)
    early_avg_rank = min(avg_rank, early_max_rank)

    total_rank_budget = early_avg_rank * len(early_param_names)
    scale = total_rank_budget / inv_sum
    raw_ranks = scale * inv

    int_ranks = np.round(raw_ranks).astype(np.int64)
    int_ranks = np.clip(int_ranks, min_rank, early_max_rank)

    rank_map = {name: base_rank for name, _ in targets}
    for name, rank in zip(early_param_names, int_ranks):
        rank_map[name] = int(rank)

    for name, rank in zip(early_param_names, int_ranks):
        advantage = (
            float(importance_map[name]) / (total_imp + eps)
            if total_imp > 0
            else 1.0 / len(early_param_names)
        )
        logger.info(
            "[RankAllocFisher] param=%s, I=%.4e, a=%.4e, rank=%s",
            name,
            importance_map[name],
            advantage,
            rank,
        )

    return rank_map


def _select_target_modules(task_name: str) -> List[str]:
    if task_name == "commonsense":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_tokenizer(model_path: str) -> AutoTokenizer:
    use_fast = not ("qwen" in model_path.lower() or "mistral" in model_path.lower())
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_model(model_path: str, model_dtype: str):
    torch_dtype = torch.bfloat16 if model_dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    return model, torch_dtype


def svd_tailor_and_save(args) -> None:
    method = args.init_method.lower()
    svd_rank = int(args.svd_rank)
    model_path = args.model_path
    target_modules = _select_target_modules(args.task_name)

    tokenizer = _load_tokenizer(model_path)
    save_path = os.path.join(args.save_path, f"{args.task_name}-{method}-r{svd_rank}")
    os.makedirs(os.path.join(save_path, "lora"), exist_ok=True)

    lora_rank = svd_rank
    lora_alpha = args.lora_alpha
    lora_dropout = 0.05

    model, model_dtype = _load_model(model_path, args.model_dtype)

    require_grad_methods = {"solar", "lora-ga", "lora-one", "grad-kfac", "corda_score"}
    named_grads: Dict[str, torch.Tensor] = {}
    kfac_stats: Dict[str, Dict[str, torch.Tensor]] = {}

    if method in require_grad_methods:
        logger.info("[Init] %s requires gradient estimation. Preparing data...", method)
        if args.dataset_path:
            init_dataset = torch.load(args.dataset_path)
            if len(init_dataset) == 0:
                raise RuntimeError(f"Local dataset is empty: {args.dataset_path}")
            init_loader = DataLoader(
                init_dataset,
                batch_size=args.init_bsz,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
        else:
            init_loader = build_loader_from_hf(
                tokenizer=tokenizer,
                hf_path=args.hf_dataset,
                hf_split=args.hf_split,
                batch_size=args.init_bsz,
                max_len=args.max_len,
            )

        with temp_enable_base_grads(model, target_modules):
            named_grads, kfac_stats = estimate_gradient(
                model=model,
                dataloader=init_loader,
                target_modules=target_modules,
                amp_dtype=args.amp_dtype,
                max_grad_steps=args.max_grad_steps,
                collect_kfac=method in {"grad-kfac", "corda_score"},
            )
        logger.info("named_grads keys: %s", list(named_grads.keys()))
        logger.info("kfac_stats keys: %s", list(kfac_stats.keys()))
    else:
        logger.info("[Init] %s does not require gradients. Skipping data loading and gradient estimation.", method)

    start_all = time.time()
    needs_subtract_global = method in {"grad-kfac"}

    logger.info(
        "[Init] method=%s, rank=%s, alpha=%s, dropout=%s, targets=%s, model_dtype=%s",
        method,
        svd_rank,
        lora_alpha,
        lora_dropout,
        target_modules,
        model_dtype,
    )

    targets = []
    for name, param in model.named_parameters():
        if is_target_weight(name, target_modules):
            if ".weight" in name:
                name = name.rsplit(".weight", 1)[0]
            targets.append((name, param))

    logger.info("[Init] target layers = %s", len(targets))

    if args.layers_split:
        if method == "grad-kfac" and len(kfac_stats) > 0:
            layer_rank_map = allocate_layer_ranks_from_fisher(
                targets=targets,
                kfac_stats=kfac_stats,
                base_rank=lora_rank,
                min_rank=args.min_rank,
                max_rank=args.max_rank,
                avg_rank=args.avg_rank,
                start_layer=args.layer_split_start,
            )
            module_rank_map = build_module_rank_map(layer_rank_map)
        else:
            layer_rank_map = {name: lora_rank for name, _ in targets}
            module_rank_map = None
    else:
        layer_rank_map = {name: lora_rank for name, _ in targets}
        module_rank_map = None

    logger.info("layer_rank_map: %s", layer_rank_map)
    logger.info("module_rank_map: %s", module_rank_map)

    if args.init_method == "grad-kfac" and args.layers_split:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            rank_pattern=module_rank_map,
        )
    else:
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

    model = get_peft_model(model, peft_config)

    with torch.no_grad():
        for index, (name, param) in enumerate(targets, start=1):
            layer_start = time.time()
            grad = named_grads.get(name)

            if grad is None and method in {"solar", "lora-ga", "lora-one", "grad-kfac"}:
                logger.warning("[%s] gradient not found for %s, skip this layer. (%s/%s)", method, name, index, len(targets))
                continue

            parent_name = "base_model.model." + name
            parent_module = model.get_submodule(parent_name)

            cov_h = None
            cov_delta = None
            if method in {"grad-kfac", "corda_score"}:
                stat = kfac_stats.get(name)
                if stat is not None:
                    cov_h = stat.get("cov_h")
                    cov_delta = stat.get("cov_delta")
                    if cov_h is None or cov_delta is None:
                        logger.warning("[grad-kfac] missing cov_h/cov_delta for %s", parent_name)

            if grad is not None and getattr(parent_module, "fan_in_fan_out", False):
                grad = grad.t().contiguous()

            eff_rank = layer_rank_map.get(name, lora_rank)
            logger.info(
                "[Init][%s/%s] layer=%s W.shape=%s method=%s, eff_rank=%s, global_rank=%s",
                index,
                len(targets),
                parent_name,
                tuple(param.data.shape),
                method,
                eff_rank,
                lora_rank,
            )

            A_init, B_init, eff_rank = initialize_lora_layer(
                W=param.data,
                rank=eff_rank,
                method=method,
                grad=grad,
                lora_alpha=lora_alpha,
                layer_name=parent_name,
                cov_h=cov_h,
                cov_delta=cov_delta,
            )

            scaling = _get_peft_scaling(parent_module, lora_alpha, lora_rank)

            dtype_w = parent_module.base_layer.weight.dtype
            delta = B_init @ A_init
            if getattr(parent_module, "fan_in_fan_out", False):
                delta = delta.t()
            delta = (scaling * delta).to(dtype_w)

            if needs_subtract_global:
                err_max, err_mean = _check_reconstruction(parent_module, A_init, B_init, scaling)
            else:
                err_max = float(delta.abs().max().cpu())
                err_mean = float(delta.abs().mean().cpu())

            if needs_subtract_global:
                parent_module.base_layer.weight.data -= delta

            parent_module.lora_A["default"].weight.data = A_init.to(torch.float32)
            parent_module.lora_B["default"].weight.data = B_init.to(torch.float32)

            logger.info(
                "[Init][%s/%s] layer=%s finished in %.3fs (recon_err_max=%.3e, mean=%.3e)",
                index,
                len(targets),
                parent_name,
                time.time() - layer_start,
                err_max,
                err_mean,
            )

    logger.info("[Save] base -> %s ; adapter -> %s", save_path, os.path.join(save_path, "lora"))
    model.save_pretrained(os.path.join(save_path, "lora"), safe_serialization=False)

    base_only = None
    if hasattr(model, "unload") and callable(getattr(model, "unload")):
        try:
            base_only = model.unload()
        except Exception as exc:
            logger.warning("[Save] model.unload() failed: %r", exc)

    if base_only is None and hasattr(model, "get_base_model"):
        try:
            base_only = model.get_base_model()
        except Exception as exc:
            logger.warning("[Save] model.get_base_model() failed: %r", exc)

    if base_only is None and hasattr(model, "base_model"):
        base_only = model.base_model

    if base_only is None:
        raise RuntimeError("Failed to obtain the pure base model (unload/get_base_model/base_model are all unavailable). Please check the PEFT version.")

    base_only.save_pretrained(save_path, safe_serialization=False)
    tokenizer.save_pretrained(save_path)

    meta = {
        "method": method,
        "svd_rank": svd_rank,
        "lora_alpha": lora_alpha,
        "needs_subtract_delta": needs_subtract_global,
        "target_modules": target_modules,
        "model_dtype": str(model_dtype),
        "save_time_ms": int((time.time() - start_all) * 1000),
    }
    with open(os.path.join(save_path, "init_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("[Done] initialization finished.")


def build_argparser():
    parser = argparse.ArgumentParser(description="SVD/gradient-based initializer for the LoRA family")
    parser.add_argument("--svd_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument(
        "--init_method",
        type=str,
        default="grad-kfac",
    )

    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--save_path", type=str, default="./svd_init_models")
    parser.add_argument("--model_dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--init_bsz", type=int, default=2)
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="meta-math/MetaMathQA",
        help="Can be either a Hugging Face repository name or a local root directory (e.g. /data/.../pissa-dataset).",
    )
    parser.add_argument("--hf_split", type=str, default="train[:1000]")
    parser.add_argument("--dataset_path", type=str, default=None, help="If provided, use a preprocessed local tensor dataset and skip Hugging Face loading.")
    parser.add_argument("--max_len", type=int, default=512)

    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "none"])
    parser.add_argument("--task_name", type=str, default="math", help="Used to name the output directory and logs.")
    parser.add_argument("--max_grad_steps", type=int, default=8, help="Maximum number of batches used for gradient estimation.")

    parser.add_argument("--min_rank", type=int, default=1, help="Minimum effective rank per layer.")
    parser.add_argument("--max_rank", type=int, default=64, help="Maximum effective rank per layer.")
    parser.add_argument("--avg_rank", type=int, default=50, help="Target average rank under layer-wise splitting.")
    parser.add_argument("--layers_split", type=int, default=1, help="Whether to enable layer-wise rank splitting.")
    parser.add_argument(
        "--layer_split_start",
        type=int,
        default=0,
        help="Starting decoder block index for layer-wise/adaptive rank allocation.",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    svd_tailor_and_save(args)