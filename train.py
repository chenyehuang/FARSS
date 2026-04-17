import ast
import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainerCallback, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

try:
    import bitsandbytes as bnb  # noqa: F401

    HAS_BNB = True
except Exception:
    HAS_BNB = False


QUESTION_CANDIDATES = ["query", "question", "instruction", "prompt", "input", "context"]
ANSWER_CANDIDATES = ["response", "answer", "output", "completion", "target"]

IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

SUPPORTED_METHODS = {
    "full",
    "lora",
    "grad-kfac",
}

logger = logging.getLogger("train")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _rank0() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def _gpu_mem_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA: N/A"

    parts = []
    for idx in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(idx) / 1024**2
        reserv = torch.cuda.memory_reserved(idx) / 1024**2
        peak = torch.cuda.max_memory_allocated(idx) / 1024**2
        parts.append(f"GPU{idx}: alloc={alloc:.0f}MB, reserv={reserv:.0f}MB, max={peak:.0f}MB")
    return " | ".join(parts)


def _normalize_dataset_field(
    dataset_field,
    available_cols: Sequence[str],
) -> Tuple[str, str]:
    """
    Normalize args.dataset_field into (q_col, a_col).

    Supported inputs:
      - None
      - List[str] / Tuple[str]
      - JSON string
      - Python literal string
      - Comma-separated string
    """
    if isinstance(dataset_field, (list, tuple)):
        if len(dataset_field) == 2:
            q_col, a_col = dataset_field[0], dataset_field[1]
            if q_col in available_cols and a_col in available_cols:
                return q_col, a_col
        elif len(dataset_field) == 1:
            dataset_field = dataset_field[0]
        else:
            dataset_field = None

    if isinstance(dataset_field, str):
        parsed = False

        for parser in (json.loads, ast.literal_eval):
            try:
                value = parser(dataset_field)
                parsed = True
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    q_col, a_col = value[0], value[1]
                    if q_col in available_cols and a_col in available_cols:
                        return q_col, a_col
            except Exception:
                pass

        if (not parsed) or ("," in dataset_field and "[" not in dataset_field and "]" not in dataset_field):
            parts = [p.strip() for p in dataset_field.split(",")]
            if len(parts) == 2 and all(p in available_cols for p in parts):
                return parts[0], parts[1]

    q_auto = next((c for c in QUESTION_CANDIDATES if c in available_cols), None)
    a_auto = next((c for c in ANSWER_CANDIDATES if c in available_cols), None)
    if q_auto and a_auto:
        return q_auto, a_auto

    raise ValueError(
        f"[dataset_field] Unable to determine the instruction/answer column names. Available columns: {sorted(list(available_cols))}\n"
        f"Please specify the column names explicitly using one of the following formats:\n"
        f"  --dataset_field query response\n"
        f'  --dataset_field \'["query","response"]\'\n'
        f"  --dataset_field query,response\n"
    )


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, List[np.ndarray]]:
    tokenized_list = [
        tokenizer(text, max_length=tokenizer.model_max_length, truncation=True)
        for text in strings
    ]
    input_ids = [np.array(tokenized.input_ids) for tokenized in tokenized_list]
    input_ids_lens = [len(tokenized.input_ids) for tokenized in tokenized_list]

    return {
        "input_ids": input_ids,
        "labels": input_ids,
        "input_ids_lens": input_ids_lens,
        "labels_lens": input_ids_lens,
    }


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    mask_source: bool = True,
) -> Dict[str, List[np.ndarray]]:
    examples = [source + target for source, target in zip(sources, targets)]
    examples_tok, sources_tok = [_tokenize_fn(items, tokenizer) for items in (examples, sources)]

    input_ids = examples_tok["input_ids"]
    labels = copy.deepcopy(input_ids)

    if mask_source:
        for label, source_len in zip(labels, sources_tok["input_ids_lens"]):
            for idx in range(source_len):
                label[idx] = IGNORE_INDEX

    return {"input_ids": input_ids, "labels": labels}


def train_tokenize_function(
    examples,
    tokenizer,
    query: str,
    response: str,
):
    sources = [PROMPT.format_map({"instruction": instruction}) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    return preprocess(sources, targets, tokenizer)


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in instances]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    output_dir: Optional[str] = field(default="./output")

    data_path: str = field(default=None, metadata={"help": "Hugging Face dataset path (hub repository or loading script)."})
    dataset_split: str = field(default="train[:100000]")
    dataset_field: Optional[List[str]] = field(
        default=None,
        metadata={"help": "For example ['query','response'] or --dataset_field query response; if omitted, fields are matched automatically."},
    )
    shuffle_dataset: bool = field(default=False)

    full_finetune: bool = field(default=False, metadata={"help": "True = full fine-tuning; False = LoRA/adapter fine-tuning."})
    method_type: str = field(
        default="lora",
        metadata={"help": "full|lora|pissa|milora|lora-ga|lora-one|solar|grad-kfac|pissa-milora|corda_score"},
    )

    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Adapter directory produced by init.py; if provided, it takes precedence."},
    )
    target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    lora_r: Optional[int] = field(default=None)
    lora_alpha: Optional[int] = field(default=None)
    lora_dropout: float = field(default=0.0)

    optim: str = field(default="adamw_torch")
    merge: bool = field(default=False, metadata={"help": "Whether to merge LoRA weights for inference after training."})
    model_max_length: int = field(default=512)
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "For example 'flash_attention_2' or 'eager'."},
    )


class ForceLogEveryStep(TrainerCallback):
    """Force logging at every step and additionally print step/s and samples/s."""

    def __init__(self):
        self._last_time = None
        self._last_step = 0
        self._effective_batch_size = None

    def on_train_begin(self, args, state, control, **kwargs):
        import time

        self._last_time = time.time()
        self._last_step = 0

        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        per_device_bs = getattr(args, "per_device_train_batch_size", 1)
        grad_accum = getattr(args, "gradient_accumulation_steps", 1)
        self._effective_batch_size = world_size * per_device_bs * grad_accum

    def on_step_end(self, args, state, control, **kwargs):
        control.should_log = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        import time

        now = time.time()
        if self._last_time is None:
            self._last_time = now

        dt = now - self._last_time
        dstep = state.global_step - self._last_step
        steps_per_sec = (dstep / dt) if dt > 0 and dstep > 0 else float("nan")
        samples_per_sec = steps_per_sec * (self._effective_batch_size or 1)

        logs = logs or {}
        loss = logs.get("loss")
        learning_rate = logs.get("learning_rate")
        epoch = logs.get("epoch")
        grad_norm = logs.get("grad_norm")

        if grad_norm is None:
            trainer = kwargs.get("trainer", None)
            try:
                ds_engine = getattr(trainer, "deepspeed", None)
                if ds_engine is not None and hasattr(ds_engine, "get_global_grad_norm"):
                    grad_norm = float(ds_engine.get_global_grad_norm())
            except Exception:
                pass

        if int(os.environ.get("RANK", "0") or "0") == 0:
            print(
                f"[step {state.global_step}/{state.max_steps}] "
                f"{{'loss': {None if loss is None else round(float(loss), 6)}, "
                f"'grad_norm': {None if grad_norm is None else round(float(grad_norm), 6)}, "
                f"'learning_rate': {None if learning_rate is None else float(learning_rate)}, "
                f"'epoch': {None if epoch is None else round(float(epoch), 4)}}} | "
                f"{steps_per_sec:.2f} step/s, {samples_per_sec:.2f} samples/s",
                flush=True,
            )

        self._last_time = now
        self._last_step = state.global_step


class ProgressLoggerCallback(TrainerCallback):
    def __init__(self):
        self._effective_batch_size = None

    def on_train_begin(self, args, state, control, **kwargs):
        world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
        per_device_bs = getattr(args, "per_device_train_batch_size", 1)
        grad_accum = getattr(args, "gradient_accumulation_steps", 1)
        self._effective_batch_size = world_size * per_device_bs * grad_accum

        logger.info(
            "[Begin] epochs=%s max_steps=%s | world=%d per_device_bs=%d accum=%d -> effective_bs=%d",
            args.num_train_epochs,
            state.max_steps,
            world_size,
            per_device_bs,
            grad_accum,
            self._effective_batch_size,
        )


class SavePeftModelCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _save(self, args, state, model, tokenizer):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = state.best_model_checkpoint
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        os.makedirs(peft_model_path, exist_ok=True)
        model.save_pretrained(peft_model_path)
        (tokenizer or self.tokenizer).save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self._save(args, state, kwargs["model"], kwargs.get("tokenizer", self.tokenizer))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, "completed"))
        self._save(args, state, kwargs["model"], kwargs.get("tokenizer", self.tokenizer))


def get_last_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not os.path.isdir(checkpoint_dir):
        return None

    if os.path.exists(os.path.join(checkpoint_dir, "completed")):
        return None

    max_step = 0
    for filename in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, filename)
        if os.path.isdir(path) and filename.startswith(PREFIX_CHECKPOINT_DIR):
            try:
                step = int(filename.replace(PREFIX_CHECKPOINT_DIR + "-", ""))
                max_step = max(max_step, step)
            except Exception:
                pass

    if max_step == 0:
        return None
    return os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{max_step}")


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str) -> None:
    """Move weights to CPU before saving during full fine-tuning to avoid OOM."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def _maybe_find_existing_adapter_dir(model_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(model_dir, "lora"),
        os.path.join(model_dir, "adapter_model"),
        os.path.join(model_dir, "adapter_model", "adapter_model"),
        os.path.join(model_dir, "ft", "adapter_model"),
        os.path.join(model_dir, "ft", "lora"),
    ]

    for path in candidates:
        if os.path.isdir(path):
            logger.info("[AdapterFinder] found adapter at: %s", path)
            return path

    logger.info("[AdapterFinder] no adapter directory found under %s", model_dir)
    return None


def _quick_sanity_check_lora(model) -> None:
    with torch.no_grad():
        lora_layer_count = 0

        for _, module in model.named_modules():
            if hasattr(module, "base_layer") and hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
                lora_layer_count += 1

        logger.info("[SANITY] number of LoRA layers = %d (check only)", lora_layer_count)


def load_tokenizer_for(
    model_path: str,
    fallback_base: Optional[str] = None,
    model_max_len: Optional[int] = None,
):
    def pick_use_fast(config_or_none, path: str) -> bool:
        model_type = (getattr(config_or_none, "model_type", "") or "").lower()
        if (
            "qwen" in model_type
            or "mistral" in model_type
            or "qwen" in path.lower()
            or "mistral" in path.lower()
        ):
            return False
        return True

    config = None
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        pass

    use_fast = pick_use_fast(config, model_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=use_fast,
            trust_remote_code=True,
            model_max_length=model_max_len,
        )
    except Exception:
        if fallback_base is None:
            raise

        fallback_config = None
        try:
            fallback_config = AutoConfig.from_pretrained(fallback_base, trust_remote_code=True)
        except Exception:
            pass

        tokenizer = AutoTokenizer.from_pretrained(
            fallback_base,
            use_fast=pick_use_fast(fallback_config, fallback_base),
            trust_remote_code=True,
            model_max_length=model_max_len,
        )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return tokenizer


def build_model(args: TrainingArguments, tokenizer):
    if getattr(args, "bf16", False):
        compute_dtype = torch.bfloat16
    elif getattr(args, "fp16", False):
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if args.attn_implementation:
        try:
            config.attn_implementation = args.attn_implementation
        except Exception:
            pass

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    method = (args.method_type or "lora").lower()
    if _rank0():
        logger.info("[LOAD] base=%s", args.model_name_or_path)
        logger.info("[LOAD] adapter(flag)=%s", args.adapter_name_or_path)

    if args.full_finetune or method == "full":
        if _rank0():
            logger.info("[Full-FT] Full fine-tuning: no adapter will be loaded.")
    else:
        adapter_dir = None
        if args.adapter_name_or_path:
            if os.path.isdir(args.adapter_name_or_path):
                adapter_dir = args.adapter_name_or_path
            else:
                raise ValueError(
                    f"adapter_name_or_path does not exist: {args.adapter_name_or_path}; "
                    f"model_name_or_path={args.model_name_or_path}"
                )
        else:
            adapter_dir = _maybe_find_existing_adapter_dir(args.model_name_or_path)

        if method in {"pissa", "milora", "lora-one", "lora-ga", "solar", "grad-kfac", "pissa-milora", "corda_score"}:
            if adapter_dir is None:
                raise ValueError("No initialized adapter directory was found. Please provide --adapter_name_or_path or place an initialized adapter under the model directory.")

            if _rank0():
                logger.info(
                    "[%s] Loading initialized adapter: %s (ignoring --lora_r/--lora_alpha/--target_modules)",
                    method,
                    adapter_dir,
                )
            model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)

            try:
                from peft import PeftConfig

                adapter_config = PeftConfig.from_pretrained(adapter_dir)
                logger.info(
                    "[AdapterCfg] r=%s, alpha=%s, targets=%s",
                    getattr(adapter_config, "r", None),
                    getattr(adapter_config, "lora_alpha", None),
                    getattr(adapter_config, "target_modules", None),
                )
            except Exception as exc:
                logger.warning("[AdapterCfg] failed to read config: %s", exc)

            if any(x is not None for x in [args.lora_r, args.lora_alpha]) or (
                args.target_modules and args.target_modules.strip()
            ):
                logger.warning(
                    "[%s] An initialized adapter (%s) was loaded, so --lora_r/--lora_alpha/--target_modules will be ignored.",
                    method,
                    adapter_dir,
                )

        elif method == "lora":
            if adapter_dir is not None:
                if _rank0():
                    logger.info("[lora] Existing adapter found, loading directly: %s", adapter_dir)
                model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
            else:
                target_modules = (
                    args.target_modules.split(",")
                    if args.target_modules
                    else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                )

                if args.lora_r is None:
                    raise ValueError(
                        "[lora] Creating a new LoRA adapter requires --lora_r "
                        "(and optionally --lora_alpha, --lora_dropout, --target_modules).\n"
                        "For example:\n"
                        "  --method_type lora \\\n"
                        "  --lora_r 64 --lora_alpha 64 --lora_dropout 0.05 \\\n"
                        "  --target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
                    )

                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=(args.lora_r if args.lora_alpha is None else args.lora_alpha),
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    target_modules=target_modules,
                    task_type="CAUSAL_LM",
                )
                if _rank0():
                    logger.info(
                        "[lora] Creating a new LoRA adapter: r=%s, alpha=%s, dropout=%s, targets=%s",
                        lora_config.r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        target_modules,
                    )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
        else:
            raise ValueError(f"Unknown method_type: {method}")

    for name, module in model.named_modules():
        if ("norm" in name) or ("gate" in name):
            try:
                module.to(torch.float32)
            except Exception:
                pass

    model.config.use_cache = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    if getattr(args, "gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if _rank0():
        logger.info("[Model] dtype=%s | %s", next(model.parameters()).dtype, _gpu_mem_summary())

    try:
        _quick_sanity_check_lora(model)
    except Exception as exc:
        logger.warning("[SANITY] skipping check: %s", exc)

    return model


def train() -> None:
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    method = (args.method_type or "lora").lower()
    if method == "full":
        args.full_finetune = True
    elif method not in SUPPORTED_METHODS - {"full"}:
        raise ValueError(f"Unknown method_type: {method}")

    set_seed(getattr(args, "seed", 888))
    if _rank0():
        logger.info("[Seed] set to %s", getattr(args, "seed", 888))

    try:
        can_use_fused = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except Exception:
        can_use_fused = False

    if not hasattr(args, "optim") or args.optim in (None, "adamw_torch"):
        if os.environ.get("USE_BNB_OPT", "0") == "1" and HAS_BNB:
            args.optim = "adamw_bnb_8bit"
        elif can_use_fused:
            args.optim = "adamw_torch_fused"
        else:
            args.optim = "adamw_torch"

    if not hasattr(args, "group_by_length") or args.group_by_length is None:
        args.group_by_length = True

    if not hasattr(args, "tf32") or args.tf32 is None:
        args.tf32 = True

    if not args.full_finetune and (not hasattr(args, "gradient_checkpointing") or not args.gradient_checkpointing):
        args.gradient_checkpointing = True

    args.logging_strategy = "steps"
    args.logging_first_step = True
    if not getattr(args, "logging_steps", None) or args.logging_steps > 1:
        args.logging_steps = 1
    args.log_on_each_node = False

    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    per_device_bs = getattr(args, "per_device_train_batch_size", 1)
    grad_accum = getattr(args, "gradient_accumulation_steps", 1)
    effective_bs = world_size * per_device_bs * grad_accum

    if _rank0():
        logger.info(
            "[TrainCfg] world=%d per_device_bs=%d accum=%d effective_bs=%d",
            world_size,
            per_device_bs,
            grad_accum,
            effective_bs,
        )
        logger.info("[Deepspeed] config=%s", getattr(args, "deepspeed", None))

    tokenizer = load_tokenizer_for(
        args.model_name_or_path,
        fallback_base=None,
        model_max_len=args.model_max_length,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    resume_from_checkpoint_dir = get_last_checkpoint(args.output_dir)
    model = build_model(args, tokenizer)

    if getattr(model.config, "max_position_embeddings", None):
        tokenizer.model_max_length = min(
            tokenizer.model_max_length,
            model.config.max_position_embeddings,
        )

    if _rank0():
        logger.info(
            "[Tokenizer] model_max_length=%s pad=%s(%s) bos=%s eos=%s",
            getattr(tokenizer, "model_max_length", None),
            tokenizer.pad_token,
            tokenizer.pad_token_id,
            getattr(tokenizer, "bos_token", None),
            getattr(tokenizer, "eos_token", None),
        )
        logger.info(
            "[EFFECTIVE PATHS] base=%s | adapter=%s",
            args.model_name_or_path,
            args.adapter_name_or_path or _maybe_find_existing_adapter_dir(args.model_name_or_path),
        )

    if ".json" in args.data_path:
        raw_dataset = load_dataset("json", data_files=args.data_path, split=args.dataset_split)
    else:
        raw_dataset = load_dataset(args.data_path, split=args.dataset_split)

    if "conversation" in raw_dataset[0]:
        raw_dataset = raw_dataset.map(
            lambda x: {
                "instruction": x["conversation"][0]["human"],
                "response": x["conversation"][0]["assistant"],
            }
        )

    if _rank0():
        logger.info("[Dataset] path=%s split=%s rows=%d", args.data_path, args.dataset_split, len(raw_dataset))

    if args.shuffle_dataset:
        raw_dataset = raw_dataset.shuffle(seed=args.seed)
        if _rank0():
            logger.info("[Dataset] shuffled with seed=%s", args.seed)

    cols = list(raw_dataset.column_names)
    q_col, a_col = _normalize_dataset_field(args.dataset_field, cols)
    if _rank0():
        logger.info("[dataset_field] question='%s' answer='%s'; columns=%s", q_col, a_col, cols)

    if _rank0():
        for idx in range(min(2, len(raw_dataset))):
            try:
                logger.info("[Raw#%d] %s=%r", idx, q_col, raw_dataset[idx][q_col])
                logger.info("[Raw#%d] %s=%r", idx, a_col, raw_dataset[idx][a_col])
            except Exception as exc:
                logger.warning("[Raw preview failed #%d] %s", idx, exc)

    train_dataset = raw_dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=2048,
        num_proc=min(8, os.cpu_count() or 8),
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=True,
        desc="Tokenizing train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": q_col,
            "response": a_col,
        },
    )

    def has_supervised_token(example) -> bool:
        for token in example["labels"]:
            if token != IGNORE_INDEX:
                return True
        return False

    train_dataset = train_dataset.filter(
        has_supervised_token,
        num_proc=min(8, os.cpu_count() or 8),
        desc="Filtering samples with no supervised tokens",
    )

    if _rank0():
        logger.info("[Tokenized] train size=%d", len(train_dataset))
        try:
            sample_n = min(5000, len(train_dataset))
            seq_lens = np.fromiter(
                (len(train_dataset[i]["input_ids"]) for i in range(sample_n)),
                dtype=np.int32,
                count=sample_n,
            )
            logger.info(
                "[SeqLen] n=%d p50=%d p90=%d p95=%d p99=%d max=%d",
                sample_n,
                int(np.percentile(seq_lens, 50)),
                int(np.percentile(seq_lens, 90)),
                int(np.percentile(seq_lens, 95)),
                int(np.percentile(seq_lens, 99)),
                int(seq_lens.max()),
            )
            for idx in range(min(2, len(train_dataset))):
                input_ids = train_dataset[idx]["input_ids"]
                labels = train_dataset[idx]["labels"]

                ignored_prefix = 0
                for token in labels:
                    if token == IGNORE_INDEX:
                        ignored_prefix += 1
                    else:
                        break

                logger.info(
                    "[Tok#%d] len=%d | source_len=%d answer_len=%d | head_ids=%s",
                    idx,
                    len(input_ids),
                    ignored_prefix,
                    len(labels) - ignored_prefix,
                    input_ids[:24],
                )
        except Exception as exc:
            logger.warning("[SeqLen stats failed] %s", str(exc))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "callbacks": [],
    }

    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    trainer.add_callback(ProgressLoggerCallback())
    trainer.add_callback(ForceLogEveryStep())

    if method != "full" and not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback(tokenizer))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint_dir)
    trainer.save_state()

    if args.full_finetune or method == "full":
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)
    else:
        trainer.model.save_pretrained(os.path.join(args.output_dir, "ft"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "ft"))

    if _rank0():
        logger.info("[Done] training finished | %s", _gpu_mem_summary())


if __name__ == "__main__":
    train()