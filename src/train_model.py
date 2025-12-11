"""
Model training script for financial LLM fine-tuning with QLoRA
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from functools import partial
from torch.utils.data import DataLoader, IterableDataset
try:
    from transformers.trainer_pt_utils import seed_worker
except ImportError:  # transformers>=4.44 removed trainer_pt_utils
    try:
        from transformers.trainer_utils import seed_worker  # type: ignore
    except ImportError:
        def seed_worker(worker_id: int, num_workers: int = 0, rank: int = 0) -> None:  # type: ignore
            """Fallback worker seeding when transformers doesn't expose helper."""
            worker_seed = torch.initial_seed() % (2 ** 32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)


def _bytes_to_gib(num_bytes: int) -> float:
    """Convert bytes to gibibytes."""
    return num_bytes / (1024 ** 3)


def _collect_parameter_stats(model):
    total_bytes = 0
    trainable_bytes = 0
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        param_bytes = param.numel() * param.element_size()
        total_bytes += param_bytes
        total_params += param.numel()
        if param.requires_grad:
            trainable_bytes += param_bytes
            trainable_params += param.numel()

    return {
        "total_bytes": total_bytes,
        "trainable_bytes": trainable_bytes,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _estimate_optimizer_bytes(trainable_param_bytes: int, config: Optional[Dict[str, Any]]) -> float:
    if trainable_param_bytes == 0:
        return 0.0

    optim_name = (config or {}).get("optim", "adamw").lower()
    if "4bit" in optim_name:
        factor = 0.5
    elif "8bit" in optim_name or "paged" in optim_name:
        factor = 1.0
    elif "lion" in optim_name:
        factor = 1.0
    else:
        factor = 2.0

    return trainable_param_bytes * factor


def _estimate_activation_bytes(model, config: Optional[Dict[str, Any]]) -> Optional[float]:
    if config is None:
        return None

    seq_len = config.get("max_length")
    batch_size = config.get("train_batch_size")
    if not seq_len or not batch_size:
        return None

    model_config = getattr(model, "config", None)
    hidden_size = getattr(model_config, "hidden_size", None)
    num_layers = getattr(model_config, "num_hidden_layers", None)
    if hidden_size is None or num_layers is None:
        return None

    dtype_bytes = 2  # bfloat16 / float16
    grad_ckpt = bool(config.get("gradient_checkpointing", True))
    activation_multiplier = 1.5 if grad_ckpt else 2.5

    return batch_size * seq_len * hidden_size * num_layers * dtype_bytes * activation_multiplier


def show_gpu_memory_status(model=None, config: Optional[Dict[str, Any]] = None):
    """Log the current GPU memory status and estimated training footprint."""
    if not torch.cuda.is_available():
        print("CUDA not available; skipping GPU memory status check.")
        return

    torch.cuda.empty_cache()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()

    seq_len = config.get("max_length") if config else None
    train_bs = config.get("train_batch_size") if config else None

    print("GPU memory status:")
    for idx in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        allocated_bytes = torch.cuda.memory_allocated(idx)
        reserved_bytes = torch.cuda.memory_reserved(idx)

        free_gib = _bytes_to_gib(free_bytes)
        total_gib = _bytes_to_gib(total_bytes)
        allocated_gib = _bytes_to_gib(allocated_bytes)
        reserved_gib = _bytes_to_gib(reserved_bytes)

        print(
            f"  [{idx}] {device_name}: "
            f"free {free_gib:.2f} GiB / total {total_gib:.2f} GiB | "
            f"allocated {allocated_gib:.2f} GiB | reserved {reserved_gib:.2f} GiB"
        )

    if model is None:
        return

    param_stats = _collect_parameter_stats(model)
    trainable_bytes = param_stats["trainable_bytes"]
    grad_bytes = trainable_bytes
    optim_bytes = _estimate_optimizer_bytes(trainable_bytes, config)
    activation_bytes = _estimate_activation_bytes(model, config)

    print("Estimated training memory breakdown (per device):")
    print(
        f"  Parameters (trainable): "
        f"{_bytes_to_gib(trainable_bytes):.2f} GiB "
        f"across {param_stats['trainable_params']:,} params"
    )
    print(f"  Gradients: {_bytes_to_gib(grad_bytes):.2f} GiB (mirrors trainable params)")
    print(f"  Optimizer state: {_bytes_to_gib(optim_bytes):.2f} GiB (heuristic)")

    if activation_bytes is None:
        print("  Activations: unavailable (missing model/config metadata)")
    else:
        seq_display = seq_len if activation_bytes else config.get("max_length")
        batch_display = (
            train_bs if activation_bytes else config.get("train_batch_size")
        )
        print(
            f"  Activations (~batch {batch_display} × seq {seq_display}): "
            f"{_bytes_to_gib(activation_bytes):.2f} GiB"
        )

    total_estimate = trainable_bytes + grad_bytes + optim_bytes + (activation_bytes or 0)
    print(f"  ----> Estimated total training footprint: {_bytes_to_gib(total_estimate):.2f} GiB")


def setup_model_and_tokenizer(config):
    """Setup model and tokenizer with optional quantization"""
    
    # Enable TF32 for faster computation
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    
    torch.cuda.empty_cache()
    
    quant_pref = config.get("quantization")
    quantization = quant_pref.lower() if isinstance(quant_pref, str) else None
    
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with QLoRA
    precision_pref = config.get("precision", "fp16")
    precision = precision_pref.lower() if isinstance(precision_pref, str) else "fp16"
    target_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    
    model_kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
    )
    
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = target_dtype

    # Attention implementation (stick with PyTorch SDPA unless explicitly overridden)
    attn_pref = config.get("attn_impl")
    chosen_attn = attn_pref or "sdpa"

    model_kwargs["attn_implementation"] = chosen_attn

    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        **model_kwargs,
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    use_gradient_checkpointing = bool(config.get('gradient_checkpointing', True))

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

    if use_gradient_checkpointing:
        # Ensure gradients flow to inputs for checkpointed layers
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


def setup_lora(model, config):
    """Setup LoRA for QLoRA fine-tuning"""
    
    # Determine target modules based on model architecture
    if "Llama" in config['model_name'] or "llama" in config['model_name']:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    elif "DialoGPT" in config['model_name']:
        target_modules = ["c_attn", "c_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]
    
    lora_r = int(config.get('lora_r', 64))
    lora_alpha = int(config.get('lora_alpha', 128))
    lora_dropout = float(config.get('lora_dropout', 0.1))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


@dataclass
class SFTDataCollator:
    """Dynamically pad input/label pairs to reduce padding waste."""

    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        input_lengths = [len(feature["input_ids"]) for feature in features]
        max_seq_len = max(input_lengths)
        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_seq_len = ((max_seq_len + multiple - 1) // multiple) * multiple

        input_ids = torch.full(
            (batch_size, max_seq_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_seq_len),
            fill_value=-100,
            dtype=torch.long,
        )

        for idx, feature in enumerate(features):
            seq = torch.tensor(feature["input_ids"], dtype=torch.long)
            seq_length = min(seq.size(0), max_seq_len)
            input_ids[idx, :seq_length] = seq[:seq_length]
            attention_mask[idx, :seq_length] = 1

            label_tensor = torch.tensor(feature["labels"], dtype=torch.long)
            labels[idx, :seq_length] = label_tensor[:seq_length]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return batch


class LengthBucketBatchSampler:
    """Batch sampler that groups indices by token-length buckets to reduce padding waste."""

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        boundaries: List[int],
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if not boundaries:
            raise ValueError("LengthBucketBatchSampler requires at least one boundary.")
        self.lengths = [int(x) for x in lengths]
        self.batch_size = batch_size
        self.boundaries = sorted(int(x) for x in boundaries)
        self.drop_last = drop_last
        self.seed = seed
        self._iter_calls = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

    def _bucket_key(self, length: int) -> int:
        for boundary in self.boundaries:
            if length <= boundary:
                return boundary
        return self.boundaries[-1]

    def __iter__(self):
        total = len(self.lengths)
        if self.seed is None:
            permutation = torch.randperm(total).tolist()
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self._iter_calls)
            permutation = torch.randperm(total, generator=generator).tolist()
        self._iter_calls += 1

        bucket_buffers: Dict[int, List[int]] = defaultdict(list)
        leftovers: List[int] = []

        for idx in permutation:
            length = self.lengths[idx]
            bucket_key = self._bucket_key(length)
            bucket_buffers[bucket_key].append(idx)
            if len(bucket_buffers[bucket_key]) == self.batch_size:
                yield list(bucket_buffers[bucket_key])
                bucket_buffers[bucket_key].clear()

        for bucket_indices in bucket_buffers.values():
            leftovers.extend(bucket_indices)

        if not self.drop_last and leftovers:
            for start in range(0, len(leftovers), self.batch_size):
                batch = leftovers[start:start + self.batch_size]
                if batch:
                    yield batch


class BucketedTrainer(Trainer):
    """Trainer subclass that swaps in a bucket-aware DataLoader for training."""

    def __init__(
        self,
        *args,
        bucket_lengths: Optional[List[int]] = None,
        bucket_boundaries: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        self.bucket_lengths = (
            [int(x) for x in bucket_lengths] if bucket_lengths is not None else None
        )
        self.bucket_boundaries = (
            sorted(int(x) for x in bucket_boundaries)
            if bucket_boundaries
            else None
        )
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        if (
            self.bucket_boundaries is None
            or self.bucket_lengths is None
            or isinstance(self.train_dataset, IterableDataset)
            or self.args.world_size > 1
            or len(self.bucket_lengths) != len(self.train_dataset)
        ):
            return super().get_train_dataloader()

        train_dataset = self._remove_unused_columns(
            self.train_dataset, description="training"
        )

        batch_sampler = LengthBucketBatchSampler(
            lengths=self.bucket_lengths,
            batch_size=self.args.per_device_train_batch_size,
            boundaries=self.bucket_boundaries,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
        )

        worker_init_fn = None
        if self.args.seed is not None:
            worker_init_fn = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )

        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.args.dataloader_persistent_workers,
        )


def tokenize_dataset(dataset, tokenizer, config):
    """Tokenize the dataset with SFT label masking (only compute loss on assistant response)."""
    
    # Get special tokens for Llama 3.1
    assistant_header_start = "<|start_header_id|>assistant<|end_header_id|>"
    force_eos = bool(config.get("force_eos_token", True))
    eos_token = tokenizer.eos_token
    filter_english_only = bool(config.get("filter_english_only", True))
    min_english_ratio = float(config.get("min_english_ratio", 0.85))
    min_chars_for_lang = int(config.get("min_chars_for_lang_check", 40))

    def _is_mostly_english(text: str) -> bool:
        """Heuristic English check: keep if ASCII-ish chars dominate."""
        if not filter_english_only:
            return True
        if not isinstance(text, str):
            return False
        stripped = text.strip()
        if len(stripped) < min_chars_for_lang:
            # Too short to judge reliably; keep to avoid over-filtering.
            return True
        total = len(stripped)
        ascii_like = sum(ch.isascii() for ch in stripped)
        return (ascii_like / max(total, 1)) >= min_english_ratio
    
    def tokenize_function(examples):
        """Tokenize texts and mask labels for SFT (only answer part)"""
        # Ensure samples end with an EOS token so the model learns to stop.
        processed_texts = []
        for raw_text in examples["text"]:
            text = raw_text.rstrip()
            if force_eos and eos_token and not text.endswith(eos_token):
                text = text + eos_token
            processed_texts.append(text)

        tokenized = tokenizer(
            processed_texts,
            truncation=True,
            padding=False,
            max_length=config['max_length'],
            return_tensors=None,
            add_special_tokens=True,
        )
        
        labels = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        lengths = []
        
        for text, input_ids in zip(processed_texts, tokenized["input_ids"]):
            label = input_ids.copy()
            
            # Find assistant response start in Llama 3.1 format
            assistant_start_pos = text.find(assistant_header_start)
            
            if assistant_start_pos != -1:
                # Tokenize prefix (everything before assistant response)
                prefix_text = text[:assistant_start_pos + len(assistant_header_start)]
                prefix_tokens = tokenizer(
                    prefix_text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                
                # Mask everything before assistant response
                prefix_len = min(len(prefix_tokens), len(label))
                for i in range(prefix_len):
                    label[i] = -100
            else:
                # Fallback: search for assistant token in tokenized sequence
                # Look for the pattern: start_header_id -> assistant -> end_header_id
                found_assistant = False
                for i in range(len(input_ids) - 5):
                    # Decode a window to find assistant header
                    window = input_ids[i:i+10]
                    decoded = tokenizer.decode(window, skip_special_tokens=False)
                    if "assistant" in decoded.lower() and "<|start_header_id|>" in decoded:
                        # Found assistant header, mask everything before content
                        # Content usually starts 3-4 tokens after header
                        content_start = i + 4
                        for j in range(min(content_start, len(label))):
                            label[j] = -100
                        found_assistant = True
                        break
                
                # If still not found, use heuristic: mask first portion
                if not found_assistant:
                    mask_length = len(label) // 3
                    for i in range(mask_length):
                        label[i] = -100
            
            # Mask padding tokens
            for i, token_id in enumerate(input_ids):
                if token_id == pad_token_id:
                    label[i] = -100
            
            labels.append(label)
            lengths.append(len(input_ids))
        
        tokenized["labels"] = labels
        tokenized["input_length"] = lengths
        return tokenized
    
    if filter_english_only:
        dataset = dataset.filter(
            lambda x: _is_mostly_english(x.get("text", "")),
            desc=f"Filtering non-English texts (ratio>={min_english_ratio:.2f})",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    
    return tokenized_dataset


def log_length_statistics(lengths: List[int], prefix: str = "Token length stats") -> Dict[str, int]:
    if not lengths:
        print(f"{prefix}: unavailable (no length data).")
        return {}

    array = np.array(lengths)
    percentiles = [50, 75, 90, 95, 98, 99, 100]
    stats = {f"p{p}": int(np.percentile(array, p)) for p in percentiles}
    print(prefix + ":")
    for key in ["p50", "p75", "p90", "p95", "p98", "p99", "p100"]:
        value = stats.get(key)
        if value is not None:
            label = key.upper()
            print(f"  {label}: {value} tokens")
    print(f"  Mean: {array.mean():.1f} tokens | Std: {array.std():.1f} tokens")
    return stats


def setup_training(model, tokenizer, tokenized_dataset, config, train_lengths: Optional[List[int]] = None):
    """Setup training arguments and trainer"""
    
    pad_to_multiple = 8 if torch.cuda.is_available() else None
    data_collator = SFTDataCollator(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple)
    
    use_eval_strategy = hasattr(TrainingArguments, '__dataclass_fields__') and \
                       'eval_strategy' in str(TrainingArguments.__dataclass_fields__)
    eval_param_name = "eval_strategy" if use_eval_strategy else "evaluation_strategy"
    
    training_args_dict = {
        "output_dir": config['output_dir'],
        "per_device_train_batch_size": config['train_batch_size'],
        "per_device_eval_batch_size": config['eval_batch_size'],
        "gradient_accumulation_steps": config['gradient_accumulation_steps'],
        "num_train_epochs": config['num_epochs'],
        "learning_rate": config['learning_rate'],
        "logging_steps": config.get('logging_steps', 25),
        eval_param_name: "steps",
        "eval_steps": config.get('eval_steps', 50),
        "save_steps": config.get('save_steps', config.get('eval_steps', 100)),
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "push_to_hub": False,
        "report_to": config.get('report_to', None),
        "load_best_model_at_end": True,
        "group_by_length": True,
        "warmup_ratio": config.get('warmup_ratio', 0.03),
        "weight_decay": config.get('weight_decay', 0.01),
        "max_grad_norm": config.get('max_grad_norm', 1.0),
        "lr_scheduler_type": "cosine",
        "dataloader_num_workers": config.get('dataloader_num_workers', 2),
        "dataloader_pin_memory": True,
        "skip_memory_metrics": True,
        "log_level": "warning",
        "include_inputs_for_metrics": False,
        "prediction_loss_only": True,
        "gradient_checkpointing": config.get('gradient_checkpointing', True),
        "optim": config.get('optim', "paged_adamw_8bit"),
    }

    label_smoothing = float(config.get("label_smoothing", 0.0) or 0.0)
    if label_smoothing > 0:
        training_args_dict["label_smoothing_factor"] = label_smoothing

    if config.get('align_save_with_eval', True):
        training_args_dict["save_steps"] = training_args_dict.get("eval_steps", training_args_dict.get("save_steps", 100))

    precision_pref = config.get("precision")
    precision_choice = precision_pref.lower() if isinstance(precision_pref, str) else None
    if precision_choice == "bf16":
        training_args_dict["bf16"] = True
        training_args_dict["fp16"] = False
    elif precision_choice == "fp16":
        training_args_dict["fp16"] = True
        training_args_dict["bf16"] = False
    else:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if use_bf16:
            training_args_dict["bf16"] = True
            training_args_dict["fp16"] = False
        else:
            training_args_dict["fp16"] = True
    
    training_args = TrainingArguments(**training_args_dict)
    
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    
    bucket_boundaries = config.get("length_bucket_boundaries")
    if bucket_boundaries and train_lengths:
        trainer = BucketedTrainer(
            **trainer_kwargs,
            bucket_lengths=train_lengths,
            bucket_boundaries=bucket_boundaries,
        )
    else:
        trainer = Trainer(**trainer_kwargs)
    
    return trainer


def save_model_and_config(model, tokenizer, trainer, config):
    """Save the trained model and configuration"""
    
    trainer.save_model(config['save_dir'])
    tokenizer.save_pretrained(config['save_dir'])
    
    config_data = {
        "base_model": config['model_name'],
        "dataset": config['dataset_name'],
        "training_config": config,
        "lora_config": {
            "r": config['lora_r'],
            "alpha": config['lora_alpha'],
            "dropout": config['lora_dropout']
        },
        "training_date": datetime.now().isoformat()
    }
    
    with open(f"{config['save_dir']}/training_config.json", "w") as f:
        json.dump(config_data, f, indent=2, default=str)
    
    test_results = trainer.evaluate()
    
    with open(f"{config['save_dir']}/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    return test_results


def run_training(config, processed_dataset):
    """Run the complete training pipeline with QLoRA"""
    
    model, tokenizer = setup_model_and_tokenizer(config)
    model = setup_lora(model, config)
    show_gpu_memory_status(model, config)
    tokenized_dataset = tokenize_dataset(processed_dataset, tokenizer, config)
    
    train_lengths = None
    train_split = tokenized_dataset["train"]
    if "input_length" in train_split.column_names:
        train_lengths = [int(x) for x in train_split["input_length"]]
    
    trainer = setup_training(
        model,
        tokenizer,
        tokenized_dataset,
        config,
        train_lengths=train_lengths,
    )

    if train_lengths:
        length_stats = log_length_statistics(
            train_lengths,
            prefix="Token length stats (processed train split)",
        )lora
        config["length_stats"] = length_stats
    
    trainer.train()
    test_results = save_model_and_config(model, tokenizer, trainer, config)
    
    return model, tokenizer, trainer


if __name__ == "__main__":
    print("train_model.py provides setup and run helpers for QLoRA fine-tuning.")
    print("Import run_training(config, processed_dataset) from another script or notebook.")