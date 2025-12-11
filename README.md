---
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- lora
- finance
- instruction-tuning
- english
- transformers
- adapter
---

# Llama for Finance (LoRA)

A financial-domain instruction-tuned LoRA adapter for `meta-llama/Meta-Llama-3.1-8B-Instruct`. Trained on a filtered subset of Finance-Instruct-500k with English-only enforcement and length-aware batching to reduce padding waste.
# Hugging Face link: https://huggingface.co/TimberGu/Llama_for_Finance

## Model Details
- **Base model:** meta-llama/Meta-Llama-3.1-8B-Instruct  
- **Adapter type:** LoRA (PEFT)  
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  
- **LoRA hyperparams:** r=64, alpha=128, dropout=0.1, bias=none  
- **Precision:** fp16 (bf16 if available) with gradient checkpointing  
- **Length bucketing:** enabled (`group_by_length=True`, custom bucket boundaries)  
- **Context length:** adaptively capped (up to 2048 in this run)  
- **Language:** English (non-English texts filtered via ASCII ratio heuristic)

## Training Data & Filtering
- **Source dataset:** `Josephgflowers/Finance-Instruct-500k`
- **Sampling caps:** 40k train / 4k validation (post-filtering counts may be lower)
- **Chat formatting:** `apply_chat_template` for system/user/assistant turns
- **Filters:**  
  - drop rows without user/assistant text  
  - truncate to max_length (adaptive)  
  - minimum length (≥30 tokens)  
  - English-only heuristic (configurable `filter_english_only`, `min_english_ratio`)

## Training Setup
- **Epochs:** 5  
- **Batching:** per-device batch 16, grad accumulation 4 (effective 64)  
- **Optimizer:** paged_adamw_8bit  
- **LR / schedule:** 1e-4, cosine, warmup_ratio 0.05  
- **Regularization:** weight_decay 0.01, max_grad_norm 1.0  
- **Eval/save:** eval_steps=50, save_steps=100, load_best_model_at_end=True  
- **Length-aware sampler:** custom bucket sampler to reduce padding

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
adapter = "TimberGu/Llama_for_Finance"

tokenizer = AutoTokenizer.from_pretrained(adapter)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
base_model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter)
model.eval()

prompt = "Explain what a yield curve inversion implies for equities."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, temperature=0.8, top_p=0.9)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Evaluation
- See `eval_50_gpt_judged_raw.jsonl` for the held-out validation metrics produced after training. (No public benchmark beyond the split provided in Finance-Instruct-500k.)

## Limitations & Risks
- Domain-focused on finance/economics; may underperform on general tasks.
- English-centric; non-English input was filtered during training.
- Hallucinations remain possible—do not use for financial advice without human review.

## Files
- `adapter_model.safetensors`, `adapter_config.json`: LoRA weights/config
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`
- `training_config.json`, `training_args.bin`, `test_results.json`
