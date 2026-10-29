import os
import re
import time
import json
import math
import shutil
import torch
from pathlib import Path
from statistics import mean, median, pstdev
from collections import deque
from transformers import TrainerCallback
from huggingface_hub import HfApi
from datasets import load_dataset, Dataset
from typing import Any, Dict, List, Optional
from varrs import REASON_START_T, REASON_END_T, SOL_START_T, SOL_END_T
from custom_constraints_verifier.eval_if import verify_single_response

# Global metrics infrastructure
TOKENIZER_FOR_METRICS = None
_GROUP_COUNTER = 0
_GROUP_BUFFER: dict[str, list[int]] = {}
_METRICS_DIR = None
_GROUP_LOG_PATH = None
_STEP_LOG_PATH = None
_EXPECTED_G = 8
NUM_GENERATIONS = 8
SHARED_GROUPS: deque[list[int]] = deque()


def initialize_metrics(output_dir: str, num_generations: int = 8):
    """Initialize metrics directory and global variables."""
    global _METRICS_DIR, _GROUP_LOG_PATH, _STEP_LOG_PATH, _EXPECTED_G, NUM_GENERATIONS

    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _METRICS_DIR = metrics_dir
    _GROUP_LOG_PATH = str(_METRICS_DIR / "group_generations.jsonl")
    _STEP_LOG_PATH = str(_METRICS_DIR / "train_steps.jsonl")
    _EXPECTED_G = num_generations
    NUM_GENERATIONS = num_generations
    print(f"Metrics will be logged to {_METRICS_DIR}")


def _is_rank0() -> bool:
    """Check if current process is rank 0."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return True


def setup_pkgs(args):
    print("Setting up optimizer and DeepSpeed configuration...")
    # Resolve optimizer
    if args.optim == "auto":
        try:
            import bitsandbytes as bnb  # noqa: F401

            resolved_optim = "adamw_bnb_8bit"
            print("Using 8-bit optimizer for training (bitsandbytes detected).")
        except Exception:
            resolved_optim = "adamw_torch_fused"
            print("Using float32 optimizer for training (bitsandbytes not found).")
    else:
        resolved_optim = args.optim

    # Resolve deepspeed config (only if available and not 'none')
    ds_cfg = None
    if args.deepspeed_config and args.deepspeed_config.lower() != "none":
        try:
            import deepspeed as _ds  # noqa: F401

            ds_cfg = args.deepspeed_config
        except Exception:
            ds_cfg = None

    # Optional Flash-RL rollout patching for longer sequences / lower memory
    if args.flashrl_config and args.flashrl_config != "off":
        os.environ["FLASHRL_CONFIG"] = args.flashrl_config
        try:
            import flash_rl  # noqa: F401
        except Exception:
            # Best-effort: continue without Flash-RL if not available
            print("Warning: flash_rl not available; continuing without Flash-RL.")
            pass

    return resolved_optim, ds_cfg


def _log_jsonl(path: str, record: dict) -> None:
    """Append a JSON record to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_metrics_files() -> None:
    """Truncate metrics files at the start of a run."""
    if _GROUP_LOG_PATH:
        with open(_GROUP_LOG_PATH, "w", encoding="utf-8"):
            pass
    if _STEP_LOG_PATH:
        with open(_STEP_LOG_PATH, "w", encoding="utf-8"):
            pass


def _percentile(sorted_values: list[int], q: float) -> float:
    """Calculate percentile from sorted values."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _gini(values: list[int]) -> float:
    """Calculate Gini coefficient for distribution fairness."""
    n = len(values)
    if n == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumvals = 0
    cumarea = 0
    for i, v in enumerate(sorted_vals, start=1):
        cumvals += v
        cumarea += cumvals
    if cumvals == 0:
        return 0.0
    fair_area = n * cumvals / 2
    return (fair_area - (cumarea - cumvals / 2)) / fair_area


def _jain_index(values: list[int]) -> float:
    """Calculate Jain's fairness index."""
    if not values:
        return 1.0
    s1 = sum(values)
    s2 = sum(v * v for v in values)
    if s2 == 0:
        return 1.0
    n = len(values)
    return (s1 * s1) / (n * s2)


def _log_group_length_metrics_for_lengths(
    token_lengths: list[int], text_arr: list, reward_dict: Optional[dict] = {}
) -> None:
    """Log detailed metrics for a group of token lengths."""
    global _GROUP_COUNTER
    if _GROUP_LOG_PATH is None:
        return
    if not token_lengths:
        return
    sorted_lens = sorted(token_lengths)
    l_min = sorted_lens[0]
    l_max = sorted_lens[-1]
    l_mean = float(mean(token_lengths))
    l_median = float(median(token_lengths))
    l_std = float(pstdev(token_lengths)) if len(token_lengths) > 1 else 0.0
    p90 = _percentile(sorted_lens, 0.90)
    p95 = _percentile(sorted_lens, 0.95)
    p99 = _percentile(sorted_lens, 0.99)
    padding_waste_tokens = int(sum(l_max - li for li in token_lengths))
    padding_waste_pct = float(
        padding_waste_tokens / (len(token_lengths) * max(l_max, 1))
    )
    straggler_gap = float(l_max - l_median)
    straggler_ratio = float(l_max / max(l_median, 1.0))
    gini = _gini(token_lengths)
    jain = _jain_index(token_lengths)

    record = {
        "group_index": _GROUP_COUNTER,
        "g": len(token_lengths),
        "token_lengths": token_lengths,
        "min_tokens": int(l_min),
        "max_tokens": int(l_max),
        "mean_tokens": l_mean,
        "median_tokens": l_median,
        "std_tokens": l_std,
        "p90_tokens": p90,
        "p95_tokens": p95,
        "p99_tokens": p99,
        "padding_waste_tokens": padding_waste_tokens,
        "padding_waste_pct": padding_waste_pct,
        "straggler_gap_tokens": straggler_gap,
        "straggler_ratio": straggler_ratio,
        "gini_length": gini,
        "jain_fairness": jain,
        "ts": time.time(),
        "texts": text_arr,
    }
    _log_jsonl(_GROUP_LOG_PATH, record | reward_dict)
    _GROUP_COUNTER += 1


def _sanitize(component: str) -> str:
    """Sanitize a string component for use in repository names."""
    safe = []
    for ch in component:
        if ch.isalnum() or ch in {".", "-"}:
            safe.append(ch)
        else:
            safe.append("-")
    return re.sub(r"-{2,}", "-", "".join(safe)).strip("-")


def _build_repo_name(
    model_name_for_repo: str, dataset: str, iteration_number: str
) -> str:
    """Build a repository name from model, dataset, and iteration number."""
    return f"{_sanitize(model_name_for_repo)}_{_sanitize(dataset)}_{_sanitize(str(iteration_number))}"


def extract_xml_answer(text: str) -> str:
    """Extract the content from the first <answer> tag."""
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except IndexError:
        return ""


def custom_prompt_suffix(dataset):
    custom_prompt_suffix = f"""You are an expert on reasoning and you solve math questions by thinking step by step. Provide reasoning and a final answer.
    First think about the reasoning process and then provide the user with the answer.
    Place your reasoning between {REASON_START_T} and {REASON_END_T}. Then, provide your solution between {SOL_START_T}{SOL_END_T}"""
    dataset = dataset.map(
        lambda x: {
            "prompt": [
                # The system message sets the context.
                {"role": "system", "content": custom_prompt_suffix},
                # The actual question from the CSV.
                {"role": "user", "content": x["prompt"]},
            ],
            "answer": x["answer"],
        }
    )
    return dataset


def get_dapo_math_dataset() -> Dataset:
    # Load and format the dataset for mathematical reasoning
    dataset_name = "open-r1/DAPO-Math-17k-Processed"
    dataset = load_dataset(dataset_name, "en", split="train")
    dataset = dataset.rename_column("solution", "answer")
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["prompt", "answer"]]
    )

    dataset = custom_prompt_suffix(dataset)
    # Save processed dataset to disk for inspection/reuse
    try:
        save_path = "data/DAPO-Math-17k-Processed"
        dataset.save_to_disk(save_path)
        print(f"Saved processed dataset to {save_path}")
    except Exception as e:
        print(f"Warning: Failed to save dataset to disk: {e}")
    return dataset


COT_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)


def get_grpo_math_constraints_dataset(
    data_path: str = "data/train/math_with_constraints.json",
    no_constraint: bool = False,
) -> Dataset:
    """Load and format the math-with-constraints dataset for GRPO training.

    Reads from a local JSON with keys: question, answer, constraint_name,
    constraint_args, and constraint_desc. Builds prompts in the format:
      user_content = question + cot + (" ".join(constraint_desc) if not no_constraint else "")

    Returns a Dataset with columns: prompt, answer, question, constraint_desc,
    constraint_args, constraint_name, suitable for GRPOTrainer with
    reward_input_keys=["question", "answer", "constraint_desc", "constraint_args", "constraint_name"].
    """
    # Load the dataset file WITHOUT modifying it on disk.
    #
    # Some versions of this repo's dataset are "Python-literal-like" (e.g., using `None`)
    # rather than strict JSON (`null`). `json.load` will fail on those.
    #
    # We also avoid `load_dataset("json", ...)` because pyarrow schema inference can fail
    # when `constraint_args` changes shape across rows.
    import ast

    with open(data_path, "r", encoding="utf-8") as f:
        raw = f.read()

    try:
        data = json.loads(raw)
    except Exception:
        data = ast.literal_eval(raw)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {data_path}, got {type(data)}")

    # Normalize shapes so downstream code and HF datasets have consistent types
    normalized: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        constraint_desc = item.get("constraint_desc") or []
        constraint_name = item.get("constraint_name") or []
        constraint_args = item.get("constraint_args") or []

        # Ensure list-typed fields
        if not isinstance(constraint_desc, list):
            constraint_desc = [constraint_desc]
        if not isinstance(constraint_name, list):
            constraint_name = [constraint_name]
        if not isinstance(constraint_args, list):
            constraint_args = [constraint_args]

        # Pad args list to match number of constraints (common when args contains None entries)
        if len(constraint_args) < len(constraint_name):
            constraint_args = constraint_args + [None] * (len(constraint_name) - len(constraint_args))

        normalized.append(
            {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "constraint_desc": constraint_desc,
                "constraint_name": constraint_name,
                "constraint_args": constraint_args,
            }
        )

    dataset = Dataset.from_list(normalized)

    def _format_row(x):
        cot = COT_INSTRUCTION
        constraint_suffix = (
            "" if no_constraint else " ".join(x["constraint_desc"])
        )
        user_content = x["question"] + cot + constraint_suffix
        system_content = (
            "You are an expert on reasoning and you solve math questions by "
            "thinking step by step. You must do both: (1) reason carefully step by step, "
            "and (2) follow all instructions given in the user message. Provide reasoning and a final answer. "
            f"First think about the reasoning process, then provide the answer. "
            f"Place your reasoning between <think> and </think>. "
            f"Then provide your solution between {SOL_START_T}{SOL_END_T}"
        )
        return {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "answer": x["answer"],
            "question": x["question"],
            "constraint_desc": x["constraint_desc"],
            "constraint_args": x["constraint_args"],
            "constraint_name": x["constraint_name"],
        }

    dataset = dataset.map(_format_row)
    return dataset


def set_generation_config(trainer, tokenizer, args):
    """Sets generation config for model and ref_model if they exist."""
    try:
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        models_to_set = [unwrapped]
        if hasattr(trainer, "ref_model") and trainer.ref_model is not None:
            models_to_set.append(trainer.accelerator.unwrap_model(trainer.ref_model))
        for m in models_to_set:
            if hasattr(m, "generation_config"):
                gen_cfg = m.generation_config
                # Early stop via EOS if available
                if (
                    hasattr(tokenizer, "eos_token_id")
                    and tokenizer.eos_token_id is not None
                ):
                    gen_cfg.eos_token_id = tokenizer.eos_token_id
                # Sampling diversity
                gen_cfg.do_sample = True
                gen_cfg.temperature = 0.8
                gen_cfg.top_p = 0.95
                # Ensure pad_token_id is set to avoid warnings
                if args.use_vllm:
                    gen_cfg.all_random = True
                    gen_cfg.pad_token_id = tokenizer.eos_token_id
                    gen_cfg.bos_token_id = tokenizer.bos_token_id
            else:
                print(
                    "Warning: model does not have generation_config attribute; skipping generation config setup."
                )
    except Exception:
        print(
            "Warning: could not set generation config parameters for reference model."
        )


class AsyncPushToHubCallback(TrainerCallback):
    """Callback to push model checkpoints to Hugging Face Hub asynchronously.
    
    Uploads happen in background threads so training continues uninterrupted.
    Local checkpoints are deleted after successful upload to save disk space.
    
    Naming convention:
    - math_constraints dataset: {model}_reasoning_instruction_G{group_size}_{iteration}
    - dapo dataset (vanilla):   {model}_reasoning_G{group_size}_{iteration}
    """

    def __init__(
        self,
        base_model: str,
        dataset: str,
        hf_token: str,
        hf_username: str,
        num_generations: int = 8,
        max_uploads: int = 4,
        private: bool = True,
        delete_after_upload: bool = True,
    ):
        import threading
        import concurrent.futures
        
        self.base_model = base_model
        self.dataset = dataset
        self.hf_token = hf_token
        self.hf_username = hf_username
        self.num_generations = num_generations
        self.max_uploads = max_uploads
        self.private = private
        self.delete_after_upload = delete_after_upload
        
        self.uploads_done = 0
        self.uploads_queued = 0
        self.iteration_counter = 0  # Tracks 1, 2, 3, 4...
        self.api = None
        
        # Thread pool for async uploads
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._pending_futures: list[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        
        # Track output directory for final cleanup
        self._output_dir = None

    def _ensure_api(self):
        if self.api is None:
            self.api = HfApi(token=self.hf_token)

    def _build_repo_name(self, iteration: int, is_final: bool = False) -> str:
        """Build repository name based on naming convention.
        
        Format:
        - math_constraints: {model}_reasoning_instruction_G{group_size}_{iteration}
        - dapo (vanilla):   {model}_reasoning_G{group_size}_{iteration}
        """
        # Extract model name (last part of path, e.g., "Llama-3.2-3B-Instruct")
        model_name = self.base_model.split("/")[-1]
        
        # Determine suffix based on dataset
        if self.dataset == "math_constraints":
            suffix = "reasoning_instruction"
        else:
            # dapo or any other dataset = vanilla GRPO
            suffix = "reasoning"
        
        # Build: {model}_{suffix}_G{group_size}_{iteration}
        repo_name = f"{model_name}_{suffix}_G{self.num_generations}_{iteration}"
        
        # Sanitize for HuggingFace repo naming (replace problematic chars)
        repo_name = re.sub(r'[^a-zA-Z0-9._-]', '-', repo_name)
        repo_name = re.sub(r'-{2,}', '-', repo_name).strip('-')
        
        return repo_name

    def _upload_checkpoint(self, checkpoint_dir: str, repo_id: str, iteration: int, step: str) -> bool:
        """Upload a checkpoint to HuggingFace (runs in background thread)."""
        try:
            self._ensure_api()
            
            # Create repo if needed
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=self.private,
            )

            # Determine training type for README
            training_type = "GRPO with instruction constraints" if self.dataset == "math_constraints" else "Vanilla GRPO"
            
            # Create README
            readme_text = f"""---
license: apache-2.0
base_model: {self.base_model}
datasets:
  - {self.dataset}
library_name: transformers
tags:
  - grpo
  - reinforcement-learning
  - reasoning
---

# {repo_id.split('/')[-1]}

This repository contains checkpoint #{iteration} trained with **{training_type}** starting from `{self.base_model}`.

- **Training step**: {step}
- **Group size (G)**: {self.num_generations}
- **Dataset**: {self.dataset}

## Contents
- Model weights (`.safetensors`)
- Config files (`config.json`, `generation_config.json`)
- Tokenizer files

Training artifacts (optimizer/scheduler states and RNG) have been intentionally excluded.
"""
            import io
            self.api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_in_repo="README.md",
                path_or_fileobj=io.BytesIO(readme_text.encode("utf-8")),
            )

            # Patterns to exclude optimizer states
            ignore_patterns = [
                "optimizer.pt",
                "optimizer.bin",
                "scheduler.pt",
                "rng_state.pth",
                "training_args.bin",
                "trainer_state.json",
                "**/optimizer.pt",
                "**/optimizer.bin",
                "**/scheduler.pt",
                "**/rng_state.pth",
                "global_step*",
            ]

            # Upload the checkpoint folder
            self.api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=checkpoint_dir,
                path_in_repo=".",
                ignore_patterns=ignore_patterns,
            )

            print(f"[AsyncPushToHubCallback] Successfully uploaded {checkpoint_dir} to https://huggingface.co/{repo_id}")
            
            # Delete local checkpoint after successful upload
            if self.delete_after_upload and os.path.isdir(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    print(f"[AsyncPushToHubCallback] Deleted local checkpoint: {checkpoint_dir}")
                except Exception as e:
                    print(f"[AsyncPushToHubCallback] Warning: Failed to delete {checkpoint_dir}: {e}")
            
            with self._lock:
                self.uploads_done += 1
            
            return True
            
        except Exception as e:
            print(f"[AsyncPushToHubCallback] Error uploading {checkpoint_dir}: {e}")
            return False

    def _upload_final_model(self, model_dir: str, repo_id: str, iteration: int) -> bool:
        """Upload the final model to HuggingFace (runs in background thread)."""
        try:
            self._ensure_api()
            
            # Create repo if needed
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=self.private,
            )

            # Determine training type for README
            training_type = "GRPO with instruction constraints" if self.dataset == "math_constraints" else "Vanilla GRPO"

            # Create README for final model
            readme_text = f"""---
license: apache-2.0
base_model: {self.base_model}
datasets:
  - {self.dataset}
library_name: transformers
tags:
  - grpo
  - reinforcement-learning
  - reasoning
---

# {repo_id.split('/')[-1]}

This is the **final model** (#{iteration}) trained with **{training_type}** starting from `{self.base_model}`.

- **Group size (G)**: {self.num_generations}
- **Dataset**: {self.dataset}

## Contents
- Model weights (`.safetensors`)
- Config files (`config.json`, `generation_config.json`)
- Tokenizer files

Training artifacts (optimizer/scheduler states and RNG) have been intentionally excluded.
"""
            import io
            self.api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_in_repo="README.md",
                path_or_fileobj=io.BytesIO(readme_text.encode("utf-8")),
            )

            # Patterns to exclude
            ignore_patterns = [
                "optimizer.pt",
                "optimizer.bin",
                "scheduler.pt",
                "rng_state.pth",
                "training_args.bin",
                "trainer_state.json",
                "**/optimizer.pt",
                "**/optimizer.bin",
                "**/scheduler.pt",
                "**/rng_state.pth",
                "global_step*",
                "checkpoint-*",  # Exclude any checkpoint subdirectories
            ]

            # Upload the model folder
            self.api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=model_dir,
                path_in_repo=".",
                ignore_patterns=ignore_patterns,
            )

            print(f"[AsyncPushToHubCallback] Successfully uploaded final model to https://huggingface.co/{repo_id}")
            
            # Delete local model directory after successful upload
            if self.delete_after_upload and os.path.isdir(model_dir):
                try:
                    shutil.rmtree(model_dir)
                    print(f"[AsyncPushToHubCallback] Deleted local model directory: {model_dir}")
                except Exception as e:
                    print(f"[AsyncPushToHubCallback] Warning: Failed to delete {model_dir}: {e}")
            
            return True
            
        except Exception as e:
            print(f"[AsyncPushToHubCallback] Error uploading final model {model_dir}: {e}")
            return False

    def on_save(self, args, state, control, **kwargs):
        """Triggered when trainer saves a checkpoint - queue async upload."""
        # Only on main process
        if not state.is_world_process_zero:
            return
        
        self._output_dir = args.output_dir
        step = str(state.global_step)
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
        
        if not os.path.isdir(checkpoint_dir):
            return

        # Check upload cap (reserve last slot for final model)
        with self._lock:
            if self.uploads_queued >= (self.max_uploads - 1):
                # Skip upload but still delete local checkpoint to save space
                if self.delete_after_upload:
                    try:
                        shutil.rmtree(checkpoint_dir)
                        print(f"[AsyncPushToHubCallback] Skipped upload (cap reached), deleted local: {checkpoint_dir}")
                    except Exception:
                        pass
                return
            self.uploads_queued += 1
            self.iteration_counter += 1
            current_iteration = self.iteration_counter

        # Build repo name using new convention
        repo_name = self._build_repo_name(current_iteration)
        repo_id = f"{self.hf_username}/{repo_name}"

        print(f"[AsyncPushToHubCallback] Queuing background upload #{current_iteration}: {checkpoint_dir} -> {repo_id}")
        
        # Submit upload task to thread pool
        future = self._executor.submit(self._upload_checkpoint, checkpoint_dir, repo_id, current_iteration, step)
        self._pending_futures.append(future)

    def on_train_end(self, args, state, control, **kwargs):
        """Triggered at end of training - upload final model and wait for all uploads."""
        import concurrent.futures
        
        # Only on main process
        if not state.is_world_process_zero:
            return
        
        output_dir = self._output_dir or args.output_dir
        
        # Final model gets the next iteration number
        with self._lock:
            self.iteration_counter += 1
            final_iteration = self.iteration_counter
        
        # Build repo name for final model using new convention
        repo_name = self._build_repo_name(final_iteration)
        repo_id = f"{self.hf_username}/{repo_name}"
        
        print(f"[AsyncPushToHubCallback] Queuing final model upload #{final_iteration}: {output_dir} -> {repo_id}")
        final_future = self._executor.submit(self._upload_final_model, output_dir, repo_id, final_iteration)
        self._pending_futures.append(final_future)
        
        # Wait for all pending uploads to complete
        print(f"[AsyncPushToHubCallback] Waiting for {len(self._pending_futures)} pending uploads to complete...")
        
        for future in concurrent.futures.as_completed(self._pending_futures):
            try:
                future.result()
            except Exception as e:
                print(f"[AsyncPushToHubCallback] Upload task failed: {e}")
        
        print(f"[AsyncPushToHubCallback] All uploads completed. Total successful: {self.uploads_done + 1}")
        
        # Shutdown executor
        self._executor.shutdown(wait=False)

    def wait_for_uploads(self, timeout: float = None):
        """Manually wait for all pending uploads (can be called externally if needed)."""
        import concurrent.futures
        
        done, not_done = concurrent.futures.wait(
            self._pending_futures, 
            timeout=timeout,
            return_when=concurrent.futures.ALL_COMPLETED
        )
        return len(done), len(not_done)


# Keep old name as alias for backwards compatibility
PushToHubOnSaveCallback = AsyncPushToHubCallback


class StepTimerCallback(TrainerCallback):
    """Callback to log training step timing."""

    def __init__(self):
        self.t0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        # Only main process logs
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return
        if self.t0 is None or _STEP_LOG_PATH is None:
            return
        dt = max(time.perf_counter() - self.t0, 0.0)
        rec = {
            "global_step": int(getattr(state, "global_step", -1)),
            "step_time_s": dt,
            "ts": time.time(),
        }
        _log_jsonl(_STEP_LOG_PATH, rec)


def accuracy_reward_func(completions: list[str], **kwargs) -> torch.Tensor:
    """Reward function that extracts the answer from the xml tags and compares it to the correct answer."""
    global NUM_GENERATIONS, TOKENIZER_FOR_METRICS, _GROUP_BUFFER, _EXPECTED_G, SHARED_GROUPS

    ground_truth_answers = kwargs["answer"]

    if isinstance(completions[0], list):
        for i in range(len(completions)):
            completions[i] = completions[i][0]["content"]

    prompts = kwargs.get("prompt") or kwargs.get("prompts")
    num_generations = 1

    # Infer G by comparing counts
    if len(ground_truth_answers) > 0:
        num_generations = max(1, len(completions) // len(ground_truth_answers))
    expanded_prompts = [p for p in prompts for _ in range(num_generations)]
    if isinstance(expanded_prompts[0][0], dict):  # patch for vllm
        expanded_prompts = [
            TOKENIZER_FOR_METRICS.apply_chat_template(p, tokenize=False)
            for p in expanded_prompts
        ]

    if len(expanded_prompts) == len(completions):
        text_arr = []
        gt_arr = []
        for i, (c, p) in enumerate(zip(completions, expanded_prompts)):
            if TOKENIZER_FOR_METRICS is not None:
                tl = len(TOKENIZER_FOR_METRICS.encode(c))
            else:
                tl = len(c)
            buf = _GROUP_BUFFER.setdefault(p, [])
            buf.append(tl)
            text_arr.append(c)

            # Identify ground truth for this completion
            gt = ground_truth_answers[i // num_generations]
            gt_arr.append(gt)

            if len(buf) >= _EXPECTED_G:
                # Capture exactly one completed group
                group_lengths = list(buf[:_EXPECTED_G])

                # Get completions and ground truths for this group
                group_completions = text_arr[:_EXPECTED_G]
                group_gts = gt_arr[:_EXPECTED_G]

                # Calculate rewards for the completions we have
                actual_answers = [extract_xml_answer(txt) for txt in group_completions]

                # Calculate accuracy per item
                accuracy_list = []
                for ans, truth in zip(actual_answers, group_gts):
                    accuracy_list.append(1.0 if ans == truth else 0.0)

                # Calculate format reward per item
                format_list = []
                for txt in group_completions:
                    format_list.append(format_reward_func([txt])[0].item())

                # Calculate finstruction reward per item 
                group_example_idx = (i - _EXPECTED_G + 1) // num_generations
                finstruction_list = []
                if all(k in kwargs for k in ("question", "constraint_desc", "constraint_args")):
                    group_kwargs = {
                        "question": [kwargs["question"][group_example_idx]],
                        "answer": [kwargs["answer"][group_example_idx]],
                        "constraint_desc": [kwargs["constraint_desc"][group_example_idx]],
                        "constraint_args": [kwargs["constraint_args"][group_example_idx]],
                    }
                    for txt in group_completions:
                        finstruction_list.append(
                            finstruction_reward_func([txt], **group_kwargs)[0].item()
                        )
                else:
                    for _ in group_completions:
                        finstruction_list.append(0.0)

                reward_dict = {
                    "expected_answers": group_gts,
                    "actual_answers": actual_answers,
                    "accuracy": (
                        sum(accuracy_list) / len(accuracy_list)
                        if accuracy_list
                        else 0.0
                    ),
                    "format": (
                        sum(format_list) / len(format_list) if format_list else 0.0
                    ),
                    "finstruction": (
                        sum(finstruction_list) / len(finstruction_list)
                        if finstruction_list
                        else 0.0
                    ),
                    "accuracy_list": accuracy_list,
                    "format_list": format_list,
                    "finstruction_list": finstruction_list,
                }

                _log_group_length_metrics_for_lengths(
                    group_lengths,
                    text_arr=group_completions,
                    reward_dict=reward_dict,
                )
                # Append to shared in-memory queue for dynamic G chooser
                try:
                    SHARED_GROUPS.append(group_lengths)
                except Exception:
                    pass
                del buf[:_EXPECTED_G]
                text_arr = []
                gt_arr = []

    num_generations = len(completions) // len(ground_truth_answers)
    expanded_ground_truth = [
        ans for ans in ground_truth_answers for _ in range(num_generations)
    ]

    extracted_responses = [extract_xml_answer(c) for c in completions]
    rewards = []
    for resp, truth in zip(extracted_responses, expanded_ground_truth):
        if resp == truth:
            rewards.append(2.0)
        else:
            rewards.append(0.0)

    return torch.tensor(rewards)


def format_reward_func(completions: list[str], **kwargs) -> torch.Tensor:
    """Reward function that checks if the completion has a specific format."""

    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    rewards = []
    for completion in completions:
        if re.match(pattern, completion, re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return torch.tensor(rewards)

def finstruction_reward_func(completions: list[str], **kwargs) -> torch.Tensor:
    """
    Reward function that uses verify_single_response.
    Passes data_dict with keys expected by verify_single_response:
    'constraint_name' (from batch 'constraint_desc'), 'constraint_args', 'question', 'answer'.
    """
    if not completions:
        return torch.tensor([], dtype=torch.float32)

    # Normalize vLLM-style list/dict completions to strings (same as accuracy_reward_func)
    if isinstance(completions[0], list):
        completions = [c[0]["content"] for c in completions]

    questions = kwargs.get("question")
    answers = kwargs.get("answer")
    constraint_descs = kwargs.get("constraint_desc")
    constraint_args = kwargs.get("constraint_args")
    if not answers or questions is None or constraint_descs is None or constraint_args is None:
        return torch.zeros(len(completions), dtype=torch.float32)

    # Infer num_generations
    num_generations = max(1, len(completions) // len(answers))

    rewards = []

    for i, response in enumerate(completions):
        example_idx = i // num_generations

        # verify_single_response expects 'constraint_name', not 'constraint_desc'
        data_dict = {
            "question": questions[example_idx],
            "answer": answers[example_idx],
            "constraint_desc": constraint_descs[example_idx],
            "constraint_args": constraint_args[example_idx],
        }

        try:
            result = verify_single_response(response, data_dict)
            strict_score = result["strict_score"]
            loose_score = result["loose_score"]

            # choose how to convert verifier output → scalar reward
            reward = strict_score + loose_score

        except Exception:
            # verifier failure should not crash training
            reward = 0.0

        rewards.append(reward)

    return torch.tensor(rewards, dtype=torch.float32)



def set_vllm_sampling(
    trainer,
    tokenizer,
    generation_kwargs: Dict[str, Any],
) -> None:
    """
    Configure vLLM SamplingParams for TRL's vLLM server mode.

    - Reuses the same generation_kwargs you pass to GRPOConfig
    - Ensures stop_token_ids exist (filled from the tokenizer if missing)
    - Applies to trainer.vllm_client if available; otherwise sets trainer.args.vllm_sampling_params
    """
    params = dict(generation_kwargs or {})

    # Normalize common fields
    # vLLM expects: temperature, top_p, max_tokens, stop (list[str]), stop_token_ids (list[int]), ignore_eos (bool)
    if "max_tokens" in params and params["max_tokens"] is not None:
        params["max_tokens"] = int(params["max_tokens"])

    # Normalize stop → list[str]
    if "stop" in params and params["stop"] is not None:
        if isinstance(params["stop"], (str, int)):
            params["stop"] = [params["stop"]]
        elif not isinstance(params["stop"], list):
            params["stop"] = []
    else:
        params["stop"] = []

    # Coerce ignore_eos to bool if it comes as a string
    if "ignore_eos" in params:
        v = params["ignore_eos"]
        if isinstance(v, str):
            params["ignore_eos"] = v.strip().lower() in {"1", "true", "yes"}

    # Ensure stop_token_ids exist; fill from tokenizer if missing
    stop_token_ids: List[int] = []
    if "stop_token_ids" in params and params["stop_token_ids"]:
        if isinstance(params["stop_token_ids"], list):
            stop_token_ids = [
                int(t) for t in params["stop_token_ids"] if isinstance(t, (int, float))
            ]
        elif isinstance(params["stop_token_ids"], (int, float)):
            stop_token_ids = [int(params["stop_token_ids"])]

    # Add EOS and common chat end tokens, de-duplicated
    try:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            stop_token_ids.append(int(tokenizer.eos_token_id))
    except Exception:
        pass
    for tok in ("<|im_end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                stop_token_ids.append(int(tid))
        except Exception:
            pass
    stop_token_ids = sorted({int(t) for t in stop_token_ids if isinstance(t, int)})

    if stop_token_ids:
        params["stop_token_ids"] = stop_token_ids

    # Apply to vLLM client (preferred), else attach to args so TRL can pick it up
    try:
        client = getattr(trainer, "vllm_client", None)
        if client is not None and hasattr(client, "update_sampling_params"):
            client.update_sampling_params(**params)
        else:
            setattr(trainer.args, "vllm_sampling_params", params)
    except Exception:
        try:
            setattr(trainer.args, "vllm_sampling_params", params)
        except Exception:
            print("Warning: could not set vLLM sampling params")


def build_stop_toks(tokenizer):
    """Build generation kwargs with stop strings and token ids."""

    # Build stop strings/ids once
    stop_token_ids = []
    stop_strings = []
    if getattr(tokenizer, "eos_token", None):
        stop_strings.append(tokenizer.eos_token)
        # Add a task-specific stop if applicable
        stop_strings.append("</answer>")

        if getattr(tokenizer, "eos_token_id", None) is not None:
            stop_token_ids.append(int(tokenizer.eos_token_id))
        for tok in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0:
                    stop_token_ids.append(int(tid))
            except Exception:
                pass
        stop_token_ids = sorted({int(t) for t in stop_token_ids})

    return stop_strings, stop_token_ids


class TimeTriggeredActions(TrainerCallback):
    def __init__(
        self,
        checkpoint_every_s=4 * 3600,
        eval_every_s=4 * 3600,
        clock_log_path=None,
        enable_eval=False,
        *,
        write_all_ranks: bool = False,
        per_rank_files: bool = True,
    ):
        import time

        self.ckpt_interval = int(checkpoint_every_s)
        self.eval_interval = int(eval_every_s)
        self.clock_log_path = clock_log_path
        self._last_ckpt_ts = None
        self._last_eval_ts = None
        self.enable_eval = enable_eval
        self._time = time
        # Logging policy
        self.write_all_ranks = bool(write_all_ranks)
        self.per_rank_files = bool(per_rank_files)
        self.t0 = None

    def _get_rank_info(self) -> tuple[int, int]:
        """Return (rank, world_size) if DDP is initialized; otherwise (0, 1)."""
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank()), int(dist.get_world_size())
        except Exception:
            pass
        # Fallback to env vars
        try:
            import os

            r = int(os.environ.get("RANK", "0"))
            ws = int(os.environ.get("WORLD_SIZE", "1"))
            return r, ws
        except Exception:
            return 0, 1

    def _resolve_clock_path(self, rank: int) -> Optional[str]:
        if not self.clock_log_path:
            return None
        if self.write_all_ranks and self.per_rank_files:
            import os

            base, ext = os.path.splitext(self.clock_log_path)
            return f"{base}.rank{rank}{ext or '.jsonl'}"
        return self.clock_log_path

    def _log_clock_mark(self, state):
        # Only log after a step has begun (for step timing)
        if self.t0 is None:
            return
        if not self.clock_log_path:
            return

        dt = max(self._time.perf_counter() - self.t0, 0.0)
        rank, world_size = self._get_rank_info()
        rec = {
            "ts": self._time.time(),
            "global_step": int(getattr(state, "global_step", -1)),
            "step_time_s": dt,
            "rank": int(rank),
            "world_size": int(world_size),
        }
        try:
            import json, os

            target_path = self._resolve_clock_path(rank)
            if target_path is None:
                return

            # If logging from all ranks, avoid contention with per-rank files (default)
            if self.write_all_ranks:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            else:
                # Rank-0 only
                flag = getattr(state, "is_world_process_zero", True)
                is_rank0 = flag() if callable(flag) else bool(flag)
                if is_rank0 or rank == 0:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
        except Exception as e:
            print(f"Error logging clock mark: {e}")

    def on_train_begin(self, args, state, control, **kwargs):
        now = self._time.time()
        # Initialize timers
        self._last_ckpt_ts = now
        self._last_eval_ts = now
        self._log_clock_mark(state)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.t0 = self._time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        # Runs once per update step (after grad accumulation)
        now = self._time.time()

        # Checkpoint trigger
        if self.ckpt_interval > 0 and (now - self._last_ckpt_ts) >= self.ckpt_interval:
            control.should_save = True
            self._last_ckpt_ts = now
            self._log_clock_mark(state)

        # Evaluation trigger
        if (
            self.eval_interval > 0
            and (now - self._last_eval_ts) >= self.eval_interval
            and self.enable_eval
        ):
            control.should_evaluate = True
            self._last_eval_ts = now
            self._log_clock_mark(state)

        return control
