import argparse
import os
from collections import deque
import torch
import inspect
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from varrs import L_MAX, DELTA, E, PG_TARGET, LEN_P, ALLOWED_G
from custom_grpo_trainer import (
    UpdateGroupSizeCallback,
    control_step_pd,
)
from pd_g_select import LagrangianRiskController

# Import all utility functions and classes from utils
from utils import (
    initialize_metrics,
    reset_metrics_files,
    accuracy_reward_func,
    format_reward_func,
    finstruction_reward_func,
    StepTimerCallback,
    AsyncPushToHubCallback,
    get_dapo_math_dataset,
    get_grpo_math_constraints_dataset,
    setup_pkgs,
    set_generation_config,
    build_stop_toks,
    TimeTriggeredActions,
    TOKENIZER_FOR_METRICS,
    NUM_GENERATIONS,
    _GROUP_BUFFER,
    _EXPECTED_G,
    SHARED_GROUPS,
)
import utils as utils  # For accessing and modifying global variables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flashrl-config",
        type=str,
        default=os.environ.get("FLASHRL_CONFIG", "off"),
        choices=["off", "bf16", "fp8"],
        help="Enable Flash-RL rollout patching (bf16/fp8). Uses env FLASHRL_CONFIG if set.",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="none",
        help="Path to DeepSpeed config JSON for ZeRO CPU offload.",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="auto",
        help="Optimizer: auto (bnb 8bit if available), adamw_bnb_8bit, adamw_torch_fused",
    )

    parser.add_argument(
        "--vllm-mode",
        type=str,
        choices=["server", "colocate"],
        default=None,
        help="Use vLLM for batched generation during rollout.",
    )

    parser.add_argument(
        "--vllm-gpu-memory",
        type=float,
        default=0.3,
        help="GPU memory fraction for vLLM (0.0-1.0). Higher = faster generation but less memory for training.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name or path to use.",
    )

    parser.add_argument(
        "--out-dir-suffix",
        type=str,
        required=True,
        help="Suffix to append to output directory name.",
    )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Group size for GRPO training.",
    )

    parser.add_argument(
        "--gen-batchsize",
        type=int,
        default=None,
        help="Number of generation groups to process in parallel during rollout.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dapo",
        choices=["dapo", "math_constraints"],
        help="Dataset to use: dapo (DAPO-Math-17k) or math_constraints (data/train/math_with_constraints.json).",
    )
    parser.add_argument(
        "--no-constraint",
        dest="no_constraint",
        action="store_true",
        help="When using math_constraints dataset, omit constraint descriptions from the user prompt.",
    )
    parser.add_argument(
        "--math-constraints-path",
        type=str,
        default="data/train/math_with_constraints.json",
        help="Path to math_with_constraints JSON when --dataset math_constraints.",
    )
    parser.add_argument(
        "--dynamic-g",
        dest="dynamic_g",
        action="store_true",
        help="Enable dynamic num_generations via UpdateGroupSizeCallback.",
    )

    parser.add_argument(
        "--per-dev-batch-size",
        dest="per_dev_batch_size",
        type=int,
        default=2,
        help="Per-device training batch size.",
    )

    parser.add_argument(
        "--grad-accum-steps",
        dest="grad_accum_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps for training.",
    )

    parser.add_argument(
        "--kl-coeff",
        dest="kl_coeff",
        type=float,
        default=0.001,
        help="KL divergence coefficient for GRPO.",
    )

    # HuggingFace Hub arguments for async model upload
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace API token for uploading models. Can also set HF_TOKEN env var.",
    )
    parser.add_argument(
        "--hf-username",
        dest="hf_username",
        type=str,
        default=os.environ.get("HF_USERNAME", ""),
        help="HuggingFace username/namespace for uploading models. Can also set HF_USERNAME env var.",
    )
    parser.add_argument(
        "--hf-max-uploads",
        dest="hf_max_uploads",
        type=int,
        default=4,
        help="Maximum number of checkpoint uploads to HuggingFace during training.",
    )
    parser.add_argument(
        "--hf-private",
        dest="hf_private",
        action="store_true",
        default=True,
        help="Make uploaded HuggingFace repos private (default: True).",
    )
    parser.add_argument(
        "--no-hf-upload",
        dest="no_hf_upload",
        action="store_true",
        help="Disable automatic HuggingFace upload of checkpoints.",
    )

    # run example:
    # python src/main.py --model-name Qwen/Qwen2.5-1.5B-Instruct --out-dir-suffix GRPO-Math-Think --num-generations 8 --dynamic-g --use-vllm

    args = parser.parse_args()
    utils.NUM_GENERATIONS = args.num_generations

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Gemma ends turns with <end_of_turn>, not <eos>. TRL treats completions that
    # do not end in tokenizer.eos_token_id as truncated and masks them from the
    # loss, so align eos with the model's real turn-ender. No-op for Llama/Qwen.
    try:
        from transformers import GenerationConfig as _GC
        _turn_end = "<end_of_turn>"
        _te_id = tokenizer.convert_tokens_to_ids(_turn_end)
        _gen_eos = _GC.from_pretrained(model_name).eos_token_id
        if isinstance(_gen_eos, int):
            _gen_eos = [_gen_eos]
        if (isinstance(_te_id, int) and _te_id >= 0 and _gen_eos
                and _te_id in _gen_eos and tokenizer.eos_token_id != _te_id):
            print(f"aligned eos_token to {_turn_end} (id {_te_id})")
            tokenizer.eos_token = _turn_end
    except Exception as exc:
        print(f"eos alignment skipped: {exc}")

    # If DeepSpeed is enabled, let TRL/Accelerate own model instantiation/placement.
    # Otherwise, explicitly load with FlashAttention 2 and keep model/device consistent for generation.

    # resolved_optim, ds_cfg = setup_pkgs(args)

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
        print(f"Using specified optimizer: {args.optim}")
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

    if ds_cfg:
        model = model_name
    else:
        try:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("Loading model with HF")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",  # Use PyTorch SDPA instead of flash_attention_2
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            if hasattr(model, "config"):
                model.config.use_cache = True
        except Exception:
            model = model_name

    output_dir = f"../models/{model_name.replace('/', '_')}-GRPO-{args.out_dir_suffix}"

    stop_strings, stop_token_ids = build_stop_toks(tokenizer)

    # Stop on every eos the model declares (generation config), not just the
    # tokenizer default, so models like Gemma stop on <end_of_turn>.
    from transformers import GenerationConfig
    eos_ids = set(int(t) for t in stop_token_ids)
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    try:
        gen_eos = GenerationConfig.from_pretrained(model_name).eos_token_id
        if isinstance(gen_eos, int):
            gen_eos = [gen_eos]
        for e in (gen_eos or []):
            eos_ids.add(int(e))
    except Exception as exc:
        print(f"could not read generation_config eos_token_id: {exc}")
    generation_kwargs = {
        "max_tokens": int(L_MAX),
        "ignore_eos": False,
        "stop_token_ids": sorted(eos_ids),
    }

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=args.per_dev_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        generation_batch_size=args.gen_batchsize,
        save_strategy="no",
        num_generations=args.num_generations,
        max_prompt_length=256,
        max_completion_length=L_MAX,
        bf16=True,
        gradient_checkpointing=True,
        beta=args.kl_coeff,
        remove_unused_columns=False,
        save_total_limit=8,
        deepspeed=ds_cfg,
        optim=resolved_optim,
        mask_truncated_completions=True,
        generation_kwargs=generation_kwargs,
        # Avoid transformers hub-kernels auto-install (kernels-community/flash-attn2),
        # which may not have a compatible prebuilt binary for this torch/cuda combo.
        # Use the standard attention implementation instead.
        model_init_kwargs={"attn_implementation": "sdpa"},  # Use PyTorch SDPA
    )

    if args.vllm_mode:
        training_args.use_vllm = True
        training_args.vllm_mode = args.vllm_mode
        training_args.vllm_server_port = 8002
        training_args.vllm_gpu_memory_utilization = args.vllm_gpu_memory

        # training_args.vllm_importance_sampling_correction = False

    # Initialize metrics infra
    utils.TOKENIZER_FOR_METRICS = tokenizer
    initialize_metrics(output_dir, args.num_generations)

    # Reset metrics files and in-memory state
    reset_metrics_files()
    utils._GROUP_COUNTER = 0
    utils._GROUP_BUFFER.clear()
    # Set expected group size to num_generations for true intra-group metrics
    utils._EXPECTED_G = int(getattr(training_args, "num_generations", 8) or 8)
    if args.dataset == "math_constraints":
        dataset = get_grpo_math_constraints_dataset(
            data_path=args.math_constraints_path,
            no_constraint=args.no_constraint,
        )
    else:
        dataset = get_dapo_math_dataset()

    # Gemma's chat template rejects a standalone system role; fold it into the
    # first user turn. Llama/Qwen accept a system role and are left unchanged.
    def _supports_system_role(tok):
        try:
            tok.apply_chat_template(
                [{"role": "system", "content": "x"},
                 {"role": "user", "content": "y"}],
                tokenize=False,
            )
            return True
        except Exception:
            return False

    if not _supports_system_role(tokenizer):
        print("folding system prompt into user turn")

        def _fold_system(example):
            msgs = example["prompt"]
            if msgs and msgs[0].get("role") == "system":
                sys_txt = msgs[0]["content"]
                rest = msgs[1:]
                if rest and rest[0].get("role") == "user":
                    rest = [{"role": "user",
                             "content": sys_txt + "\n\n" + rest[0]["content"]}] + rest[1:]
                else:
                    rest = [{"role": "user", "content": sys_txt}] + rest
                example["prompt"] = rest
            return example

        dataset = dataset.map(_fold_system)

    # NOTE: Some TRL versions accept `reward_input_keys`, others don't.
    # The dataset columns are still forwarded to custom reward functions via **kwargs either way.
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        reward_funcs=[accuracy_reward_func, format_reward_func, finstruction_reward_func],
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    desired_reward_input_keys = [
        "question",
        "answer",
        "constraint_desc",
        "constraint_name",
        "constraint_args",
    ]
    if "reward_input_keys" in inspect.signature(GRPOTrainer.__init__).parameters:
        trainer_kwargs["reward_input_keys"] = desired_reward_input_keys
    else:
        print(
            "Note: installed TRL does not support `reward_input_keys` in GRPOTrainer; "
            "dataset columns will still be forwarded to reward functions via **kwargs."
        )

    trainer = GRPOTrainer(**trainer_kwargs)

    # set_generation_config(trainer, tokenizer, args)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    models_to_set = [unwrapped]
    if hasattr(trainer, "ref_model") and trainer.ref_model is not None:
        models_to_set.append(trainer.accelerator.unwrap_model(trainer.ref_model))
    for m in models_to_set:
        if hasattr(m, "generation_config"):
            gen_cfg = m.generation_config
            # Early stop via EOS if available
            print(gen_cfg)

        else:
            print(
                "Warning: model does not have generation_config attribute; skipping generation config setup."
            )
    # Dynamic group-size chooser based on control_step_pd
    # Initialize Persistent Controller
    controller = LagrangianRiskController(allowed_sizes=ALLOWED_G)
    controller.__post_init__()

    def _choose_next_g() -> int:
        """Choose next num_generations using in-memory completed groups.

        Single-process version: drains SHARED_GROUPS and runs control_step_pd.
        """
        # Current G as fallback
        cur_g = int(utils.NUM_GENERATIONS)

        # Drain local queue
        groups: list[list[int]] = []
        try:
            while SHARED_GROUPS:
                groups.append(SHARED_GROUPS.popleft())
        except Exception:
            pass

        next_g = cur_g

        try:
            info = control_step_pd(
                groups=groups,
                controller=controller,
                current_G=cur_g,
                L_max=L_MAX,
                delta=DELTA,
            )
            next_g = int(info.get("G_next", cur_g))
        except Exception as e:
            print(f"control_step_pd failed: {e}")
            next_g = cur_g

        # Sync metrics grouping to avoid partial buffered groups
        utils._EXPECTED_G = next_g
        utils.NUM_GENERATIONS = next_g
        try:
            utils._GROUP_BUFFER.clear()
        except Exception:
            pass

        return int(next_g)

    if args.dynamic_g:
        updater = UpdateGroupSizeCallback(trainer, _choose_next_g)
        trainer.add_callback(updater)

    trainer.add_callback(
        TimeTriggeredActions(
            checkpoint_every_s=10**9,  # disable periodic checkpointing
            clock_log_path=f"{output_dir}/metrics/clock.jsonl",
            enable_eval=False,
        )
    )
    
    # Add async HuggingFace upload callback if credentials provided
    if not args.no_hf_upload and args.hf_token and args.hf_username:
        print(f"[main] Enabling async HuggingFace upload to {args.hf_username}'s account")
        print(f"[main] Dataset: {args.dataset}, G: {args.num_generations}, Max uploads: {args.hf_max_uploads}")
        hf_callback = AsyncPushToHubCallback(
            base_model=model_name,
            dataset=args.dataset,
            hf_token=args.hf_token,
            hf_username=args.hf_username,
            num_generations=args.num_generations,
            max_uploads=args.hf_max_uploads,
            private=args.hf_private,
            delete_after_upload=True,  # Save disk space
        )
        trainer.add_callback(hf_callback)
    elif not args.no_hf_upload:
        if not args.hf_token:
            print("[main] Warning: --hf-token not provided, skipping HuggingFace upload")
        if not args.hf_username:
            print("[main] Warning: --hf-username not provided, skipping HuggingFace upload")
    
    # trainer.add_callback(StepTimerCallback())
    trainer.train()
    # Ensure final model is saved separately from checkpoints
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
