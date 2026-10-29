import subprocess
import signal
import psutil
from eval_aime import evaluate_aime_dataset, load_aime_dataset
from typing import Optional
import os
from openai import AsyncOpenAI
import time
import asyncio

TP_SIZE = 1
VLLM_WAIT_TIME = 120  # seconds


def get_vllm_cmd(
    model,
    max_model_len,
    max_num_seqs,
    chat_template: Optional[str] = None,
    enable_reasoning_parser: bool = False,
):

    gpu_mem_util = 0.92
    cmd = f"vllm serve {model} --max-model-len {max_model_len} --dtype bfloat16 --no-enable-prefix-caching --max-num-seqs {max_num_seqs} --enable-prompt-tokens-details --gpu-memory-utilization {gpu_mem_util} --tensor-parallel-size {TP_SIZE}"
    if chat_template is not None:
        cmd += f" --chat-template '{chat_template}'"
    if enable_reasoning_parser:
        cmd += " --reasoning-parser deepseek_r1"
    return cmd


# run the command using subprocess and capture the output while also check process return code
def run_vllm_command(command):
    """
    Run a shell command in the background.
    The command's stdout and stderr will be inherited from the parent process.
    Returns the subprocess.Popen object representing the running process.
    """
    # Using shell=True as vllm_cmd is a formatted string command.
    # Ensure the command string is from a trusted source if shell=True is used.
    process = subprocess.Popen(command, shell=True)
    return process


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        print(f"Parent process {parent_pid} does not exist.")
        return
    children = parent.children(recursive=True)
    for chd in children:
        chd.send_signal(sig)
    # Also kill the parent process
    print(f"Killing parent process {parent_pid}")
    parent.send_signal(sig)


async def run_full_evaluation(
    csv_path: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: Optional[str] = None,
    output_path="aime_evaluation_results.json",
    max_concurrent: int = 10,
):
    """
    Run full evaluation on all problems using async parallel requests.

    Args:
        csv_path: Path to CSV file
        model: Model name
        temperature: Temperature for generation
        max_tokens: Max tokens to generate
        api_key: OpenAI API key (if using OpenAI)
        output_path: Where to save results
        max_concurrent: Max concurrent requests
    """

    # Initialize OpenAI client
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    base_url = None if api_key else "http://localhost:8000/v1"
    final_api_key = api_key if api_key else "EMPTY"

    # Create async client
    if base_url:
        client = AsyncOpenAI(base_url=base_url, api_key=final_api_key)
    else:
        client = AsyncOpenAI(api_key=final_api_key)

    # Run async evaluation
    results = await evaluate_aime_dataset(
        csv_path=csv_path,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        client=client,
        output_path=output_path,
        max_concurrent=max_concurrent,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AIME evaluation with vLLM.")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the AIME problems CSV file.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name to use for evaluation."
    )
    parser.add_argument(
        "--temperature", type=float, required=True, help="Temperature for the model."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=True,
        help="Max tokens for the model response.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, will try to use vLLM server.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="aime_evaluation_results.json",
        help="Path to save evaluation results.",
    )

    parser.add_argument(
        "--start_vllm",
        action="store_true",
        help="Whether to start a vLLM server before evaluation.",
    )

    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="Maximum number of sequences for vLLM server.",
    )

    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Custom chat template to use with vLLM server.",
    )

    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10).",
    )

    args = parser.parse_args()

    vllm_process = None
    if args.start_vllm:
        if args.max_num_seqs is None:
            raise ValueError("max_num_seqs must be provided when starting vLLM server.")

        vllm_cmd = get_vllm_cmd(
            model=args.model,
            max_model_len=args.max_tokens + 1024,
            max_num_seqs=args.max_num_seqs,
            chat_template=args.chat_template,
        )

        print(f"Starting vLLM server with command: {vllm_cmd}")
        vllm_process = run_vllm_command(vllm_cmd)
        time.sleep(VLLM_WAIT_TIME)  # Give it some time to start
        print(f"vLLM server started with PID: {vllm_process.pid}")

    results = asyncio.run(
        run_full_evaluation(
            csv_path=args.csv_path,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            output_path=args.output_path,
            max_concurrent=args.max_concurrent,
        )
    )

    if args.start_vllm and vllm_process is not None:
        print("Killing vLLM server...")
        kill_child_processes(vllm_process.pid)
        print("vLLM server killed.")


# run example with vllm
# python src/run_vllm.py --csv_path data/eval/aime_23_to_25.csv --model Qwen/Qwen2.5-1.5B-Instruct --temperature 0.7 --max_tokens 8000 --start_vllm --max_num_seqs 128 --output_path aime_vllm_results.json --max_concurrent 128
# python src/run_vllm.py --csv_path data/eval/aime_23_to_25.csv --model /work/nvme/bdgi/ahmed1/grpo/preliminary/Qwen2.5-1.5B-Instruct-GRPO-Math-Think/checkpoint-1764 --temperature 0.7 --max_tokens 8000 --start_vllm --max_num_seqs 128 --output_path aime_vllm_results.json --max_concurrent 128
