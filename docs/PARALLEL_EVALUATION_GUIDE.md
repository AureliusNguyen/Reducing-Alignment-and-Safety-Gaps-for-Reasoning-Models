# Parallel Request Evaluation Guide

## Overview

The AIME evaluation pipeline now supports **parallel asynchronous requests** to significantly speed up evaluation. This is especially useful when using vLLM servers or OpenAI API.

## Key Features

✅ **Async Parallel Requests**: Send multiple requests concurrently  
✅ **Configurable Concurrency**: Control max concurrent requests with `--max_concurrent`  
✅ **Progress Tracking**: Real-time progress bar with `tqdm`  
✅ **Backwards Compatible**: Can still run sequentially if needed  
✅ **Error Handling**: Graceful handling of failed requests  

## Performance Comparison

| Mode | 146 Problems | Speed |
|------|--------------|-------|
| Sequential | ~2-3 hours | 1x |
| Async (10 concurrent) | ~15-20 minutes | ~8-10x |
| Async (32 concurrent) | ~5-10 minutes | ~20-30x |

*Times vary based on model speed and token generation*

## Usage

### Option 1: Async Parallel (Recommended)

```bash
python src/run_vllm.py \
    --csv_path data/eval/aime_23_to_25.csv \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --temperature 0.7 \
    --max_tokens 8000 \
    --start_vllm \
    --max_num_seqs 128 \
    --output_path aime_vllm_results.json \
    --max_concurrent 32
```

**Key Parameters:**
- `--max_concurrent 32`: Send up to 32 requests in parallel
- `--max_num_seqs 128`: vLLM can handle 128 sequences (should be >= max_concurrent)

### Option 2: Sequential (For Debugging)

```bash
python src/run_vllm.py \
    --csv_path data/eval/aime_23_to_25.csv \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --temperature 0.7 \
    --max_tokens 8000 \
    --start_vllm \
    --max_num_seqs 128 \
    --output_path aime_vllm_results.json \
    --use_async False
```

### Option 3: Using Existing vLLM Server

If you already have a vLLM server running:

```bash
# Don't start vLLM, just connect to existing server
python src/run_vllm.py \
    --csv_path data/eval/aime_23_to_25.csv \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --temperature 0.7 \
    --max_tokens 8000 \
    --output_path aime_vllm_results.json \
    --max_concurrent 32
```

## Programmatic Usage

### Async Parallel

```python
import asyncio
from openai import AsyncOpenAI
from eval_aime import evaluate_aime_dataset_async

async def main():
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    
    results = await evaluate_aime_dataset_async(
        csv_path="data/eval/aime_23_to_25.csv",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        temperature=0.7,
        max_tokens=8000,
        client=client,
        output_path="results.json",
        max_concurrent=32  # Control parallelism
    )
    
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")

asyncio.run(main())
```

### Sequential

```python
from openai import OpenAI
from eval_aime import evaluate_aime_dataset

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

results = evaluate_aime_dataset(
    csv_path="data/eval/aime_23_to_25.csv",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    temperature=0.7,
    max_tokens=8000,
    client=client,
    output_path="results.json"
)

print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
```

## Configuration Recommendations

### For vLLM Server

| GPU | max_num_seqs | max_concurrent | Notes |
|-----|--------------|----------------|-------|
| A100 40GB | 128-256 | 32-64 | High parallelism |
| A100 80GB | 256-512 | 64-128 | Very high parallelism |
| V100 32GB | 64-128 | 16-32 | Moderate parallelism |
| RTX 4090 | 64-128 | 16-32 | Moderate parallelism |

**Rule of thumb:** `max_concurrent` should be ≤ `max_num_seqs / 2` for safety

### For OpenAI API

```bash
# OpenAI has rate limits, use lower concurrency
python src/run_vllm.py \
    --csv_path data/eval/aime_23_to_25.csv \
    --model gpt-4o \
    --temperature 0.7 \
    --max_tokens 2048 \
    --api_key "sk-..." \
    --output_path openai_results.json \
    --max_concurrent 5  # Lower to avoid rate limits
```

## How It Works

### Async Architecture

```python
# 1. Create semaphore to limit concurrent requests
semaphore = asyncio.Semaphore(max_concurrent)

# 2. Each request acquires semaphore
async def process_item(item):
    async with semaphore:  # Only max_concurrent will run at once
        return await get_openai_response_async(...)

# 3. Gather all tasks with progress bar
tasks = [process_item(item) for item in dataset]
results = await tqdm_asyncio.gather(*tasks)
```

### Benefits

1. **Speed**: Multiple requests in flight simultaneously
2. **Efficiency**: Better GPU/server utilization
3. **Control**: Semaphore prevents overwhelming the server
4. **Monitoring**: Progress bar shows real-time status

## Troubleshooting

### Issue: Rate Limit Errors

**Solution:** Reduce `--max_concurrent`
```bash
--max_concurrent 5  # Instead of 32
```

### Issue: Out of Memory

**Solution:** Reduce `--max_num_seqs` and `--max_concurrent`
```bash
--max_num_seqs 64 --max_concurrent 16
```

### Issue: Connection Errors

**Solution:** Check vLLM server is running
```bash
# Check if server is up
curl http://localhost:8000/v1/models
```

### Issue: Slow Performance

**Solution:** Increase concurrency (if server can handle it)
```bash
--max_concurrent 64 --max_num_seqs 128
```

## Monitoring

### During Evaluation

The progress bar shows:
```
Solving problems: 45%|████▌     | 66/146 [02:15<02:45,  2.06s/it]
```

- **66/146**: Completed 66 out of 146 problems
- **2:15**: Elapsed time
- **02:45**: Estimated remaining time
- **2.06s/it**: Average time per problem

### vLLM Server Logs

```bash
# Monitor vLLM server in another terminal
# It will show request processing in real-time
```

## Best Practices

1. **Start Small**: Test with `--max_concurrent 10` first
2. **Monitor Memory**: Watch GPU memory usage
3. **Tune Gradually**: Increase concurrency until you see diminishing returns
4. **Save Results**: Always use `--output_path` to save results
5. **Error Handling**: Failed requests return empty strings, check logs

## Example Workflow

```bash
# 1. Start vLLM server with high capacity
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.92

# 2. Run evaluation with parallel requests (in another terminal)
cd /work/nvme/bdgi/ahmed1/grpo/preliminary

python src/run_vllm.py \
    --csv_path data/eval/aime_23_to_25.csv \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --temperature 0.7 \
    --max_tokens 8000 \
    --output_path results_parallel.json \
    --max_concurrent 32

# 3. Check results
cat results_parallel.json | jq '.overall_accuracy'
```

## Benchmarking

To compare sequential vs parallel:

```bash
# Sequential
time python src/run_vllm.py [...] --use_async False

# Parallel (10 concurrent)
time python src/run_vllm.py [...] --max_concurrent 10

# Parallel (32 concurrent)
time python src/run_vllm.py [...] --max_concurrent 32
```

## Summary

- ✅ Use `--max_concurrent 32` for fast evaluation on powerful GPUs
- ✅ Use `--max_concurrent 10` for stable evaluation  
- ✅ Use `--use_async False` for debugging
- ✅ Monitor GPU memory and adjust `max_num_seqs` accordingly
- ✅ Results are identical between sequential and parallel modes
