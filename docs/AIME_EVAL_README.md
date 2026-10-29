# AIME Evaluation Pipeline

This pipeline evaluates mathematical reasoning models on AIME (American Invitational Mathematics Examination) problems from 2023-2025 using the OpenAI API.

## Features

- **Automatic Dataset Loading**: Loads AIME problems from CSV and converts to OpenAI message format
- **Structured Prompting**: Uses system and user prompts designed for step-by-step reasoning
- **Answer Extraction**: Robust extraction using math_verify library with regex fallbacks
- **Per-Year Evaluation**: Reports accuracy for each year (2023, 2024, 2025) separately
- **OpenAI Integration**: Easy integration with OpenAI models (GPT-4, GPT-4o, etc.)

## Installation

```bash
# Install required packages
pip install openai pandas tqdm math-verify
```

## Usage

### Basic Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Run evaluation
python eval_aime.py
```

### Command Line Arguments

```bash
python eval_aime.py \
    --csv_path data/eval/aime_23_to_25.csv \
    --model gpt-4o \
    --temperature 0.7 \
    --max_tokens 2048
```

### Programmatic Usage

```python
from eval_aime import evaluate_aime_dataset

# Run evaluation
results = evaluate_aime_dataset(
    csv_path='data/eval/aime_23_to_25.csv',
    api_key='your-api-key',  # or use OPENAI_API_KEY env var
    model='gpt-4o',
    temperature=0.7,
    max_tokens=2048
)

# Access results
print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
for year, acc in results['year_accuracies'].items():
    print(f"Year {year}: {acc:.4f}")
```

### Quick Test

Test the pipeline without making API calls:

```bash
python run_aime_eval.py --test
```

## Dataset Format

The CSV file should have the following columns:
- `problem`: The problem statement
- `answer`: The correct answer (as a number, may include leading zeros)
- `year`: Year of the AIME exam (2023, 2024, or 2025)
- `problem_idx`: Problem number within that year

## Prompt Structure

### System Prompt
Defines the model as an expert mathematical problem solver with emphasis on rigorous reasoning.

### User Prompt
Provides:
1. The problem statement
2. Clear instructions for step-by-step solving
3. Requirement to output answer in `\boxed{}` format

Example:
```
Solve the following problem step by step. Think carefully through each step of your reasoning.

Problem: [PROBLEM TEXT]

Instructions:
1. Read the problem carefully and identify what is being asked.
2. Break down the problem into smaller, manageable steps.
3. Show all your work, including equations, calculations, and logical reasoning.
4. Verify your answer makes sense in the context of the problem.
5. The answer is always a number (it may be an integer, decimal, or fraction).
6. Conclude your solution by providing the final numerical answer inside \boxed{}.
```

## Answer Extraction

The `extract_answer()` function uses a multi-strategy approach:

1. **Primary**: math_verify library with LaTeX and expression extraction
2. **Fallback patterns** (in order):
   - `\boxed{...}` environments
   - `\framebox{...}` environments
   - LaTeX expressions at end of text `$...$` or `$$...$$`
   - "Answer:" or "Final answer:" patterns
   - Last number in the text

## Output

The evaluation produces:

### Console Output
```
==============================================================
EVALUATION RESULTS
==============================================================

Overall Accuracy: 0.3567 (52/146)

Accuracy by Year:
--------------------------------------------------------------
Year 2023: 0.3400 (17/50)
Year 2024: 0.3667 (22/60)
Year 2025: 0.3611 (13/36)
==============================================================
```

### JSON Output
Results are saved to `aime_evaluation_results.json`:
```json
{
  "overall_accuracy": 0.3567,
  "year_accuracies": {
    "2023": 0.34,
    "2024": 0.3667,
    "2025": 0.3611
  },
  "all_predictions": ["191", "881", ...],
  "all_targets": ["191", "881", ...]
}
```

## Functions

### `load_aime_dataset(csv_path: str) -> List[Dict]`
Loads CSV and converts to OpenAI format with metadata.

### `create_reasoning_prompt(problem: str) -> str`
Creates the user prompt with instructions.

### `extract_answer(answer: str) -> str`
Extracts numerical answer from model output.

### `get_openai_response(client, messages, model, temperature, max_tokens) -> str`
Gets response from OpenAI API.

### `evaluate_aime_dataset(csv_path, api_key, model, temperature, max_tokens) -> Dict`
Main evaluation function that processes all problems and returns results.

## Tips

1. **Model Selection**: GPT-4 and GPT-4o perform best on mathematical reasoning
2. **Temperature**: Lower (0.3-0.5) for more consistent answers, higher (0.7-1.0) for more creative approaches
3. **Max Tokens**: AIME problems often require detailed solutions, recommend 2048+
4. **Cost Management**: Process a subset first to estimate costs before full evaluation
5. **Rate Limits**: The pipeline includes error handling but may need delays for rate limits

## Error Handling

- API errors are caught and empty strings are returned
- Failed extractions return empty strings
- All errors are logged to console

## License

This code is part of the GRPO project.
