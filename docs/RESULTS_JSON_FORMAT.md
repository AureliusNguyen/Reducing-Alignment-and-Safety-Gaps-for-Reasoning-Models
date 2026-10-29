# Results JSON Format

## Overview

The evaluation results now include the **full model responses** for each problem, allowing you to analyze how the model arrived at its answers.

## JSON Structure

```json
{
  "overall_accuracy": 0.3562,
  "year_accuracies": {
    "2023": 0.34,
    "2024": 0.3667,
    "2025": 0.3611
  },
  "all_predictions": ["191", "881", "607", ...],
  "all_targets": ["191", "881", "607", ...],
  "results_by_year": {
    "2023": {
      "accuracy": 0.34,
      "correct": 17,
      "total": 50,
      "problems": [
        {
          "problem_idx": 1,
          "problem": "Five men and nine women stand equally spaced...",
          "predicted": "191",
          "target": "191",
          "correct": true,
          "model_response": "Let me solve this step by step.\n\n1. First, I need to understand..."
        },
        {
          "problem_idx": 2,
          "problem": "Positive real numbers $b \\neq 1$ and $n$ satisfy...",
          "predicted": "880",
          "target": "881",
          "correct": false,
          "model_response": "I'll work through this problem carefully.\n\n..."
        }
      ]
    },
    "2024": {
      "accuracy": 0.3667,
      "correct": 22,
      "total": 60,
      "problems": [...]
    },
    "2025": {
      "accuracy": 0.3611,
      "correct": 13,
      "total": 36,
      "problems": [...]
    }
  }
}
```

## Fields Explained

### Top Level

| Field | Type | Description |
|-------|------|-------------|
| `overall_accuracy` | float | Overall accuracy across all years |
| `year_accuracies` | dict | Accuracy for each year (2023, 2024, 2025) |
| `all_predictions` | list[str] | All predicted answers in order |
| `all_targets` | list[str] | All target answers in order |
| `results_by_year` | dict | Detailed results organized by year |

### Per Year

| Field | Type | Description |
|-------|------|-------------|
| `accuracy` | float | Accuracy for this year |
| `correct` | int | Number of correct answers |
| `total` | int | Total number of problems |
| `problems` | list[dict] | Detailed problem-by-problem results |

### Per Problem

| Field | Type | Description |
|-------|------|-------------|
| `problem_idx` | int | Problem number within the year |
| `problem` | str | Full problem statement |
| `predicted` | str | Extracted answer from model |
| `target` | str | Correct answer |
| `correct` | bool | Whether prediction matches target |
| `model_response` | str | **Full model output including reasoning** |

## Usage Examples

### Load and Analyze Results

```python
import json

# Load results
with open('aime_evaluation_results.json', 'r') as f:
    results = json.load(f)

# Overall stats
print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")

# Per year
for year, data in results['results_by_year'].items():
    print(f"\nYear {year}:")
    print(f"  Accuracy: {data['accuracy']:.2%}")
    print(f"  Correct: {data['correct']}/{data['total']}")
```

### Analyze Specific Problems

```python
# Get all wrong answers for 2023
year_2023 = results['results_by_year']['2023']
wrong_problems = [p for p in year_2023['problems'] if not p['correct']]

print(f"\nWrong answers in 2023: {len(wrong_problems)}")

# Look at first wrong answer
if wrong_problems:
    p = wrong_problems[0]
    print(f"\nProblem #{p['problem_idx']}:")
    print(f"Problem: {p['problem'][:100]}...")
    print(f"Predicted: {p['predicted']}")
    print(f"Correct: {p['target']}")
    print(f"\nModel's reasoning:")
    print(p['model_response'])
```

### Find Common Mistakes

```python
# Analyze where extraction might have failed
for year, data in results['results_by_year'].items():
    empty_predictions = [
        p for p in data['problems'] 
        if not p['predicted'] and p['model_response']
    ]
    
    if empty_predictions:
        print(f"\nYear {year}: {len(empty_predictions)} problems with empty predictions")
        for p in empty_predictions[:3]:  # Show first 3
            print(f"  Problem #{p['problem_idx']}")
            print(f"  Last 100 chars of response: ...{p['model_response'][-100:]}")
```

### Compare Predictions to Targets

```python
# Find off-by-one errors
for year, data in results['results_by_year'].items():
    for p in data['problems']:
        try:
            pred_num = int(p['predicted'])
            target_num = int(p['target'])
            
            if abs(pred_num - target_num) == 1:
                print(f"\nOff by 1 - Year {year}, Problem #{p['problem_idx']}")
                print(f"  Predicted: {pred_num}")
                print(f"  Target: {target_num}")
        except (ValueError, TypeError):
            pass  # Not numeric answers
```

### Extract Reasoning Patterns

```python
# Find problems where model used specific reasoning keywords
import re

keywords = ['boxed', 'therefore', 'thus', 'final answer']

for keyword in keywords:
    count = 0
    for year, data in results['results_by_year'].items():
        for p in data['problems']:
            if re.search(keyword, p['model_response'], re.IGNORECASE):
                count += 1
    
    print(f"'{keyword}' appears in {count} responses")
```

### Export to DataFrame

```python
import pandas as pd

# Flatten results into a DataFrame
rows = []
for year, data in results['results_by_year'].items():
    for p in data['problems']:
        rows.append({
            'year': year,
            'problem_idx': p['problem_idx'],
            'predicted': p['predicted'],
            'target': p['target'],
            'correct': p['correct'],
            'response_length': len(p['model_response'])
        })

df = pd.DataFrame(rows)

# Analyze
print(df.groupby('year')['correct'].mean())
print(df.groupby('correct')['response_length'].mean())
```

### Save Specific Wrong Answers for Review

```python
# Save all wrong answers to a separate file
wrong_answers = {}

for year, data in results['results_by_year'].items():
    wrong_answers[year] = [
        {
            'problem_idx': p['problem_idx'],
            'problem': p['problem'],
            'predicted': p['predicted'],
            'target': p['target'],
            'model_response': p['model_response']
        }
        for p in data['problems'] 
        if not p['correct']
    ]

with open('wrong_answers_detailed.json', 'w') as f:
    json.dump(wrong_answers, f, indent=2)

print(f"Saved detailed wrong answers to wrong_answers_detailed.json")
```

### Generate HTML Report

```python
# Create an HTML report with all responses
html = """
<!DOCTYPE html>
<html>
<head>
    <title>AIME Evaluation Results</title>
    <style>
        .correct { background-color: #d4edda; }
        .incorrect { background-color: #f8d7da; }
        .problem { margin: 20px; padding: 10px; border: 1px solid #ccc; }
        .response { white-space: pre-wrap; font-family: monospace; }
    </style>
</head>
<body>
    <h1>AIME Evaluation Results</h1>
"""

for year in sorted(results['results_by_year'].keys()):
    data = results['results_by_year'][year]
    html += f"<h2>Year {year} - Accuracy: {data['accuracy']:.2%}</h2>"
    
    for p in data['problems']:
        css_class = 'correct' if p['correct'] else 'incorrect'
        html += f"""
        <div class="problem {css_class}">
            <h3>Problem #{p['problem_idx']}</h3>
            <p><strong>Problem:</strong> {p['problem']}</p>
            <p><strong>Predicted:</strong> {p['predicted']}</p>
            <p><strong>Target:</strong> {p['target']}</p>
            <details>
                <summary>Model Response</summary>
                <div class="response">{p['model_response']}</div>
            </details>
        </div>
        """

html += """
</body>
</html>
"""

with open('evaluation_report.html', 'w') as f:
    f.write(html)

print("HTML report saved to evaluation_report.html")
```

## File Size Considerations

With full model responses included, the JSON file will be **significantly larger**:

- **Without responses**: ~50 KB
- **With responses**: ~5-20 MB (depending on response length)

This is normal and expected. The detailed responses are valuable for:
- Debugging extraction issues
- Understanding model reasoning
- Identifying systematic errors
- Training data for future improvements

## Tips

1. **Use `jq` for quick queries**:
   ```bash
   # Count correct answers in 2023
   cat results.json | jq '.results_by_year["2023"].correct'
   
   # Get all predicted answers
   cat results.json | jq '.all_predictions'
   
   # Find problems where predicted != target
   cat results.json | jq '.results_by_year["2023"].problems[] | select(.correct == false)'
   ```

2. **Stream large files**:
   ```python
   import json
   
   # For very large files, use ijson for streaming
   import ijson
   
   with open('large_results.json', 'rb') as f:
       for problem in ijson.items(f, 'results_by_year.2023.problems.item'):
           if not problem['correct']:
               print(problem['problem_idx'])
   ```

3. **Compress for storage**:
   ```bash
   gzip aime_evaluation_results.json
   # Creates aime_evaluation_results.json.gz (~1/10 the size)
   ```
