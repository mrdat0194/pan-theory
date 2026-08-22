# Inspect AI Evaluation Demo Sandbox (`eval_sandbox`)

Welcome to the **AI Evaluation Demo Sandbox**! This directory is a standalone playground designed to test and understand **Inspect AI** (developed by the UK AI Safety Institute & JJ Allaire) before integrating evals into the production [`service-agent`](../service-agent).

---

## 📁 Directory Structure

```text
eval_sandbox/
├── README.md               # This documentation file
├── dataset.json            # Toy benchmark dataset (3 test samples)
├── basic_eval.py           # Task 1: Prompt formatting, Chain-of-Thought & Self-Critique
├── custom_scorer_eval.py   # Task 2: Custom Python word-count scorer & LLM-as-a-Judge
├── guardrails_eval.py     # Task 3: Guardrails AI validation rules & output compliance scoring
└── run_sandbox.py          # Central runner script (executes tasks against Gemini 3.6 Flash)
```

---

## 🚀 How to Run the Evaluation

### Option 1: Execute Runner Script (Recommended)
Run the central runner script using Python:

```bash
python eval_sandbox/run_sandbox.py
```

This script automatically:
1. Loads secrets (`GEMINI_API_KEY`) from `service-agent/mcp_configs/secrets.env`.
2. Runs `sandbox_basic_task` (CoT + Self-Critique).
3. Runs `sandbox_custom_scorer_task` (Custom Scorer + LLM Judge).
4. Runs `sandbox_guardrails_task` (Guardrails AI output validation & compliance scoring).

---

### Option 2: Run via Inspect CLI directly
You can also run individual tasks directly using the `inspect` CLI tool:

```bash
# Run basic task
inspect eval eval_sandbox/basic_eval.py --model google/gemini-3.6-flash

# Run custom scorer task
inspect eval eval_sandbox/custom_scorer_eval.py --model google/gemini-3.6-flash

# Run Guardrails AI evaluation demo
inspect eval eval_sandbox/guardrails_eval.py --model google/gemini-3.6-flash
```

---

## 📊 Viewing Evaluation Results

### Interactive Web Dashboard (`inspect view`)
Inspect AI includes a built-in local web server to visually inspect prompt histories, model thinking monologues, tool calls, and LLM judge explanations:

```bash
inspect view
```
Then open your browser to **`http://localhost:7575`**.

---

## 💡 How the Code Works

### 1. Benchmark Dataset (`dataset.json`)
Defines evaluation test cases containing prompt inputs and ground truth targets:
```json
[
  {
    "input": "What is the capital of France?",
    "target": "Paris"
  }
]
```

### 2. Solver Pipelines (`basic_eval.py`)
Inspect AI solvers transform the `TaskState`. We chain solvers together into execution pipelines:
- `prompt_template(...)`: Formats the prompt.
- `chain_of_thought()`: Instructs the model to output step-by-step reasoning.
- `generate()`: Calls the model API.
- `self_critique()`: Asks the model to critique and refine its own output.

### 3. Custom Scorers & LLM Judges (`custom_scorer_eval.py`)
- **Custom Deterministic Scorer (`concise_length_scorer`)**: Evaluates word counts in Python, returning `Score(value=1.0)` if within limits.
- **LLM-as-a-Judge (`model_graded_qa()`)**: Uses a secondary LLM call to evaluate if the answer matches the target ground truth.
