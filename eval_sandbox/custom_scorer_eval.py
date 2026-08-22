"""
custom_scorer_eval.py — Sandbox Task testing custom Scorers & Model-Graded Judges
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load API keys from external or service-agent secrets.env
external_env = Path("C:/Users/mrdat/gemini_secrets.env")
service_agent_env = Path(__file__).parent.parent / "service-agent" / "mcp_configs" / "secrets.env"

if external_env.exists():
    load_dotenv(dotenv_path=external_env, override=True)
elif service_agent_env.exists():
    load_dotenv(dotenv_path=service_agent_env, override=True)
else:
    load_dotenv(override=True)

if "GEMINI_API_KEY" in os.environ:
    key = os.environ["GEMINI_API_KEY"].strip()
    os.environ["GEMINI_API_KEY"] = key
    os.environ["GOOGLE_API_KEY"] = key


from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import scorer, Score, accuracy, stderr, model_graded_qa

DATASET_PATH = str(Path(__file__).parent / "dataset.json")

@scorer(metrics=[accuracy(), stderr()])
def concise_length_scorer(max_words=60):
    """
    Custom deterministic scorer checking if output word count is under `max_words`.
    """
    async def score(state, target):
        text = state.output.completion or ""
        word_count = len(text.split())
        passed = word_count <= max_words
        return Score(
            value=1.0 if passed else 0.0,
            explanation=f"Word count: {word_count}/{max_words} (Limit: {max_words})"
        )
    return score

@task
def sandbox_custom_scorer_task():
    """
    Evaluation task combining a custom deterministic word-count scorer
    with an LLM-as-a-Judge (model_graded_qa).
    """
    return Task(
        dataset=json_dataset(DATASET_PATH),
        plan=[generate()],
        scorer=[
            concise_length_scorer(max_words=60),
            model_graded_qa()
        ]
    )
