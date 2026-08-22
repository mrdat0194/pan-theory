"""
basic_eval.py — Sandbox Task testing built-in Solvers (Prompt Template, Chain-of-Thought, Self-Critique)
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
from inspect_ai.solver import prompt_template, chain_of_thought, generate, self_critique
from inspect_ai.scorer import match

DATASET_PATH = str(Path(__file__).parent / "dataset.json")

@task
def sandbox_basic_task():
    """
    Basic sandbox evaluation task.
    Pipelining:
      1. Format input prompt
      2. Apply Chain-of-Thought reasoning
      3. Generate response
      4. Perform Self-Critique loop
      5. Grade with exact match scorer
    """
    return Task(
        dataset=json_dataset(DATASET_PATH),
        plan=[
            prompt_template("Answer the following prompt step-by-step:\n\n{prompt}"),
            chain_of_thought(),
            generate(),
            self_critique(),
        ],
        scorer=match(location="any")
    )
