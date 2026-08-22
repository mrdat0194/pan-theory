"""
guardrails_eval.py — Sandbox Task evaluating Guardrails AI (Validation & Scoring) inside Inspect AI
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
from inspect_ai.scorer import scorer, Score, accuracy, stderr

# Guardrails AI Imports
from guardrails import Guard
from guardrails.validator_base import Validator, ValidationResult, PassResult, FailResult, register_validator

DATASET_PATH = str(Path(__file__).parent / "dataset.json")

# Define a custom Guardrails Validator for demonstration
@register_validator(name="concise_sentence_count", data_type="string")
class ConciseSentenceCount(Validator):
    """
    Guardrails AI custom validator that ensures completion is concise (<= 3 sentences).
    """
    def __init__(self, max_sentences: int = 3, **kwargs):
        super().__init__(max_sentences=max_sentences, **kwargs)
        self.max_sentences = max_sentences

    def validate(self, value: str, metadata: dict) -> ValidationResult:
        if not value:
            return FailResult(error_message="Completion is empty.")
        
        # Simple sentence count estimation
        sentences = [s.strip() for s in value.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) > self.max_sentences:
            return FailResult(
                error_message=f"Output has {len(sentences)} sentences, exceeding limit of {self.max_sentences}."
            )
        return PassResult()

# Build Guard instance with validation rules
guardrails_validator = Guard().use(ConciseSentenceCount(max_sentences=3))

@scorer(metrics=[accuracy(), stderr()])
def guardrails_compliance_scorer():
    """
    Custom Inspect AI Scorer that evaluates LLM completions against Guardrails AI rules.
    """
    async def score(state, target):
        completion = state.output.completion or ""
        
        try:
            # Validate output using Guardrails Guard
            res = guardrails_validator.validate(completion)
            validation_passed = getattr(res, "validation_passed", True)
            error_message = getattr(res, "error", None) or getattr(res, "validation_summaries", "")
            
            explanation = (
                f"Guardrails AI Validation PASSED."
                if validation_passed
                else f"Guardrails AI Validation FAILED: {error_message}"
            )
            return Score(
                value=1.0 if validation_passed else 0.0,
                explanation=explanation
            )
        except Exception as e:
            return Score(
                value=0.0,
                explanation=f"Guardrails AI Validation Exception: {str(e)}"
            )
    return score

@task
def sandbox_guardrails_task():
    """
    Evaluation task testing LLM output compliance against Guardrails AI rules.
    """
    return Task(
        dataset=json_dataset(DATASET_PATH),
        plan=[generate()],
        scorer=[guardrails_compliance_scorer()]
    )
