"""
run_sandbox.py — Python entrypoint to execute Inspect AI eval sandbox tasks
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout encoding for Windows UTF-8 safety
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# Load API keys from external or service-agent secrets.env
external_env = Path("C:/Users/mrdat/gemini_secrets.env")
service_agent_env = Path(__file__).parent.parent / "service-agent" / "mcp_configs" / "secrets.env"

if external_env.exists():
    load_dotenv(dotenv_path=external_env, override=True)
    print(f"[INFO] Loaded secrets from external {external_env}")
elif service_agent_env.exists():
    load_dotenv(dotenv_path=service_agent_env, override=True)
    print(f"[INFO] Loaded secrets from service-agent {service_agent_env}")
else:
    load_dotenv(override=True)

if "GEMINI_API_KEY" in os.environ:
    key = os.environ["GEMINI_API_KEY"].strip()
    os.environ["GEMINI_API_KEY"] = key
    os.environ["GOOGLE_API_KEY"] = key
    print(f"[INFO] GEMINI_API_KEY set successfully ({key[:6]}...{key[-4:]}).")

from inspect_ai import eval
from basic_eval import sandbox_basic_task
from custom_scorer_eval import sandbox_custom_scorer_task
from guardrails_eval import sandbox_guardrails_task

def main():
    print("=" * 60)
    print(" INSPECT AI DEMO SANDBOX — RUNNER ")
    print("=" * 60)
    
    model_name = "google/gemini-3.6-flash"
    print(f"\n[1/3] Running Basic Evaluation Task with Model: {model_name}...")
    logs_basic = eval(sandbox_basic_task(), model=model_name)
    print(f"Basic Task complete. Log samples processed: {len(logs_basic[0].samples) if logs_basic else 0}")

    print(f"\n[2/3] Running Custom Scorer + LLM Judge Task with Model: {model_name}...")
    logs_custom = eval(sandbox_custom_scorer_task(), model=model_name)
    print(f"Custom Task complete. Log samples processed: {len(logs_custom[0].samples) if logs_custom else 0}")

    print(f"\n[3/3] Running Guardrails AI Validation Task with Model: {model_name}...")
    logs_guardrails = eval(sandbox_guardrails_task(), model=model_name)
    print(f"Guardrails Task complete. Log samples processed: {len(logs_guardrails[0].samples) if logs_guardrails else 0}")

    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETE! ")
    print(" Launch interactive web dashboard: inspect view")
    print("=" * 60)

if __name__ == "__main__":
    main()
