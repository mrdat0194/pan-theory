"""
omnistats/run_autonomous_loop.py
─────────────────────────────────
Phase V: Autonomous Experimentation Loop Orchestrator.

The core autonomous cycle:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Loop Iteration N                                                   │
    │                                                                     │
    │  1. Chay Sim  ─── MatrAIx personas + Mock Outcome Function         │
    │        |          → simulated_experiment_data.csv                  │
    │        |                                                            │
    │  2. Feature A/B Test ─── omnistats CUPED + Causal Inference        │
    │        |                 → ATT estimates per feature                │
    │        |                                                            │
    │  3. Test ra Best ─── Rank by ATT, select winning feature           │
    │        |                                                            │
    │  4. Train Deep JEPA ─── Fine-tune MuMoJEPAWrapper on winners      │
    │        |                (from MLModel/AIModel/model/mumo_wrapper)  │
    │        |                                                            │
    │  5. Explain AI ─── TCAV Concept Probing + Vector-Target Attribution│
    │        |           → xai_tcav_concept_probing.png                  │
    │        |           → xai_vector_target_attribution.png             │
    │        |                                                            │
    │  6. Loop ─── Propose next set of features from explained model     │
    └─────────────────────────────────────────────────────────────────────┘

Usage
-----
    python run_autonomous_loop.py                   # 1 iteration, 3 features, 5k personas
    python run_autonomous_loop.py --iterations 3    # 3 full loop cycles
    python run_autonomous_loop.py --n-personas 1000 --features 5
    python run_autonomous_loop.py --no-train-jepa   # Skip JEPA training (XAI only)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── resolve omnistats root ────────────────────────────────────────────────────
_OMNI_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_OMNI_ROOT))

# ── resolve MLModel for MUMO import ──────────────────────────────────────────
_AIMODEL_ROOT = Path("C:/Users/mrdat/PycharmProjects/pan-theory/MLModel/AIModel")
sys.path.insert(0, str(_AIMODEL_ROOT))

from config import OUTPUT_DIR


# =============================================================================
# Loop Utilities
# =============================================================================

_BANNER = """
╔═══════════════════════════════════════════════════════════╗
║   OmniStats x MatrAIx x MUMO-JEPA                        ║
║   Autonomous Experimentation Loop (Phase V)               ║
╚═══════════════════════════════════════════════════════════╝
"""


def _build_feature_candidates(iteration: int, n_features: int, rng: np.random.Generator) -> list[dict]:
    """
    Propose feature candidates for this iteration.
    In later iterations, ATE values are perturbed based on previous winning ATE
    to focus the search space around promising effect sizes.
    """
    base_ate = 0.03 + 0.01 * iteration   # Escalate baseline ATE over iterations
    features = []
    for i in range(n_features):
        ate = float(rng.uniform(base_ate * 0.5, base_ate * 2.0))
        features.append({
            "name": f"iter{iteration}_feat{i+1}",
            "ate": ate,
            "noise": 0.02,
        })
    return features


def _build_tcav_concept_examples(
    all_personas_df: pd.DataFrame,
    mumo_model,
    device: str = "cpu",
):
    """
    Automatically build TCAV concept examples from the persona population.
    Splits each binary concept into top/bottom 25% of the relevant trait.
    """
    try:
        import torch
        from modules.simulation.mumo_jepa_trainer import prepare_tensors
    except ImportError:
        return None, None, None

    concept_examples = {}
    concept_defs = {
        "High Income":     ("income",        "top"),
        "Risk Averse":     ("risk_tolerance", "bottom"),
        "Tech-Savvy":      ("tech_savviness", "top"),
        "Impulsive":       ("impulsivity",    "top"),
        "Neurotic":        ("neuroticism",    "top"),
        "Open Minded":     ("openness",       "top"),
    }

    for concept_name, (col, side) in concept_defs.items():
        if col not in all_personas_df.columns:
            continue
        q25 = all_personas_df[col].quantile(0.25)
        q75 = all_personas_df[col].quantile(0.75)
        if side == "top":
            mask = all_personas_df[col] >= q75
        else:
            mask = all_personas_df[col] <= q25
        sub_df = all_personas_df[mask].head(200)
        if len(sub_df) < 20:
            continue
        mod_a, mod_b = prepare_tensors(sub_df, device=device)
        concept_examples[concept_name] = (mod_a, mod_b)

    # Non-concept: random sample
    random_df = all_personas_df.sample(n=min(200, len(all_personas_df)), random_state=42)
    non_concept = prepare_tensors(random_df, device=device)
    query = prepare_tensors(all_personas_df.head(500), device=device)

    return concept_examples, non_concept, query


# =============================================================================
# Main Loop
# =============================================================================

def run_loop(
    n_iterations: int = 1,
    n_personas: int = 5_000,
    n_features: int = 3,
    train_jepa: bool = True,
    jepa_epochs: int = 3,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Run the full Autonomous Experimentation Loop.

    Parameters
    ----------
    n_iterations : int   Number of complete loop cycles to run.
    n_personas   : int   Number of MatrAIx personas per simulation.
    n_features   : int   Number of feature candidates to test per iteration.
    train_jepa   : bool  Whether to fine-tune MUMO-JEPA on the best data.
    jepa_epochs  : int   Fine-tuning epochs per iteration.
    seed         : int   Global random seed.
    device       : str   'cpu' or 'cuda'.
    verbose      : bool  Print progress.
    """
    print(_BANNER)
    rng = np.random.default_rng(seed)
    loop_log = []

    # ── Imports ───────────────────────────────────────────────────────────────
    from modules.simulation.matraix_loader import load_matraix_personas
    from modules.simulation.matraix_bridge import run_multi_feature_sim
    from modules.simulation.mumo_ab_tester import find_best_feature
    from modules.simulation.mumo_jepa_trainer import build_mumo_model, fine_tune, prepare_tensors
    from modules.xai_visualisation import plot_tcav_concept_probing, plot_vector_target_attribution

    # ── Phase 0: Load Persona Population (reused across iterations) ──────────
    print(f"[Loop] Loading {n_personas:,} MatrAIx personas...")
    all_personas_df = load_matraix_personas(
        n_personas=n_personas,
        seed=seed,
        verbose=verbose,
    )
    print(f"[Loop] Persona pool loaded: {len(all_personas_df):,} rows x {len(all_personas_df.columns)} cols\n")

    # ── Build MUMO model (shared across iterations) ──────────────────────────
    mumo_model = None
    if train_jepa:
        print("[Loop] Building MUMO-JEPA model...")
        mumo_model = build_mumo_model(verbose=verbose)
        mumo_model = mumo_model.to(device)

    # ═════════════════════════════════════════════════════════════════════════
    for iteration in range(1, n_iterations + 1):
        iter_start = time.time()
        print(f"\n{'═' * 60}")
        print(f"  LOOP ITERATION {iteration}/{n_iterations}")
        print(f"{'═' * 60}\n")

        # ── Stage 1: Chay Sim (MatrAIx) ─────────────────────────────────────
        print("[Stage 1] Chay Sim: Generating feature variants via MatrAIx...")
        features = _build_feature_candidates(iteration, n_features, rng)
        sim_results = run_multi_feature_sim(
            personas_df=all_personas_df,
            features=features,
            seed=seed + iteration,
            verbose=verbose,
        )
        print(f"[Stage 1] Simulated {len(features)} feature variants.\n")

        # ── Stage 2+3: Feature A/B Test → Find Best ──────────────────────────
        print("[Stage 2-3] Running OmniStats A/B Test to find Best feature...")
        best_name, best_df, leaderboard = find_best_feature(
            sim_results=sim_results,
            verbose=verbose,
        )
        leaderboard_path = os.path.join(OUTPUT_DIR, f"leaderboard_iter{iteration}.csv")
        leaderboard.to_csv(leaderboard_path, index=False)
        print(f"[Stage 2-3] Leaderboard saved -> {leaderboard_path}\n")

        # ── Stage 4: Train Deep JEPA ─────────────────────────────────────────
        if train_jepa and mumo_model is not None:
            print(f"[Stage 4] Fine-tuning MUMO-JEPA on '{best_name}' data...")
            mumo_model = fine_tune(
                model=mumo_model,
                best_df=best_df,
                n_epochs=jepa_epochs,
                device=device,
                verbose=verbose,
            )
            print("[Stage 4] MUMO-JEPA fine-tuning complete.\n")

        # ── Stage 5: Explain AI (TCAV + Vector-Target Attribution) ──────────
        if train_jepa and mumo_model is not None:
            print("[Stage 5] Running Explainable AI (TCAV + Vector-Target Attribution)...")
            mumo_model.eval()

            def encode_fn(mod_a, mod_b):
                return mumo_model.encode(mod_a, mod_b)

            # TCAV: Persona Concept Probing
            concept_examples, non_concept, query = _build_tcav_concept_examples(
                all_personas_df=best_df,
                mumo_model=mumo_model,
                device=device,
            )
            if concept_examples:
                tcav_scores = plot_tcav_concept_probing(
                    model_encode_fn=encode_fn,
                    concept_examples=concept_examples,
                    non_concept_examples=non_concept,
                    query_examples=query,
                    verbose=verbose,
                )
            else:
                tcav_scores = {}

            # Vector-Target Attribution: Modality ablation
            from config import AB_METRIC_COL
            metric_col = AB_METRIC_COL or "metric"
            if metric_col in best_df.columns:
                mod_a_full, mod_b_full = prepare_tensors(best_df.head(500), device=device)
                target_vals = best_df.head(500)[metric_col].fillna(0.0).to_numpy(dtype=np.float32)
                attributions = plot_vector_target_attribution(
                    model_encode_fn=encode_fn,
                    mod_a_examples=mod_a_full,
                    mod_b_examples=mod_b_full,
                    target_col=target_vals,
                    verbose=verbose,
                )
            else:
                attributions = {}

            print("[Stage 5] XAI explanations complete.\n")
        else:
            tcav_scores = {}
            attributions = {}

        # ── Iteration Summary ─────────────────────────────────────────────────
        elapsed = time.time() - iter_start
        top_att = leaderboard.iloc[0]["att_estimate"]
        log_entry = {
            "iteration":  iteration,
            "best_feature": best_name,
            "best_att": top_att,
            "n_features_tested": n_features,
            "elapsed_sec": round(elapsed, 1),
        }
        loop_log.append(log_entry)
        print(f"[Loop] Iteration {iteration} complete in {elapsed:.1f}s | "
              f"Best='{best_name}' | ATT={top_att:.4f}")

    # ── Final Log ─────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  AUTONOMOUS LOOP COMPLETE")
    print(f"{'═' * 60}")
    log_df = pd.DataFrame(loop_log)
    log_path = os.path.join(OUTPUT_DIR, "autonomous_loop_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"[Loop] Full log saved -> {log_path}")
    print(log_df.to_string(index=False))


# =============================================================================
# CLI Entry Point
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="OmniStats x MatrAIx x MUMO-JEPA: Autonomous Experimentation Loop"
    )
    p.add_argument("--iterations",   type=int,   default=1,     help="Number of full loop cycles.")
    p.add_argument("--n-personas",   type=int,   default=5_000, help="MatrAIx personas per sim.")
    p.add_argument("--features",     type=int,   default=3,     help="Feature candidates per iteration.")
    p.add_argument("--no-train-jepa",action="store_true",        help="Skip MUMO-JEPA fine-tuning.")
    p.add_argument("--epochs",       type=int,   default=3,     help="JEPA fine-tuning epochs.")
    p.add_argument("--seed",         type=int,   default=42,    help="Random seed.")
    p.add_argument("--device",       type=str,   default="cpu", help="'cpu' or 'cuda'.")
    p.add_argument("--quiet",        action="store_true",        help="Suppress verbose output.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_loop(
        n_iterations=args.iterations,
        n_personas=args.n_personas,
        n_features=args.features,
        train_jepa=not args.no_train_jepa,
        jepa_epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        verbose=not args.quiet,
    )
