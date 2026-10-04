"""
omnistats/modules/xai/llm_xai_narrator.py
"""

def narrate(
    best_action: dict,
    tcav_scores_debiased: dict,
    opa_value: float = 0.0,
    llm_client = None
) -> str:
    """
    Returns APA narrative string for the optimal Off-Policy Evaluated plan.
    """
    top_concept = max(tcav_scores_debiased.items(), key=lambda x: x[1]) if tcav_scores_debiased else ("None", 0.0)
    
    narrative = (
        f"The optimal experiment design targets action:\n"
        f"  {best_action}\n"
        f"with Doubly Robust value: {opa_value:.4f}.\n"
        f"Dominant XAI concept: {top_concept[0]} (debiased TCAV: {top_concept[1]:.4f})."
    )
    return narrative
