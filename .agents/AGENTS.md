# pan-theory Code Guidelines and Alignment

1. **Coding Standards:**
   - Always use Python 3.x standards.
   - For mathematical modeling, prioritize the custom Convex Optimization solvers (FISTA, ADMM) found in `MLModel/model/` over generic `scikit-learn` implementations where applicable.
   - For experiments, always adhere to the 5-stage OmniStats pipeline.

2. **Skill Delegation:**
   - Do not attempt to guess workflows for Git, OmniStats, or Google APIs. Always defer to the specialized skills located in `.agents/skills/`.

3. **Documentation:**
   - Maintain existing docstrings and inline comments. Do not delete educational or structural comments without explicit permission.
