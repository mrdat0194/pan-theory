---
name: sota-git-workflow
description: Activates whenever the user asks to commit code, create a branch, push to remote, or perform any Git version control tasks.
---
# SOTA Git Workflow

When performing Git operations in this repository, you MUST follow these rules:

1. **Staging Discipline:**
   - Never use `git add .` blindly. Check `git status` first.
   - Only stage files relevant to the current logical change.
   - Never stage lockfiles, `.env` files, or temporary testing scripts unless explicitly requested.

2. **Branching Strategy:**
   - Always check current branch before working.
   - When asked to create a new branch, use the format: `<type>/<short-description>` (e.g., `feat/add-fista-solver`, `fix/omnistats-lpa-bug`).

3. **Conventional Commits:**
   Commit messages must follow the Conventional Commits specification:
   `<type>(<scope>): <description>`
   
   *Types:*
   - `feat`: A new feature
   - `fix`: A bug fix
   - `docs`: Documentation only changes
   - `style`: Changes that do not affect the meaning of the code
   - `refactor`: A code change that neither fixes a bug nor adds a feature
   - `perf`: A code change that improves performance
   - `test`: Adding missing tests or correcting existing tests
   
   *Example:* `feat(omnistats): add CUPED variance reduction module`

4. **Pre-commit Verification:**
   - Ensure the code has no glaring syntax errors before committing. 
   - Write a detailed commit body if the change is complex (explaining the "Why" and "What").

5. **No Force Pushes:**
   - Never run `git push --force` or `-f` unless the user explicitly commands it in all caps.
