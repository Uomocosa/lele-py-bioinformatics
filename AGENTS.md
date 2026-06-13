# Project Rules

This workspace is Python-only (bioinformatics package). Skills under `~/.config/opencode/skills/`
are auto-discovered. `lele-syntax-py` is always loaded — it defines import rules,
code organization, testing conventions, and naming.

Use `pixi` for environment and testing (see pixi skill).

## Git Workflow
See `.agents/skills/git-workflow/SKILL.md` — loaded on demand for git tasks.

## Project Commands

Values shared across skills use the `[[AGENTS.md::KEY]]` convention.
Agents resolve these by reading this section.

- `[[AGENTS.md::RUN_ALL_TESTS]]`: `pixi run pytest`
- `[[AGENTS.md::BUILD]]`: `pixi run build`
- `[[AGENTS.md::LINT]]`: `pixi run ruff`

Contextual template variables (`{{package}}`, `{{Module}}`, etc.) are
resolved at runtime by the agent based on context — they are not defined here.

## CRITICAL: Commit Authorization

**Never stage, commit, push, merge, rebase, or amend without an explicit command from the user.**
An "explicit command" is a direct statement such as "commit", "stage that file",
"push to origin", or "merge the PR". Implied intent, prior agreement, or
"go ahead" + silence does not count. If unsure, ask.
