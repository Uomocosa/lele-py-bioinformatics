# Project Rules

This workspace is Python-only (bioinformatics package). Skills under `.agents/skills/`
are auto-discovered. `project-conventions` is always loaded — it defines import rules,
code organization, testing conventions, and naming.

Use `pixi` for environment and testing (see pixi skill).

## Git Workflow
See `.agents/skills/git-workflow/SKILL.md` — loaded on demand for git tasks.

## CRITICAL: Commit Authorization

**Never stage, commit, push, merge, rebase, or amend without an explicit command from the user.**
An "explicit command" is a direct statement such as "commit", "stage that file",
"push to origin", or "merge the PR". Implied intent, prior agreement, or
"go ahead" + silence does not count. If unsure, ask.
