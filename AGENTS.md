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
- `[[AGENTS.md::INTEGRATE_PAPER_SCRAPER]]`: `pixi run integrate_paper_scraper` (see README "Integrating `paper-scraper` data")

## Gotchas

- **`python -m bio.<module>` does NOT work.** The `new-python-import-system` lazy loader raises
  `CallableLoader has no attribute get_code`. Run entry points via a `pixi run <task>` or
  `pixi run python -c "from bio.<module> import main; main()"` — this applies to every `bio` module.
- **`bio` must stay editable.** `import bio` asserts `DATASETS/`, `VOCABULARIES/`, etc. exist
  (`bio/__global__.py`); a non-editable copy fails the assertion.
- **Joblib cache** lives in `.cache_dir`; editing a `@CACHE_MEMORY.cache` function auto-invalidates
  it. Delete the dir if you suspect stale featurization results.
- **`paper-scraper` data** belongs in `DATASETS/PDCC/paper_scraper/` (the `*_without_conflicts.csv`
  files + the two `paper_scraper_complete_*.json` dicts).
- **Windows `pixi install`** can fail with "Access is denied" on a directory rename (AV/indexer
  lock). Manually `Move-Item` the leftover `.<pkg>-<rand>` temp dir in the rattler cache to its final
  name and rerun; use `pixi run --frozen` once the env exists.

Contextual template variables (`{{package}}`, `{{Module}}`, etc.) are
resolved at runtime by the agent based on context — they are not defined here.

## CRITICAL: Commit Authorization

**Never stage, commit, push, merge, rebase, or amend without an explicit command from the user.**
An "explicit command" is a direct statement such as "commit", "stage that file",
"push to origin", or "merge the PR". Implied intent, prior agreement, or
"go ahead" + silence does not count. If unsure, ask.
