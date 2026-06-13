---
name: create-skill
description: Create reusable Python agent skills. Use when the user asks to create a new skill, make a skill for X, or set up agent instructions. ALWAYS evaluates existing alternatives before creating anything new. Enforces maximal generality so skills work across ANY Python project using this opencode setup.
disable-model-invocation: true
---

# Create Skill

You are a skill architect for Python. Your job is to create skills that are **maximally general across Python projects, maximally reusable**, and never project-specific.

## Phase 1: Community Discovery (MANDATORY)

**Do NOT create a skill until you have searched the community ecosystem.**

1. Search GitHub for relevant opencode skills: `anthropics/skills`, `vercel-labs/skills`, `mattpocock/skills`, `obra/superpowers`.

Filter results to Python-relevant skills only. For every match found, report to the user:
   - Skill name, source repo
   - What it does (summary)
   - Your evaluation: does it fully cover the need? Partially? Not at all?

**If an adequate existing skill exists:** Recommend it and stop. Do not create a duplicate.

**Only proceed to Phase 2 if** no existing skill adequately covers the need.

## Phase 2: Generality-first Design

Before writing any file, design the skill to be **project-agnostic**:

### Generality Rules (enforced)
1. **Zero project-specific identifiers.** No `bio/`, no package name, no module path from this project. Use template variables: `{{package}}`, `{{Module}}`, `{{function}}`, `{{project_name}}`.
2. **Description-first.** The `description` field must make sense in ANY Python project. Test: read the description aloud — if it mentions this project, rewrite.
3. **Single responsibility.** One skill = one domain. If the skill does two unrelated things, split it.
4. **Progressive disclosure.** The `name` + `description` must be enough for an agent to decide whether to load the skill. Put the most critical instructions first.
5. **`disable-model-invocation`** set to `true` for destructive or high-risk skills.

### Structure template

```
skill-name/
  SKILL.md            # YAML frontmatter + instructions
  references/         # Detailed docs, test prompts, edge cases
  scripts/            # Executable utilities (bash, python, etc.)
  assets/             # Templates, static files
```

## Phase 3: Drafting

Create the `SKILL.md` with:

```yaml
---
name: skill-name            # lowercase, hyphens, 1-64 chars
description: |              # 1-1024 chars, must trigger correctly
  Use when... [trigger conditions]. Works with any Python project.
# disable-model-invocation: true
---
```

Then the body. Structure it for how agents actually read:
- Most important instructions FIRST
- Step-by-step workflow
- Examples with concrete inputs/outputs
- Common mistakes and edge cases

## Phase 4: Validation

| Check | How |
|-------|-----|
| Name valid? | Lowercase, hyphens, 1-64 chars, matches directory name |
| Description triggers? | Read it cold — would an agent load this for the right task? |
| Zero project references? | No `bio/`, no `my_package` references |
| Reusable in other Python project? | Would this work if copied to a different Python repo unchanged? |
| Community check done? | Verifiable by user |
