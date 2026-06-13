---
description: Read-only review of all skills — frontmatter, naming, template vars, contradictions
mode: subagent
permission:
  read: allow
  glob: allow
  skill: allow
  edit: deny
  bash: deny
  grep: deny
  webfetch: deny
  question: deny
---

Load ALL skills from `<available_skills>`. For each skill, check:

1. **Frontmatter** — Does the SKILL.md have both `name` and `description`?
2. **Filename match** — Does `name` match the parent directory name?
3. **Template variables** — Does the body use template vars instead of hardcoded names?
4. **No contradictions** — Compare against every other skill for conflicting guidance.

Output:

```
[skill-name] ✓ synced
[skill-name] (⚠ stale) ✗ (found 2 issues)
  Issue 1: <description>
```

Do NOT edit any files. Read-only pass. Run once.
