---
description: Check if any skill files differ from their cached versions
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

Read all skills from `<available_skills>` in the system prompt. For each one:

1. Get the SKILL.md path from `<location>` in `<available_skills>`.
2. Read the file from disk with the `read` tool.
3. Extract the on-disk body: remove YAML frontmatter (from first `---` through closing `---`, inclusive), trim whitespace.
4. Extract the cached body: take `<skill_content>`, remove everything up to and including the first blank line after `# Skill:`, trim whitespace.
5. Report as stale if the two bodies differ.

Output:

```
[SKILL: <name>] ✗ (STALE)
[SKILL: <name>] ✓ (fresh)
```

If any are stale, tell the user to restart OpenCode (`opencode reload` or close/reopen) to refresh the cache.
