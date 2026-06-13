---
name: remove-dead-code
description: Find .py files whose primary pub item has zero internal consumers in the package. Detect removal candidates by searching for import references, then delete with verification.
---

## Goal
Find and delete files whose primary `pub` item has zero internal consumers in the codebase.

## Detection Method
For each `.py` file that defines a class or top-level function:

1. Identify the item name.
2. Search the package for `from bio.` or `import bio.` references to that name.
3. Exclude the file's own imports and its module's `__init__.py` re-export.
4. If zero external references remain, the file is a removal candidate.

## Exemptions
- Items explicitly intended as public API for external consumers.
- Do not remove public API surface without explicit confirmation.

## Verification
After removal, run:
```
pixi run pytest
```
