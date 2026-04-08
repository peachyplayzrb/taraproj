## Source
`school_docs/files/validate_customization.py`

## Summary
Governance validator for `.github/` control surfaces. Checks YAML frontmatter schema for agents (`.agent.md`), instructions (`.instructions.md`), prompts (`.prompt.md`), skills (`SKILL.md`), and hooks (`hooks.json`). Reports missing required fields and prints a pass/fail summary. Currently passing with 0 errors.

Validated fields per type:
- Agents: `name`, `description`
- Instructions: `applyTo`
- Prompts: `description`
- Skills: `name`, `description`
- Hooks: `hookId`, `event`, `runOn`

## Relevance to Implementation
**None.** This is tooling/workflow infrastructure, not information retrieval content. It has no impact on the search engine runtime.

## Implement Now
No.

## If Yes — Proposed Change
N/A.

## Confidence
High
