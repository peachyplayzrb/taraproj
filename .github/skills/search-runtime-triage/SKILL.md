---
name: search-runtime-triage
description: "Use when triaging Flask runtime issues in app.py, search endpoint behavior, artifact loading, and click-store persistence."
---

# Search Runtime Triage Skill

Triage Flow
1. Confirm required artifacts exist and load.
2. Check /search returns expected fields.
3. Check /click persists updates.
4. Validate selected sort mode behavior quickly.

Guidelines
- Prefer smallest reproducible checks.
- Separate data issues from logic issues.
- Report precise file/symbol locations.
